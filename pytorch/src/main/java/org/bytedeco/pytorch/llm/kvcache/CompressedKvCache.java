/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.kvcache;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Compressed / MLA latent KV cache for DeepSeek-style Multi-Latent Attention.
 *
 * <p>Stores low-rank latent {@code c_kv} ({@code [kvLoraRank]}) and optional
 * decoupled RoPE key {@code k_rope} ({@code [qkRopeHeadDim]}) per token instead
 * of full multi-head K/V. Pairs with
 * {@link org.bytedeco.pytorch.llm.modules.MultiLatentAttention#forwardCached}.
 *
 * <p>{@link #append} expects {@code kLayers[L] = c_kv token(s)} and
 * {@code vLayers[L] = k_rope token(s)} (naming follows the KvCache pair slots).
 * {@link #gather} returns {@code {c_kv [T, rank], k_rope [T, ropeDim]}}.
 *
 * <p>For multi-layer MLA each layer has its own latent stream (same rank).
 */
public class CompressedKvCache implements KvCache {

    public final LongAdder appendCount = new LongAdder();

    private final int numLayers;
    private final int kvLoraRank;
    private final int qkRopeHeadDim;
    private final int maxLen;
    private final TensorOptions options;
    private final AtomicLong nextId = new AtomicLong(1);
    private final ConcurrentHashMap<Long, Seq> seqs = new ConcurrentHashMap<>();

    public CompressedKvCache(int numLayers, int kvLoraRank, int qkRopeHeadDim, int maxLen) {
        this(numLayers, kvLoraRank, qkRopeHeadDim, maxLen, new TensorOptions(torch.ScalarType.Float));
    }

    public CompressedKvCache(int numLayers, int kvLoraRank, int qkRopeHeadDim, int maxLen,
                             TensorOptions options) {
        if (numLayers <= 0 || kvLoraRank <= 0 || maxLen <= 0) {
            throw new IllegalArgumentException("invalid dims");
        }
        this.numLayers = numLayers;
        this.kvLoraRank = kvLoraRank;
        this.qkRopeHeadDim = Math.max(0, qkRopeHeadDim);
        this.maxLen = maxLen;
        this.options = options;
    }

    public int kvLoraRank() { return kvLoraRank; }
    public int qkRopeHeadDim() { return qkRopeHeadDim; }
    public int maxLen() { return maxLen; }

    @Override
    public long createSequence() {
        long id = nextId.getAndIncrement();
        seqs.put(id, new Seq(numLayers));
        return id;
    }

    @Override
    public void releaseSequence(long seqId) {
        seqs.remove(seqId);
    }

    /**
     * @param kLayers per-layer c_kv tokens: {@code [rank]}, {@code [T,rank]}, or {@code [B,T,rank]}
     * @param vLayers per-layer k_rope tokens: {@code [ropeDim]} / {@code [T,ropeDim]} (may be null/empty if ropeDim=0)
     */
    @Override
    public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
        Seq s = require(seqId);
        if (kLayers == null || kLayers.length != numLayers) {
            throw new IllegalArgumentException("kLayers length must equal numLayers");
        }
        synchronized (s) {
            int tokens = inferTokens(kLayers[0]);
            for (int t = 0; t < tokens; t++) {
                if (s.len >= maxLen) {
                    for (int L = 0; L < numLayers; L++) {
                        s.cKv.get(L).remove(0);
                        if (qkRopeHeadDim > 0) {
                            s.kRope.get(L).remove(0);
                        }
                    }
                    s.len--;
                }
                for (int L = 0; L < numLayers; L++) {
                    s.cKv.get(L).add(sliceLatent(kLayers[L], t, kvLoraRank).contiguous().clone());
                    if (qkRopeHeadDim > 0) {
                        if (vLayers == null || vLayers[L] == null) {
                            s.kRope.get(L).add(zeros(new long[]{qkRopeHeadDim}, options));
                        } else {
                            s.kRope.get(L).add(sliceLatent(vLayers[L], t, qkRopeHeadDim).contiguous().clone());
                        }
                    }
                }
                s.len++;
                s.totalAppended++;
                appendCount.increment();
            }
        }
    }

    /**
     * @return {@code {c_kv [T, rank], k_rope [T, ropeDim]}} (k_rope may be empty rank-0 dim if ropeDim=0)
     */
    @Override
    public Tensor[] gather(long seqId, int layer) {
        Seq s = require(seqId);
        synchronized (s) {
            if (layer < 0 || layer >= numLayers) {
                throw new IllegalArgumentException("layer");
            }
            List<Tensor> cs = s.cKv.get(layer);
            if (cs.isEmpty()) {
                Tensor ec = zeros(new long[]{0, kvLoraRank}, options);
                Tensor er = zeros(new long[]{0, Math.max(1, qkRopeHeadDim)}, options);
                return new Tensor[]{ec, er};
            }
            TensorVector cv = new TensorVector();
            for (Tensor t : cs) {
                cv.push_back(t.unsqueeze(0));
            }
            Tensor cCat = cat(cv, 0);
            if (qkRopeHeadDim <= 0) {
                return new Tensor[]{cCat, zeros(new long[]{cs.size(), 1}, options)};
            }
            TensorVector rv = new TensorVector();
            for (Tensor t : s.kRope.get(layer)) {
                rv.push_back(t.unsqueeze(0));
            }
            return new Tensor[]{cCat, cat(rv, 0)};
        }
    }

    /** Convenience: gather as past tensors for MLA ({@code pastCkv}, {@code pastKr}) with batch dim. */
    public Tensor[] gatherForMla(long seqId, int layer) {
        Tensor[] g = gather(seqId, layer); // [T, rank], [T, rope]
        // MultiLatentAttention expects past along seq — return [1,T,rank] style if useful
        return new Tensor[]{g[0].unsqueeze(0), g[1].unsqueeze(0)}; // [1,T,*]
    }

    @Override
    public int sequenceLength(long seqId) {
        return require(seqId).totalAppended;
    }

    @Override
    public int retainedLength(long seqId) {
        return require(seqId).len;
    }

    @Override
    public int numLayers() {
        return numLayers;
    }

    @Override
    public void close() {
        seqs.clear();
    }

    private Seq require(long id) {
        Seq s = seqs.get(id);
        if (s == null) {
            throw new IllegalArgumentException("unknown seq " + id);
        }
        return s;
    }

    private static int inferTokens(Tensor t) {
        if (t.dim() == 1) {
            return 1;
        }
        if (t.dim() == 2) {
            // [T, rank] or [rank] ambiguous — if size(-1) looks like feature, T = size(0)
            return (int) t.size(0);
        }
        if (t.dim() == 3) {
            return (int) t.size(1); // [B,T,rank]
        }
        throw new IllegalArgumentException("unsupported latent rank " + t.dim());
    }

    private static Tensor sliceLatent(Tensor t, int tIdx, int feat) {
        if (t.dim() == 1) {
            return t; // [feat]
        }
        if (t.dim() == 2) {
            // [T, feat]
            if (t.size(1) == feat || t.size(1) > 1) {
                return t.select(0, tIdx);
            }
            // [feat, ?] unlikely
            return t.select(0, tIdx);
        }
        if (t.dim() == 3) {
            return t.select(0, 0).select(0, tIdx); // [B,T,feat]
        }
        throw new IllegalArgumentException("rank");
    }

    private static final class Seq {
        final List<List<Tensor>> cKv;
        final List<List<Tensor>> kRope;
        int len;
        int totalAppended;

        Seq(int layers) {
            cKv = new ArrayList<>(layers);
            kRope = new ArrayList<>(layers);
            for (int i = 0; i < layers; i++) {
                cKv.add(new ArrayList<>());
                kRope.add(new ArrayList<>());
            }
        }
    }
}
