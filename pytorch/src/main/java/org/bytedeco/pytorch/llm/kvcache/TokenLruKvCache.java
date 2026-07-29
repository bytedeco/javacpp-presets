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
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Token-level LRU KV cache under a fixed per-sequence budget.
 *
 * <p>Each sequence stores a list of per-token K/V ({@code [H,D]} per layer).
 * On overflow, the least-recently-<em>accessed</em> non-protected tokens are
 * dropped (protected = first {@code protectSink} tokens, never evicted).
 * {@link #gather} returns a compact contiguous tensor of survivors.
 *
 * <p>Access order: {@link #gather} and {@link #touch} refresh LRU; append adds
 * new tokens as most-recent.
 */
public class TokenLruKvCache implements KvCache {

    public final LongAdder appendCount = new LongAdder();
    public final LongAdder evictCount = new LongAdder();

    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int budget;
    private final int protectSink;
    private final TensorOptions options;
    private final AtomicLong nextId = new AtomicLong(1);
    private final ConcurrentHashMap<Long, Seq> seqs = new ConcurrentHashMap<>();

    public TokenLruKvCache(int numLayers, int numHeads, int headDim, int budget, int protectSink) {
        this(numLayers, numHeads, headDim, budget, protectSink, new TensorOptions(torch.ScalarType.Float));
    }

    public TokenLruKvCache(int numLayers, int numHeads, int headDim, int budget, int protectSink,
                           TensorOptions options) {
        if (numLayers <= 0 || numHeads <= 0 || headDim <= 0 || budget <= 0) {
            throw new IllegalArgumentException("invalid dims/budget");
        }
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.budget = budget;
        this.protectSink = Math.max(0, protectSink);
        this.options = options;
    }

    @Override
    public long createSequence() {
        long id = nextId.getAndIncrement();
        seqs.put(id, new Seq());
        return id;
    }

    @Override
    public void releaseSequence(long seqId) {
        Seq s = seqs.remove(seqId);
        if (s != null) {
            s.clear();
        }
    }

    @Override
    public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
        Seq s = require(seqId);
        if (kLayers == null || kLayers.length != numLayers) {
            throw new IllegalArgumentException("kLayers length must equal numLayers");
        }
        synchronized (s) {
            int tokens = inferTokens(kLayers[0]);
            for (int t = 0; t < tokens; t++) {
                TokenSlot slot = new TokenSlot(numLayers, s.totalAppended);
                for (int L = 0; L < numLayers; L++) {
                    slot.k[L] = sliceToken(kLayers[L], t).contiguous().clone();
                    slot.v[L] = sliceToken(vLayers[L], t).contiguous().clone();
                }
                s.tokens.put(slot.absPos, slot); // LinkedHashMap access-order
                s.totalAppended++;
                appendCount.increment();
                evictIfNeeded(s);
            }
        }
    }

    /** Refresh LRU for existing positions (e.g. after attending them). */
    public void touch(long seqId, int[] positions) {
        Seq s = require(seqId);
        synchronized (s) {
            for (int p : positions) {
                TokenSlot slot = s.tokens.get((long) p);
                if (slot != null) {
                    s.tokens.get(slot.absPos); // access-order touch
                }
            }
        }
    }

    @Override
    public Tensor[] gather(long seqId, int layer) {
        Seq s = require(seqId);
        synchronized (s) {
            if (layer < 0 || layer >= numLayers) {
                throw new IllegalArgumentException("layer");
            }
            List<Tensor> ks = new ArrayList<>();
            List<Tensor> vs = new ArrayList<>();
            // Iterate in absolute position order for causal attention, not LRU order
            List<TokenSlot> ordered = new ArrayList<>(s.tokens.values());
            ordered.sort((a, b) -> Long.compare(a.absPos, b.absPos));
            for (TokenSlot slot : ordered) {
                s.tokens.get(slot.absPos); // LRU touch on gather
                ks.add(slot.k[layer]);
                vs.add(slot.v[layer]);
            }
            if (ks.isEmpty()) {
                Tensor e = zeros(new long[]{0, numHeads, headDim}, options);
                return new Tensor[]{e, e};
            }
            // each slot.k is [H,D] → stack to [T,H,D]
            TensorVector kv = new TensorVector();
            TensorVector vv = new TensorVector();
            for (Tensor t : ks) {
                kv.push_back(t.unsqueeze(0));
            }
            for (Tensor t : vs) {
                vv.push_back(t.unsqueeze(0));
            }
            return new Tensor[]{cat(kv, 0), cat(vv, 0)};
        }
    }

    @Override
    public int sequenceLength(long seqId) {
        return require(seqId).totalAppended;
    }

    @Override
    public int retainedLength(long seqId) {
        return require(seqId).tokens.size();
    }

    @Override
    public int numLayers() {
        return numLayers;
    }

    public int budget() {
        return budget;
    }

    public int protectSink() {
        return protectSink;
    }

    /** Absolute positions currently retained, sorted. */
    public long[] retainedPositions(long seqId) {
        Seq s = require(seqId);
        synchronized (s) {
            return s.tokens.keySet().stream().mapToLong(Long::longValue).sorted().toArray();
        }
    }

    @Override
    public void close() {
        for (Seq s : seqs.values()) {
            s.clear();
        }
        seqs.clear();
    }

    private void evictIfNeeded(Seq s) {
        while (s.tokens.size() > budget) {
            // Evict eldest access-order entry that is not a protected sink
            Iterator<Map.Entry<Long, TokenSlot>> it = s.tokens.entrySet().iterator();
            boolean removed = false;
            while (it.hasNext()) {
                Map.Entry<Long, TokenSlot> e = it.next();
                if (e.getValue().absPos < protectSink) {
                    continue;
                }
                it.remove();
                evictCount.increment();
                removed = true;
                break;
            }
            if (!removed) {
                // all protected — force-evict eldest anyway beyond sink
                it = s.tokens.entrySet().iterator();
                if (it.hasNext()) {
                    it.next();
                    it.remove();
                    evictCount.increment();
                } else {
                    break;
                }
            }
        }
    }

    private Seq require(long seqId) {
        Seq s = seqs.get(seqId);
        if (s == null) {
            throw new IllegalArgumentException("unknown seq " + seqId);
        }
        return s;
    }

    private static int inferTokens(Tensor t) {
        if (t.dim() == 2) {
            return 1; // [H,D]
        }
        if (t.dim() == 3) {
            // [T,H,D] or [1,H,D]
            return (int) t.size(0);
        }
        if (t.dim() == 4) {
            return (int) t.size(2); // [B,H,T,D] take T (B must be 1)
        }
        throw new IllegalArgumentException("unsupported k/v rank " + t.dim());
    }

    private Tensor sliceToken(Tensor t, int tIdx) {
        if (t.dim() == 2) {
            return t; // [H,D]
        }
        if (t.dim() == 3) {
            return t.select(0, tIdx); // [T,H,D] → [H,D]
        }
        if (t.dim() == 4) {
            // [B,H,T,D]: select B=0 → [H,T,D], select dim1 (T) → [H,D]
            return t.select(0, 0).select(1, tIdx);
        }
        throw new IllegalArgumentException("rank");
    }

    private static final class TokenSlot {
        final long absPos;
        final Tensor[] k;
        final Tensor[] v;

        TokenSlot(int layers, long absPos) {
            this.absPos = absPos;
            this.k = new Tensor[layers];
            this.v = new Tensor[layers];
        }
    }

    private static final class Seq {
        // access-order LRU
        final LinkedHashMap<Long, TokenSlot> tokens = new LinkedHashMap<>(16, 0.75f, true);
        int totalAppended = 0;

        void clear() {
            tokens.clear();
            totalAppended = 0;
        }
    }
}
