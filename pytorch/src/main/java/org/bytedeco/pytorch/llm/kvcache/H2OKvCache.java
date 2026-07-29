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

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
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
import static org.bytedeco.pytorch.global.torch.topk;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * H2O Heavy-Hitter Oracle KV cache (Zhang et al.).
 *
 * <p>Budget {@code = heavy + recent}. After each step the cache keeps:
 * <ul>
 *   <li>top-{@code heavyBudget} tokens by <b>cumulative attention mass</b></li>
 *   <li>the most recent {@code recentBudget} tokens</li>
 * </ul>
 * Survivors are compacted in absolute position order for standard SDPA.
 *
 * <p>Use {@link #appendWithScores} (or {@link #accumulateScores} after plain
 * {@link #append}) with masses from {@link org.bytedeco.pytorch.llm.modules.H2OAttention}.
 */
public class H2OKvCache implements KvCache {

    public final LongAdder appendCount = new LongAdder();
    public final LongAdder evictCount = new LongAdder();
    public final LongAdder compressCount = new LongAdder();

    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int heavyBudget;
    private final int recentBudget;
    private final TensorOptions options;
    private final AtomicLong nextId = new AtomicLong(1);
    private final ConcurrentHashMap<Long, Seq> seqs = new ConcurrentHashMap<>();

    public H2OKvCache(int numLayers, int numHeads, int headDim, int heavyBudget, int recentBudget) {
        this(numLayers, numHeads, headDim, heavyBudget, recentBudget,
                new TensorOptions(torch.ScalarType.Float));
    }

    public H2OKvCache(int numLayers, int numHeads, int headDim, int heavyBudget, int recentBudget,
                      TensorOptions options) {
        if (numLayers <= 0 || numHeads <= 0 || headDim <= 0) {
            throw new IllegalArgumentException("invalid dims");
        }
        if (heavyBudget < 0 || recentBudget <= 0) {
            throw new IllegalArgumentException("heavyBudget>=0 and recentBudget>0 required");
        }
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.heavyBudget = heavyBudget;
        this.recentBudget = recentBudget;
        this.options = options;
    }

    public int budget() {
        return heavyBudget + recentBudget;
    }

    public int heavyBudget() {
        return heavyBudget;
    }

    public int recentBudget() {
        return recentBudget;
    }

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

    @Override
    public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
        appendWithScores(seqId, kLayers, vLayers, null);
    }

    /**
     * @param scores optional {@code [Tk]} or {@code [B,Tk]} mass for <em>all</em> positions
     *               after this append (full-seq cumulative). If null, only recent window
     *               is enforced until scores arrive via {@link #accumulateScores}.
     */
    public void appendWithScores(long seqId, Tensor[] kLayers, Tensor[] vLayers, Tensor scores) {
        Seq s = require(seqId);
        if (kLayers == null || kLayers.length != numLayers) {
            throw new IllegalArgumentException("kLayers length");
        }
        synchronized (s) {
            int tokens = inferTokens(kLayers[0]);
            for (int t = 0; t < tokens; t++) {
                for (int L = 0; L < numLayers; L++) {
                    s.k.get(L).add(sliceToken(kLayers[L], t).contiguous().clone());
                    s.v.get(L).add(sliceToken(vLayers[L], t).contiguous().clone());
                }
                s.score.add(0.0);
                s.totalAppended++;
                appendCount.increment();
            }
            if (scores != null && scores.defined()) {
                applyScores(s, scores);
            }
            compress(s);
        }
    }

    /** Add / replace cumulative mass {@code [Tk]} aligned with current full length before compress. */
    public void accumulateScores(long seqId, Tensor scores) {
        Seq s = require(seqId);
        synchronized (s) {
            applyScores(s, scores);
            compress(s);
        }
    }

    private void applyScores(Seq s, Tensor scores) {
        Tensor flat = scores.dim() > 1 ? scores.reshape(-1) : scores;
        int n = (int) Math.min(flat.numel(), s.score.size());
        for (int i = 0; i < n; i++) {
            s.score.set(i, flat.select(0, i).item().toDouble());
        }
    }

    private void compress(Seq s) {
        int n = s.score.size();
        int budget = budget();
        if (n <= budget) {
            return;
        }
        compressCount.increment();

        // recent: last recentBudget indices always kept
        boolean[] keep = new boolean[n];
        int recentStart = Math.max(0, n - recentBudget);
        for (int i = recentStart; i < n; i++) {
            keep[i] = true;
        }

        // heavy: top-heavyBudget by score among non-recent
        if (heavyBudget > 0 && recentStart > 0) {
            int cand = recentStart;
            Tensor scoreT = zeros(new long[]{cand}, new TensorOptions(torch.ScalarType.Float));
            for (int i = 0; i < cand; i++) {
                scoreT.narrow(0, i, 1).fill_(new Scalar(s.score.get(i)));
            }
            int k = Math.min(heavyBudget, cand);
            T_TensorTensor_T top = topk(scoreT, k, -1L, true, true);
            Tensor idx = top.get1();
            for (int j = 0; j < k; j++) {
                int pos = (int) idx.select(0, j).item().toLong();
                keep[pos] = true;
            }
        }

        // compact
        List<List<Tensor>> newK = new ArrayList<>(numLayers);
        List<List<Tensor>> newV = new ArrayList<>(numLayers);
        List<Double> newScore = new ArrayList<>();
        for (int L = 0; L < numLayers; L++) {
            newK.add(new ArrayList<>());
            newV.add(new ArrayList<>());
        }
        int dropped = 0;
        for (int i = 0; i < n; i++) {
            if (!keep[i]) {
                dropped++;
                continue;
            }
            for (int L = 0; L < numLayers; L++) {
                newK.get(L).add(s.k.get(L).get(i));
                newV.get(L).add(s.v.get(L).get(i));
            }
            newScore.add(s.score.get(i));
        }
        s.k = newK;
        s.v = newV;
        s.score = newScore;
        evictCount.add(dropped);
    }

    @Override
    public Tensor[] gather(long seqId, int layer) {
        Seq s = require(seqId);
        synchronized (s) {
            if (layer < 0 || layer >= numLayers) {
                throw new IllegalArgumentException("layer");
            }
            List<Tensor> ks = s.k.get(layer);
            if (ks.isEmpty()) {
                Tensor e = zeros(new long[]{0, numHeads, headDim}, options);
                return new Tensor[]{e, e};
            }
            TensorVector kv = new TensorVector();
            TensorVector vv = new TensorVector();
            for (int i = 0; i < ks.size(); i++) {
                kv.push_back(ks.get(i).unsqueeze(0));
                vv.push_back(s.v.get(layer).get(i).unsqueeze(0));
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
        Seq s = require(seqId);
        synchronized (s) {
            return s.score.size();
        }
    }

    @Override
    public int numLayers() {
        return numLayers;
    }

    /** Current per-token scores (retained order). */
    public double[] scores(long seqId) {
        Seq s = require(seqId);
        synchronized (s) {
            double[] out = new double[s.score.size()];
            for (int i = 0; i < out.length; i++) {
                out[i] = s.score.get(i);
            }
            return out;
        }
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
        if (t.dim() == 2) {
            return 1;
        }
        if (t.dim() == 3) {
            return (int) t.size(0);
        }
        if (t.dim() == 4) {
            return (int) t.size(2);
        }
        throw new IllegalArgumentException("rank " + t.dim());
    }

    private static Tensor sliceToken(Tensor t, int tIdx) {
        if (t.dim() == 2) {
            return t;
        }
        if (t.dim() == 3) {
            return t.select(0, tIdx);
        }
        if (t.dim() == 4) {
            return t.select(0, 0).select(1, tIdx);
        }
        throw new IllegalArgumentException("rank");
    }

    private static final class Seq {
        List<List<Tensor>> k;
        List<List<Tensor>> v;
        List<Double> score = new ArrayList<>();
        int totalAppended = 0;

        Seq(int layers) {
            k = new ArrayList<>(layers);
            v = new ArrayList<>(layers);
            for (int i = 0; i < layers; i++) {
                k.add(new ArrayList<>());
                v.add(new ArrayList<>());
            }
        }
    }
}
