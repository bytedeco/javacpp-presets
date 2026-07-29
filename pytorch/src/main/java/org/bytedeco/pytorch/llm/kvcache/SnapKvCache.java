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
 * SnapKV-style cache compression (Li et al. / similar observation-window selection).
 *
 * <p>When length exceeds {@code maxRetained}, an observation window of the most
 * recent {@code obsWindow} tokens votes (via attention scores) for which older
 * keys to keep. Top-{@code maxRetained - obsWindow} older tokens by vote mass
 * are retained together with the full observation window.
 *
 * <p>If scores are unavailable, falls back to keeping the most recent
 * {@code maxRetained} tokens (prefix drop).
 */
public class SnapKvCache implements KvCache {

    public final LongAdder appendCount = new LongAdder();
    public final LongAdder compressCount = new LongAdder();
    public final LongAdder evictCount = new LongAdder();

    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int maxRetained;
    private final int obsWindow;
    private final TensorOptions options;
    private final AtomicLong nextId = new AtomicLong(1);
    private final ConcurrentHashMap<Long, Seq> seqs = new ConcurrentHashMap<>();

    public SnapKvCache(int numLayers, int numHeads, int headDim, int maxRetained, int obsWindow) {
        this(numLayers, numHeads, headDim, maxRetained, obsWindow,
                new TensorOptions(torch.ScalarType.Float));
    }

    public SnapKvCache(int numLayers, int numHeads, int headDim, int maxRetained, int obsWindow,
                       TensorOptions options) {
        if (numLayers <= 0 || numHeads <= 0 || headDim <= 0 || maxRetained <= 0) {
            throw new IllegalArgumentException("invalid dims");
        }
        if (obsWindow <= 0 || obsWindow >= maxRetained) {
            throw new IllegalArgumentException("obsWindow must be in (0, maxRetained)");
        }
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxRetained = maxRetained;
        this.obsWindow = obsWindow;
        this.options = options;
    }

    public int maxRetained() { return maxRetained; }
    public int obsWindow() { return obsWindow; }

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
     * @param scores optional mass over <em>current full</em> length {@code [Tk]} —
     *               typically from the observation-window queries only (caller may
     *               zero-pad older positions).
     */
    public void appendWithScores(long seqId, Tensor[] kLayers, Tensor[] vLayers, Tensor scores) {
        Seq s = require(seqId);
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
                Tensor flat = scores.dim() > 1 ? scores.reshape(-1) : scores;
                int n = (int) Math.min(flat.numel(), s.score.size());
                for (int i = 0; i < n; i++) {
                    // accumulate votes
                    s.score.set(i, s.score.get(i) + flat.select(0, i).item().toDouble());
                }
            }
            compress(s);
        }
    }

    private void compress(Seq s) {
        int n = s.score.size();
        if (n <= maxRetained) {
            return;
        }
        compressCount.increment();
        int keepOld = maxRetained - obsWindow;
        int obsStart = n - obsWindow;
        boolean[] keep = new boolean[n];
        for (int i = obsStart; i < n; i++) {
            keep[i] = true;
        }
        if (keepOld > 0 && obsStart > 0) {
            Tensor scoreT = zeros(new long[]{obsStart}, new TensorOptions(torch.ScalarType.Float));
            for (int i = 0; i < obsStart; i++) {
                scoreT.narrow(0, i, 1).fill_(new Scalar(s.score.get(i)));
            }
            int k = Math.min(keepOld, obsStart);
            T_TensorTensor_T top = topk(scoreT, k, -1L, true, true);
            Tensor idx = top.get1();
            for (int j = 0; j < k; j++) {
                keep[(int) idx.select(0, j).item().toLong()] = true;
            }
        } else if (keepOld <= 0) {
            // only observation window
        }

        compact(s, keep);
    }

    private void compact(Seq s, boolean[] keep) {
        List<List<Tensor>> newK = new ArrayList<>(numLayers);
        List<List<Tensor>> newV = new ArrayList<>(numLayers);
        List<Double> newScore = new ArrayList<>();
        for (int L = 0; L < numLayers; L++) {
            newK.add(new ArrayList<>());
            newV.add(new ArrayList<>());
        }
        int dropped = 0;
        for (int i = 0; i < keep.length; i++) {
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
        if (t.dim() == 2) return 1;
        if (t.dim() == 3) return (int) t.size(0);
        if (t.dim() == 4) return (int) t.size(2);
        throw new IllegalArgumentException("rank " + t.dim());
    }

    private static Tensor sliceToken(Tensor t, int tIdx) {
        if (t.dim() == 2) return t;
        if (t.dim() == 3) return t.select(0, tIdx);
        if (t.dim() == 4) return t.select(0, 0).select(1, tIdx);
        throw new IllegalArgumentException("rank");
    }

    private static final class Seq {
        List<List<Tensor>> k;
        List<List<Tensor>> v;
        List<Double> score = new ArrayList<>();
        int totalAppended;

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
