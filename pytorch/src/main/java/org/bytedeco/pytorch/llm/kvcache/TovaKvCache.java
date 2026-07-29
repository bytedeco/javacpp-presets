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
 * TOVA — Token Omission Via Attention (Oren et al.).
 *
 * <p>At each decode step, when over budget, drop the token with the
 * <b>lowest latest-step</b> attention weight (not cumulative). Recent token
 * just appended is always kept. Simple, decode-time online policy.
 */
public class TovaKvCache implements KvCache {

    public final LongAdder appendCount = new LongAdder();
    public final LongAdder evictCount = new LongAdder();

    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int budget;
    private final TensorOptions options;
    private final AtomicLong nextId = new AtomicLong(1);
    private final ConcurrentHashMap<Long, Seq> seqs = new ConcurrentHashMap<>();

    public TovaKvCache(int numLayers, int numHeads, int headDim, int budget) {
        this(numLayers, numHeads, headDim, budget, new TensorOptions(torch.ScalarType.Float));
    }

    public TovaKvCache(int numLayers, int numHeads, int headDim, int budget, TensorOptions options) {
        if (numLayers <= 0 || numHeads <= 0 || headDim <= 0 || budget <= 0) {
            throw new IllegalArgumentException("invalid dims/budget");
        }
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.budget = budget;
        this.options = options;
    }

    public int budget() { return budget; }

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
     * @param latestScores attention weights from the <em>current</em> query over
     *                     all current keys {@code [Tk]} (or {@code [B,Tk]}). Required
     *                     to choose eviction victim when over budget; if null, drops oldest.
     */
    public void appendWithScores(long seqId, Tensor[] kLayers, Tensor[] vLayers, Tensor latestScores) {
        Seq s = require(seqId);
        synchronized (s) {
            int tokens = inferTokens(kLayers[0]);
            for (int t = 0; t < tokens; t++) {
                for (int L = 0; L < numLayers; L++) {
                    s.k.get(L).add(sliceToken(kLayers[L], t).contiguous().clone());
                    s.v.get(L).add(sliceToken(vLayers[L], t).contiguous().clone());
                }
                s.totalAppended++;
                appendCount.increment();
            }
            if (latestScores != null && latestScores.defined()) {
                s.lastScores = latestScores.dim() > 1 ? latestScores.reshape(-1).contiguous().clone()
                        : latestScores.contiguous().clone();
            }
            while (s.k.get(0).size() > budget) {
                evictOne(s);
            }
        }
    }

    private void evictOne(Seq s) {
        int n = s.k.get(0).size();
        if (n <= budget) {
            return;
        }
        int victim = 0;
        if (s.lastScores != null && s.lastScores.defined() && s.lastScores.numel() >= n) {
            // lowest score among all but the last token
            double best = Double.POSITIVE_INFINITY;
            for (int i = 0; i < n - 1; i++) {
                double sc = s.lastScores.select(0, i).item().toDouble();
                if (sc < best) {
                    best = sc;
                    victim = i;
                }
            }
        } else {
            victim = 0; // drop oldest
        }
        for (int L = 0; L < numLayers; L++) {
            s.k.get(L).remove(victim);
            s.v.get(L).remove(victim);
        }
        if (s.lastScores != null && s.lastScores.defined() && s.lastScores.numel() >= n) {
            // rebuild scores without victim
            List<Double> sc = new ArrayList<>(n - 1);
            for (int i = 0; i < n; i++) {
                if (i == victim) continue;
                sc.add(s.lastScores.select(0, Math.min(i, (int) s.lastScores.numel() - 1)).item().toDouble());
            }
            Tensor ns = zeros(new long[]{sc.size()}, new TensorOptions(torch.ScalarType.Float));
            for (int i = 0; i < sc.size(); i++) {
                ns.narrow(0, i, 1).fill_(new Scalar(sc.get(i)));
            }
            s.lastScores = ns;
        }
        evictCount.increment();
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
            return s.k.get(0).size();
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
        if (s == null) throw new IllegalArgumentException("unknown seq " + id);
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
        final List<List<Tensor>> k;
        final List<List<Tensor>> v;
        Tensor lastScores;
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
