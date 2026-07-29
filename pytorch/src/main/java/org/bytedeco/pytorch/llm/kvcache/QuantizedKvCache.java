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
 * KIVI-lite style per-token int8 KV cache (didactic).
 *
 * <p>Each token's K and V ({@code [H,D]}) are stored as int8 with a per-token
 * scale (absmax / 127). {@link #gather} dequantizes back to float for attention.
 * Not bit-identical to production KIVI (group-wise / channel-wise variants);
 * sufficient for MSE and finite checks.
 */
public class QuantizedKvCache implements KvCache {

    public final LongAdder appendCount = new LongAdder();
    public final LongAdder quantCount = new LongAdder();

    private final int numLayers;
    private final int numHeads;
    private final int headDim;
    private final int maxLen;
    private final TensorOptions floatOpts;
    private final AtomicLong nextId = new AtomicLong(1);
    private final ConcurrentHashMap<Long, Seq> seqs = new ConcurrentHashMap<>();

    public QuantizedKvCache(int numLayers, int numHeads, int headDim, int maxLen) {
        if (numLayers <= 0 || numHeads <= 0 || headDim <= 0 || maxLen <= 0) {
            throw new IllegalArgumentException("invalid dims");
        }
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxLen = maxLen;
        this.floatOpts = new TensorOptions(torch.ScalarType.Float);
    }

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

    @Override
    public void append(long seqId, Tensor[] kLayers, Tensor[] vLayers) {
        Seq s = require(seqId);
        synchronized (s) {
            int tokens = inferTokens(kLayers[0]);
            for (int t = 0; t < tokens; t++) {
                if (s.len >= maxLen) {
                    // drop oldest
                    for (int L = 0; L < numLayers; L++) {
                        s.kQ.get(L).remove(0);
                        s.vQ.get(L).remove(0);
                        s.kScale.get(L).remove(0);
                        s.vScale.get(L).remove(0);
                    }
                    s.len--;
                }
                for (int L = 0; L < numLayers; L++) {
                    Quant qk = quantize(sliceToken(kLayers[L], t));
                    Quant qv = quantize(sliceToken(vLayers[L], t));
                    s.kQ.get(L).add(qk.data);
                    s.kScale.get(L).add(qk.scale);
                    s.vQ.get(L).add(qv.data);
                    s.vScale.get(L).add(qv.scale);
                    quantCount.increment();
                }
                s.len++;
                s.totalAppended++;
                appendCount.increment();
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
            int n = s.kQ.get(layer).size();
            if (n == 0) {
                Tensor e = zeros(new long[]{0, numHeads, headDim}, floatOpts);
                return new Tensor[]{e, e};
            }
            TensorVector kv = new TensorVector();
            TensorVector vv = new TensorVector();
            for (int i = 0; i < n; i++) {
                kv.push_back(dequant(s.kQ.get(layer).get(i), s.kScale.get(layer).get(i)).unsqueeze(0));
                vv.push_back(dequant(s.vQ.get(layer).get(i), s.vScale.get(layer).get(i)).unsqueeze(0));
            }
            return new Tensor[]{cat(kv, 0), cat(vv, 0)};
        }
    }

    /** Mean squared error of round-trip quant for a single {@code [H,D]} tensor (debug). */
    public static double roundTripMse(Tensor x) {
        Quant q = quantize(x);
        Tensor y = dequant(q.data, q.scale);
        Tensor diff = x.to(torch.ScalarType.Float).sub(y);
        return diff.mul(diff).mean().item().toDouble();
    }

    static Quant quantize(Tensor x) {
        Tensor xf = x.to(torch.ScalarType.Float).contiguous();
        Tensor absmax = xf.abs().max();
        double scale = Math.max(absmax.item().toDouble(), 1e-8) / 127.0;
        Tensor q = xf.div(new Scalar(scale)).round()
                .clamp(new org.bytedeco.pytorch.ScalarOptional(new Scalar(-127)),
                        new org.bytedeco.pytorch.ScalarOptional(new Scalar(127)))
                .to(torch.ScalarType.Char).contiguous().clone();
        return new Quant(q, scale);
    }

    static Tensor dequant(Tensor q, double scale) {
        return q.to(torch.ScalarType.Float).mul(new Scalar(scale));
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

    private static final class Quant {
        final Tensor data;
        final double scale;
        Quant(Tensor data, double scale) {
            this.data = data;
            this.scale = scale;
        }
    }

    private static final class Seq {
        final List<List<Tensor>> kQ;
        final List<List<Tensor>> vQ;
        final List<List<Double>> kScale;
        final List<List<Double>> vScale;
        int len;
        int totalAppended;

        Seq(int layers) {
            kQ = new ArrayList<>(layers);
            vQ = new ArrayList<>(layers);
            kScale = new ArrayList<>(layers);
            vScale = new ArrayList<>(layers);
            for (int i = 0; i < layers; i++) {
                kQ.add(new ArrayList<>());
                vQ.add(new ArrayList<>());
                kScale.add(new ArrayList<>());
                vScale.add(new ArrayList<>());
            }
        }
    }
}
