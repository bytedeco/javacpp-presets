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
package org.bytedeco.pytorch.llm.modules;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.modules.attn.AttentionOps;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

/**
 * StreamingLLM attention-sink + sliding window (Xiao et al., 2023).
 *
 * <p>Keeps the first {@code sinkTokens} positions always visible and a recent
 * window of {@code windowTokens}; middle tokens are masked out. Works with
 * standard KV cache — eviction of the middle is the cache's job
 * ({@link org.bytedeco.pytorch.llm.kvcache.SlidingWindowKvCache}).
 *
 * <pre>
 *   mask[q, k] = 0  if k &lt; sink or k in [q-window+1, q]
 *              = −∞ otherwise (and causal)
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StreamingSinkAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;

    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final double ropeTheta;
    private final boolean useRope;
    private final int sinkTokens;
    private final int windowTokens;

    public StreamingSinkAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                                  double ropeTheta, boolean useRope, boolean qkvBias,
                                  int sinkTokens, int windowTokens) {
        super("StreamingSinkAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        if (sinkTokens < 0 || windowTokens <= 0) {
            throw new IllegalArgumentException("sinkTokens>=0 and windowTokens>0 required");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.sinkTokens = sinkTokens;
        this.windowTokens = windowTokens;

        long qDim = (long) nHeads * hd;
        long kvDim = (long) nKvHeads * hd;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(false)));
    }

    public static StreamingSinkAttention gqa(long hiddenSize, int nHeads, int nKvHeads,
                                             double ropeTheta, int sink, int window) {
        return new StreamingSinkAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, sink, window);
    }

    public static StreamingSinkAttention paperDefault(long hiddenSize, int nHeads, double ropeTheta) {
        return gqa(hiddenSize, nHeads, nHeads, ropeTheta, 4, 16);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public int sinkTokens() { return sinkTokens; }
    public int windowTokens() { return windowTokens; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
        long B = x.size(0);
        long T = x.size(1);

        Tensor q = q_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);
        Tensor k = k_proj.forward(x).view(B, T, nKvHeads, headDim).transpose(1, 2);
        Tensor v = v_proj.forward(x).view(B, T, nKvHeads, headDim).transpose(1, 2);

        if (useRope) {
            q = RotaryEmbedding.apply(q, ropeTheta, positionOffset);
            k = RotaryEmbedding.apply(k, ropeTheta, positionOffset);
        }
        int nRep = nHeads / nKvHeads;
        k = AttentionOps.repeatKv(k, nRep);
        v = AttentionOps.repeatKv(v, nRep);

        Tensor newK = k;
        Tensor newV = v;
        long pastLen = 0L;
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            Tensor[] merged = AttentionOps.mergePast(pastK, pastV, k, v, B);
            k = merged[0];
            v = merged[1];
            pastLen = k.size(2) - T;
        }

        long total = pastLen + T;
        Tensor mask = AttentionOps.sinkWindowMask(total, pastLen, T, sinkTokens, windowTokens);
        double sc = AttentionOps.scale(headDim);
        Tensor[] sdpa = AttentionOps.denseSdpa(q, k, v, mask, sc, 0.0, false);
        Tensor y = sdpa[0].transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV};
    }
}
