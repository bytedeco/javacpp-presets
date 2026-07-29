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

import static org.bytedeco.pytorch.global.torch.sigmoid;

/**
 * Gated Attention — applies a learned sigmoid gate on the attention output
 * (common in modern LLM variants / GLU-style residual control).
 *
 * <pre>
 *   y = Attention(x)
 *   out = y ⊙ σ(Wg x)     // elementwise or head-wise gate
 * </pre>
 *
 * <p>Supports standard KV cache via {@link #forwardCached}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GatedAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;
    public final LinearImpl gate_proj;

    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final double ropeTheta;
    private final boolean useRope;
    private final int slidingWindow;
    private final boolean isCausal;

    public GatedAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                          double ropeTheta, boolean useRope, boolean qkvBias,
                          int slidingWindow, boolean isCausal) {
        super("GatedAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.slidingWindow = slidingWindow;
        this.isCausal = isCausal;

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
        this.gate_proj = register_module("gate_proj",
                new LinearImpl(new LinearOptions(hiddenSize, hiddenSize).bias(false)));
    }

    public static GatedAttention gqa(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new GatedAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, -1, true);
    }

    public static GatedAttention mha(long hiddenSize, int nHeads, double ropeTheta) {
        return gqa(hiddenSize, nHeads, nHeads, ropeTheta);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }

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

        Tensor mask = null;
        if (isCausal) {
            mask = pastLen == 0
                    ? AttentionOps.causalMask(T, slidingWindow)
                    : AttentionOps.causalMaskCached(pastLen, T, slidingWindow);
        }
        double sc = AttentionOps.scale(headDim);
        Tensor[] sdpa = AttentionOps.denseSdpa(q, k, v, mask, sc, 0.0, false);
        Tensor y = sdpa[0].transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        y = o_proj.forward(y);
        Tensor gate = sigmoid(gate_proj.forward(x));
        return new Tensor[]{y.mul(gate), newK, newV};
    }
}
