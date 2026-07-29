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
 * FlashAttention-style multi-head attention (Dao et al., 2022/2023) — didactic
 * pure-Tensor reference using <b>blockwise online softmax</b>.
 *
 * <p>Numerically matches dense SDPA ({@link Attention}) for identical Q/K/V/O
 * weights within fp32 tolerance. Does <em>not</em> require a CUDA kernel; the
 * name refers to the algorithm (tiling + online softmax), not hardware.
 *
 * <pre>
 *   for each K/V tile:
 *     s = Q_tile K_tileᵀ / √d
 *     m' = max(m, rowmax(s));  α = exp(m − m')
 *     P = exp(s − m');  ℓ' = α·ℓ + rowsum(P)
 *     O' = α·O + P V_tile
 *   y = O / ℓ
 * </pre>
 *
 * <p>HF-compatible parameter names: {@code q_proj}, {@code k_proj}, {@code v_proj}, {@code o_proj}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class FlashAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;
    public final RMSNorm q_norm;
    public final RMSNorm k_norm;

    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final double ropeTheta;
    private final double ropeScaling;
    private final boolean useRope;
    private final boolean useQkNorm;
    private final int slidingWindow;
    private final boolean isCausal;
    private final int blockQ;
    private final int blockK;

    public FlashAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                          double ropeTheta, double ropeScaling, boolean useRope,
                          boolean qkvBias, boolean oBias, boolean useQkNorm, double rmsNormEps,
                          int slidingWindow, boolean isCausal, int blockQ, int blockK) {
        super("FlashAttention");
        if (nHeads <= 0 || nKvHeads <= 0) {
            throw new IllegalArgumentException("nHeads/nKvHeads must be > 0");
        }
        if (nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("nHeads must be divisible by nKvHeads (GQA)");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        if (hd <= 0) {
            throw new IllegalArgumentException("headDim must be > 0");
        }
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.ropeTheta = ropeTheta;
        this.ropeScaling = ropeScaling <= 0 ? 1.0 : ropeScaling;
        this.useRope = useRope;
        this.useQkNorm = useQkNorm;
        this.slidingWindow = slidingWindow;
        this.isCausal = isCausal;
        this.blockQ = Math.max(1, blockQ);
        this.blockK = Math.max(1, blockK);

        long qDim = (long) nHeads * hd;
        long kvDim = (long) nKvHeads * hd;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(oBias)));

        if (useQkNorm) {
            this.q_norm = register_module("q_norm", new RMSNorm(hd, rmsNormEps));
            this.k_norm = register_module("k_norm", new RMSNorm(hd, rmsNormEps));
        } else {
            this.q_norm = null;
            this.k_norm = null;
        }
    }

    public static FlashAttention mha(long hiddenSize, int nHeads, double ropeTheta) {
        return new FlashAttention(hiddenSize, nHeads, nHeads, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, false, false, false, 1e-6, -1, true, 16, 64);
    }

    public static FlashAttention gqa(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new FlashAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, false, false, false, 1e-6, -1, true, 16, 64);
    }

    public static FlashAttention llama(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return gqa(hiddenSize, nHeads, nKvHeads, ropeTheta);
    }

    public static FlashAttention slidingWindow(long hiddenSize, int nHeads, int nKvHeads,
                                               double ropeTheta, int window) {
        return new FlashAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, false, false, false, 1e-6, window, true, 16, 64);
    }

    /** Copy dense {@link Attention} projection weights into this module (parity tests). */
    public void copyWeightsFrom(Attention src) {
        q_proj.weight().copy_(src.q_proj.weight());
        k_proj.weight().copy_(src.k_proj.weight());
        v_proj.weight().copy_(src.v_proj.weight());
        o_proj.weight().copy_(src.o_proj.weight());
        if (src.q_proj.bias() != null && src.q_proj.bias().defined()
                && q_proj.bias() != null && q_proj.bias().defined()) {
            q_proj.bias().copy_(src.q_proj.bias());
            k_proj.bias().copy_(src.k_proj.bias());
            v_proj.bias().copy_(src.v_proj.bias());
        }
        if (src.o_proj.bias() != null && src.o_proj.bias().defined()
                && o_proj.bias() != null && o_proj.bias().defined()) {
            o_proj.bias().copy_(src.o_proj.bias());
        }
        if (useQkNorm && src.useQkNorm() && q_norm != null && src.q_norm != null) {
            q_norm.weight().copy_(src.q_norm.weight());
            k_norm.weight().copy_(src.k_norm.weight());
        }
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public int blockQ() { return blockQ; }
    public int blockK() { return blockK; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    /**
     * @return {output [B,T,C], newK [B,H,T,D] (GQA-repeated), newV same}
     */
    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
        long B = x.size(0);
        long T = x.size(1);

        Tensor q = q_proj.forward(x).view(B, T, nHeads, headDim);
        Tensor k = k_proj.forward(x).view(B, T, nKvHeads, headDim);
        Tensor v = v_proj.forward(x).view(B, T, nKvHeads, headDim);

        if (useQkNorm) {
            q = q_norm.forward(q);
            k = k_norm.forward(k);
        }
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        if (useRope) {
            q = RotaryEmbedding.apply(q, ropeTheta, positionOffset, ropeScaling);
            k = RotaryEmbedding.apply(k, ropeTheta, positionOffset, ropeScaling);
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

        double sc = AttentionOps.scale(headDim);
        Tensor y;
        if (pastLen == 0 && positionOffset == 0) {
            y = AttentionOps.flashOnlineSdpa(q, k, v, isCausal, slidingWindow, blockQ, blockK, sc);
        } else {
            y = AttentionOps.flashOnlineSdpaCached(q, k, v, positionOffset, isCausal, slidingWindow,
                    blockQ, blockK, sc);
        }
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV};
    }
}
