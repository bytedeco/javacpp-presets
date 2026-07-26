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
package org.bytedeco.pytorch.utils.transformers.modeling;
import org.bytedeco.pytorch.distributed.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.triu;

/**
 * Multi-head / Grouped-Query self-attention with RoPE and causal mask.
 *
 * <p>HF-compatible parameter names: {@code q_proj}, {@code k_proj}, {@code v_proj}, {@code o_proj}.
 * Optional Qwen3 {@code q_norm}/{@code k_norm} (RMSNorm over head_dim).
 * Qwen2 uses bias on q/k/v; Llama/Mistral/Qwen3 typically do not.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ModelingAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;
    /** Qwen3 QK-Norm over head_dim (null when disabled). */
    public final RMSNorm q_norm;
    public final RMSNorm k_norm;

    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final double ropeTheta;
    private final boolean useRope;
    private final boolean useQkNorm;

    public ModelingAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                             double ropeTheta, boolean useRope, boolean qkvBias, boolean useQkNorm,
                             double rmsNormEps) {
        super("ModelingAttention");
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
        // Classic models require hidden == nHeads * headDim; Qwen3 may differ but still
        // projects to nHeads*headDim then o_proj back to hiddenSize.
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.useQkNorm = useQkNorm;

        long qDim = (long) nHeads * hd;
        long kvDim = (long) nKvHeads * hd;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        // o_proj is bias-free in Llama/Qwen2/Qwen3/Mistral
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(false)));

        if (useQkNorm) {
            this.q_norm = register_module("q_norm", new RMSNorm(hd, rmsNormEps));
            this.k_norm = register_module("k_norm", new RMSNorm(hd, rmsNormEps));
        } else {
            this.q_norm = null;
            this.k_norm = null;
        }
    }

    /** Llama-style: RoPE on, no qkv bias, no qk-norm, headDim = hidden/nHeads. */
    public ModelingAttention(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        this(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, false, 1e-6);
    }

    public ModelingAttention(long hiddenSize, int nHeads, int nKvHeads,
                             double ropeTheta, boolean useRope, boolean qkvBias) {
        this(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, useRope, qkvBias, false, 1e-6);
    }

    /** Qwen2-style: RoPE on, qkv bias on, no qk-norm. */
    public static ModelingAttention qwen2(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new ModelingAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, true, false, 1e-6);
    }

    /**
     * Qwen3-style: RoPE on, no qkv bias, QK-Norm over head_dim.
     * {@code headDim} may differ from {@code hiddenSize / nHeads}.
     */
    public static ModelingAttention qwen3(long hiddenSize, int nHeads, int nKvHeads,
                                          int headDim, double ropeTheta, double rmsNormEps) {
        return new ModelingAttention(hiddenSize, nHeads, nKvHeads, headDim,
                ropeTheta, true, false, true, rmsNormEps);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public boolean useQkNorm() { return useQkNorm; }

    @Override
    public Tensor forward(Tensor x) {
        long B = x.size(0);
        long T = x.size(1);

        Tensor q = q_proj.forward(x).view(B, T, nHeads, headDim);
        Tensor k = k_proj.forward(x).view(B, T, nKvHeads, headDim);
        Tensor v = v_proj.forward(x).view(B, T, nKvHeads, headDim);

        // Qwen3: RMSNorm over last dim (head_dim) before transpose
        if (useQkNorm) {
            q = q_norm.forward(q);
            k = k_norm.forward(k);
        }
        q = q.transpose(1, 2); // [B, H, T, D]
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        if (useRope) {
            q = ModelingRope.apply(q, ropeTheta);
            k = ModelingRope.apply(k, ropeTheta);
        }

        int nRep = nHeads / nKvHeads;
        k = ModelingRope.repeatKv(k, nRep);
        v = ModelingRope.repeatKv(v, nRep);

        double scale = 1.0 / Math.sqrt(headDim);
        Tensor att = matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale));
        att = att.add(causalMask(T));
        att = softmax(att, -1L);
        Tensor y = matmul(att, v);
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return o_proj.forward(y);
    }

    private static Tensor causalMask(long T) {
        Tensor onesT = ones(new long[]{T, T});
        Tensor tri = triu(onesT, 1L);
        return tri.mul(new Scalar(-1e9f));
    }

    /**
     * Causal mask for decode with past: shape [1,1,T_cur,totalLen].
     * Positions attending into the future (j > pastLen + i) are masked.
     */
    private static Tensor causalMaskCached(long pastLen, long T) {
        long total = pastLen + T;
        Tensor onesT = ones(new long[]{T, total});
        // Mask column j when j > pastLen + row
        // Build via full causal of total then take last T rows.
        Tensor full = causalMask(total); // [total, total]
        return full.narrow(0, pastLen, T); // [T, total]
    }

    // ---- cache-aware incremental decode (named method, no native vtable touch) ----

    /**
     * Cache-aware forward for inference serving.
     *
     * @param x             input hidden states, shape [B, T, C]  (T=1 for decode, T≥1 for prefill)
     * @param positionOffset starting position index for RoPE (total tokens already processed)
     * @param pastK          cached K from previous steps, shape [B, nHeads, pastLen, headDim]
     *                       (already GQA-repeated), or null / empty
     * @param pastV          cached V from previous steps, same layout, or null / empty
     * @return array of 3 tensors: {output [B,T,C], newK [B,nHeads,T,headDim], newV [B,nHeads,T,headDim]}
     *         where newK/newV are GQA-repeated (nHeads, not nKvHeads) for cache storage consistency
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
            q = ModelingRope.apply(q, ropeTheta, positionOffset);
            k = ModelingRope.apply(k, ropeTheta, positionOffset);
        }

        // GQA repeat BEFORE cache concat so past/new share the same head count
        int nRep = nHeads / nKvHeads;
        k = ModelingRope.repeatKv(k, nRep);
        v = ModelingRope.repeatKv(v, nRep);

        long pastLen = 0L;
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            // Accept [B,H,past,D] or [past,H,D] (from PagedKvCache.gather)
            if (pastK.dim() == 3) {
                // [T, H, D] → [1, H, T, D] then expand batch
                pastK = pastK.permute(1, 0, 2).unsqueeze(0); // [1, H, T, D]
                pastV = pastV.permute(1, 0, 2).unsqueeze(0);
                if (B > 1) {
                    pastK = pastK.expand(new long[]{B, pastK.size(1), pastK.size(2), pastK.size(3)});
                    pastV = pastV.expand(new long[]{B, pastV.size(1), pastV.size(2), pastV.size(3)});
                }
            }
            pastLen = pastK.size(2);
        }

        Tensor newK = k; // [B, H, T, D] — what callers store into cache
        Tensor newV = v;

        if (pastLen > 0) {
            k = cat(new org.bytedeco.pytorch.TensorVector(pastK, k), 2);
            v = cat(new org.bytedeco.pytorch.TensorVector(pastV, v), 2);
        }

        long totalLen = pastLen + T;
        double scale = 1.0 / Math.sqrt(headDim);
        Tensor att = matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale));
        if (pastLen == 0) {
            att = att.add(causalMask(T));
        } else {
            att = att.add(causalMaskCached(pastLen, T));
        }
        att = softmax(att, -1L);
        Tensor y = matmul(att, v);
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);

        return new Tensor[]{o_proj.forward(y), newK, newV};
    }
}
