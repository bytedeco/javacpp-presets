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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import org.bytedeco.pytorch.TensorOptions;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.arange;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.ones;
import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.tril;
import static org.bytedeco.pytorch.global.torch.triu;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Multi-head / Grouped-Query / Multi-Query self-attention with RoPE and causal mask.
 *
 * <p>HF-compatible parameter names: {@code q_proj}, {@code k_proj}, {@code v_proj}, {@code o_proj}.
 * Optional Qwen3 {@code q_norm}/{@code k_norm} (RMSNorm over head_dim).
 *
 * <p>Factory helpers:
 * <ul>
 *   <li>{@link #mha} — classic MHA (nKvHeads == nHeads)</li>
 *   <li>{@link #gqa} — GQA (Llama3 / Qwen2)</li>
 *   <li>{@link #mqa} — multi-query (nKvHeads == 1)</li>
 *   <li>{@link #llama} / {@link #qwen2} / {@link #qwen3} / {@link #gpt2}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Attention extends Module {

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
    private final double ropeScaling;
    private final boolean useRope;
    private final boolean useQkNorm;
    private final boolean useAlibi;
    private final int slidingWindow; // <=0 means full causal
    private final double dropoutP;
    private final boolean isCausal;

    public Attention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                     double ropeTheta, double ropeScaling, boolean useRope,
                     boolean qkvBias, boolean oBias, boolean useQkNorm, double rmsNormEps,
                     boolean useAlibi, int slidingWindow, double dropoutP, boolean isCausal) {
        super("Attention");
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
        this.useAlibi = useAlibi;
        this.slidingWindow = slidingWindow;
        this.dropoutP = Math.max(0.0, dropoutP);
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
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(oBias)));

        if (useQkNorm) {
            this.q_norm = register_module("q_norm", new RMSNorm(hd, rmsNormEps));
            this.k_norm = register_module("k_norm", new RMSNorm(hd, rmsNormEps));
        } else {
            this.q_norm = null;
            this.k_norm = null;
        }
    }

    // ---- factories ----

    public static Attention mha(long hiddenSize, int nHeads, double ropeTheta) {
        return new Attention(hiddenSize, nHeads, nHeads, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, false, false, false, 1e-6, false, -1, 0.0, true);
    }

    public static Attention gqa(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new Attention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, false, false, false, 1e-6, false, -1, 0.0, true);
    }

    public static Attention mqa(long hiddenSize, int nHeads, double ropeTheta) {
        return new Attention(hiddenSize, nHeads, 1, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, false, false, false, 1e-6, false, -1, 0.0, true);
    }

    /** Llama / Mistral: RoPE, no qkv bias, no qk-norm. */
    public static Attention llama(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return gqa(hiddenSize, nHeads, nKvHeads, ropeTheta);
    }

    /** Qwen2: RoPE, qkv bias on. */
    public static Attention qwen2(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new Attention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, true, false, false, 1e-6, false, -1, 0.0, true);
    }

    /** Qwen3: RoPE, no qkv bias, QK-Norm, optional custom headDim. */
    public static Attention qwen3(long hiddenSize, int nHeads, int nKvHeads,
                                  int headDim, double ropeTheta, double rmsNormEps) {
        return new Attention(hiddenSize, nHeads, nKvHeads, headDim,
                ropeTheta, 1.0, true, false, false, true, rmsNormEps, false, -1, 0.0, true);
    }

    /** GPT-2 style: absolute pos handled outside; no RoPE; bias on qkv and o. */
    public static Attention gpt2(long hiddenSize, int nHeads) {
        return new Attention(hiddenSize, nHeads, nHeads, (int) (hiddenSize / nHeads),
                10000.0, 1.0, false, true, true, false, 1e-5, false, -1, 0.0, true);
    }

    /** Mistral sliding-window attention. */
    public static Attention slidingWindow(long hiddenSize, int nHeads, int nKvHeads,
                                          double ropeTheta, int window) {
        return new Attention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, 1.0, true, false, false, false, 1e-6, false, window, 0.0, true);
    }

    /** ALiBi attention (no RoPE). */
    public static Attention alibi(long hiddenSize, int nHeads) {
        return new Attention(hiddenSize, nHeads, nHeads, (int) (hiddenSize / nHeads),
                10000.0, 1.0, false, false, false, false, 1e-6, true, -1, 0.0, true);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public boolean useQkNorm() { return useQkNorm; }
    public boolean useRope() { return useRope; }
    public boolean useAlibi() { return useAlibi; }
    public int slidingWindow() { return slidingWindow; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    /**
     * Cache-aware forward.
     *
     * @return {output [B,T,C], newK [B,nHeads,T,D] (GQA-repeated), newV same}
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
        q = q.transpose(1, 2); // [B, H, T, D]
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        if (useRope) {
            q = RotaryEmbedding.apply(q, ropeTheta, positionOffset, ropeScaling);
            k = RotaryEmbedding.apply(k, ropeTheta, positionOffset, ropeScaling);
        }

        int nRep = nHeads / nKvHeads;
        k = RotaryEmbedding.repeatKv(k, nRep);
        v = RotaryEmbedding.repeatKv(v, nRep);

        long pastLen = 0L;
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            if (pastK.dim() == 3) {
                pastK = pastK.permute(1, 0, 2).unsqueeze(0);
                pastV = pastV.permute(1, 0, 2).unsqueeze(0);
                if (B > 1) {
                    pastK = pastK.expand(new long[]{B, pastK.size(1), pastK.size(2), pastK.size(3)});
                    pastV = pastV.expand(new long[]{B, pastV.size(1), pastV.size(2), pastV.size(3)});
                }
            }
            pastLen = pastK.size(2);
        }

        Tensor newK = k;
        Tensor newV = v;
        if (pastLen > 0) {
            k = cat(new TensorVector(pastK, k), 2);
            v = cat(new TensorVector(pastV, v), 2);
        }

        double scale = 1.0 / Math.sqrt(headDim);
        Tensor att = matmul(q, k.transpose(-2, -1)).mul(new Scalar(scale));

        if (isCausal) {
            if (pastLen == 0) {
                att = att.add(causalMask(T, slidingWindow));
            } else {
                att = att.add(causalMaskCached(pastLen, T, slidingWindow));
            }
        }
        if (useAlibi) {
            att = att.add(alibiBias(nHeads, pastLen + T, pastLen, T));
        }

        att = softmax(att, -1L);
        if (dropoutP > 0 && is_training()) {
            att = org.bytedeco.pytorch.global.torch.dropout(att, dropoutP, true);
        }
        Tensor y = matmul(att, v);
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV};
    }

    // ---- masks ----

    static Tensor causalMask(long T, int window) {
        Tensor onesT = ones(new long[]{T, T});
        Tensor tri = triu(onesT, 1L);
        if (window > 0 && window < T) {
            // also mask positions older than window: j < i - window + 1 → lower triangle far
            Tensor lowerFar = tril(ones(new long[]{T, T}), -(long) window);
            tri = tri.add(lowerFar);
        }
        return tri.mul(new Scalar(-1e9f));
    }

    static Tensor causalMaskCached(long pastLen, long T, int window) {
        long total = pastLen + T;
        Tensor full = causalMask(total, window); // [total, total]
        return full.narrow(0, pastLen, T); // [T, total]
    }

    /**
     * ALiBi slopes bias [1, H, T_q, T_k].
     * slope_h = 2^(-8/H)^(h+1); values are {@code -slope * |i-j|}.
     */
    static Tensor alibiBias(int nHeads, long totalLen, long pastLen, long T) {
        double base = Math.pow(2.0, -8.0 / Math.max(1, nHeads));
        Tensor qPos = arange(new Scalar(pastLen), new Scalar(pastLen + T), new Scalar(1L),
                new TensorOptions(ScalarType.Float));
        Tensor kPos = arange(new Scalar(0L), new Scalar(totalLen), new Scalar(1L),
                new TensorOptions(ScalarType.Float));
        Tensor absDist = kPos.unsqueeze(0).sub(qPos.unsqueeze(1)).abs(); // [T, total]
        Tensor out = zeros(new long[]{1, nHeads, T, totalLen}, new TensorOptions(ScalarType.Float));
        for (int h = 0; h < nHeads; h++) {
            double slope = Math.pow(base, h + 1);
            out.select(1, h).copy_(absDist.mul(new Scalar(-slope)));
        }
        return out;
    }
}
