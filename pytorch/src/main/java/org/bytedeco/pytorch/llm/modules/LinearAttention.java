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
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.elu;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Causal linear attention (Katharopoulos et al. / Performer-style ELU+1 kernel).
 *
 * <p>Uses feature map {@code φ(x) = elu(x) + 1} and maintains recurrent state
 * {@code S = Σ φ(k)ᵀ v}, {@code Z = Σ φ(k)} so decode is O(D²) not O(T·D).
 *
 * <pre>
 *   q' = φ(q);  k' = φ(k)
 *   // causal via prefix cumulative (prefill) or recurrent (decode)
 *   y_t = (q'_t S_t) / (q'_t · Z_t + ε)
 * </pre>
 *
 * <p>Cache payload is {@code {S [B,H,D,D], Z [B,H,D]}} — not full K/V.
 * {@link #forwardCached} returns {@code {out, S, Z}}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LinearAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;

    private final int nHeads;
    private final int headDim;
    private final double eps;

    public LinearAttention(long hiddenSize, int nHeads, int headDim, boolean qkvBias, double eps) {
        super("LinearAttention");
        if (nHeads <= 0) {
            throw new IllegalArgumentException("nHeads must be > 0");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        if (hd <= 0 || hiddenSize % nHeads != 0 && headDim <= 0) {
            // allow hidden != nHeads*hd via projections
            hd = Math.max(1, hd);
        }
        this.nHeads = nHeads;
        this.headDim = hd;
        this.eps = eps > 0 ? eps : 1e-6;

        long dim = (long) nHeads * hd;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, dim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, dim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, dim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(dim, hiddenSize).bias(false)));
    }

    public static LinearAttention mha(long hiddenSize, int nHeads) {
        return new LinearAttention(hiddenSize, nHeads, (int) (hiddenSize / nHeads), false, 1e-6);
    }

    public int nHeads() { return nHeads; }
    public int headDim() { return headDim; }

    /** φ(x) = elu(x) + 1 (positive feature map). */
    public static Tensor featureMap(Tensor x) {
        return elu(x).add(new Scalar(1.0));
    }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, null, null)[0];
    }

    /**
     * @param pastS cumulative S [B,H,D,D] or null
     * @param pastZ cumulative Z [B,H,D] or null
     * @return {out [B,T,C], newS, newZ}
     */
    public Tensor[] forwardCached(Tensor x, Tensor pastS, Tensor pastZ) {
        long B = x.size(0);
        long T = x.size(1);

        Tensor q = q_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2); // [B,H,T,D]
        Tensor k = k_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);
        Tensor v = v_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);

        Tensor qf = featureMap(q);
        Tensor kf = featureMap(k);

        Tensor S = pastS != null && pastS.defined()
                ? pastS
                : zeros(new long[]{B, nHeads, headDim, headDim}, x.options());
        Tensor Z = pastZ != null && pastZ.defined()
                ? pastZ
                : zeros(new long[]{B, nHeads, headDim}, x.options());

        Tensor out = zeros(new long[]{B, nHeads, T, headDim}, x.options());
        // Recurrent causal scan (didactic; clear and correct)
        for (long t = 0; t < T; t++) {
            Tensor kt = kf.select(2, t); // [B,H,D]
            Tensor vt = v.select(2, t);  // [B,H,D]
            Tensor qt = qf.select(2, t); // [B,H,D]
            // S += kᵀ v  →  S[b,h] += outer(kt, vt)
            Tensor outer = kt.unsqueeze(-1).mul(vt.unsqueeze(-2)); // [B,H,D,D]
            S = S.add(outer);
            Z = Z.add(kt);
            // y = (q S) / (q·Z + eps)
            Tensor num = matmulBH(qt, S); // [B,H,D]
            Tensor den = qt.mul(Z).sum(new long[]{-1L}, true, new org.bytedeco.pytorch.ScalarTypeOptional()).add(new Scalar(eps)); // [B,H,1]
            out.select(2, t).copy_(num.div(den));
        }

        Tensor y = out.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), S, Z};
    }

    /** Batched [B,H,D] × [B,H,D,D] → [B,H,D]. */
    private static Tensor matmulBH(Tensor q, Tensor S) {
        // q.unsqueeze(-2) @ S → [B,H,1,D] → squeeze
        return q.unsqueeze(-2).matmul(S).squeeze(-2);
    }
}
