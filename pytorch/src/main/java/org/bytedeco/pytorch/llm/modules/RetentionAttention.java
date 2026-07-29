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

import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Multi-scale Retention (RetNet, Sun et al. 2023) — didactic recurrent form.
 *
 * <p>Each head has a decay {@code γ_h ∈ (0,1)}; state updates as
 * {@code S_t = γ S_{t-1} + k_tᵀ v_t}, {@code y_t = q_t S_t}.
 * Lightning Attention is a related linear+decay kernel; this class covers the
 * multi-scale retention recurrence used in RetNet.
 *
 * <p>Cache payload: {@code S [B,H,D,D]}. {@link #forwardCached} → {@code {out, S}}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RetentionAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;
    /** Per-head decay factors γ, registered as buffer-like parameter (fixed after init). */
    public final Tensor gamma;

    private final int nHeads;
    private final int headDim;

    public RetentionAttention(long hiddenSize, int nHeads, int headDim, boolean qkvBias) {
        super("RetentionAttention");
        if (nHeads <= 0) {
            throw new IllegalArgumentException("nHeads must be > 0");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.headDim = Math.max(1, hd);

        long dim = (long) nHeads * this.headDim;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, dim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, dim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, dim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(dim, hiddenSize).bias(false)));

        // γ_h = 1 - 2^(-5 - h·…)  (RetNet-style multi-scale)
        Tensor gg = zeros(new long[]{nHeads});
        for (int h = 0; h < nHeads; h++) {
            double gh = 1.0 - Math.pow(2.0, -5.0 - h * (7.0 / Math.max(1, nHeads - 1)));
            if (gh >= 1.0) {
                gh = 0.999;
            }
            if (gh <= 0) {
                gh = 0.5;
            }
            gg.narrow(0, h, 1).fill_(new Scalar(gh));
        }
        this.gamma = register_parameter("gamma", gg, false);
    }

    public static RetentionAttention mha(long hiddenSize, int nHeads) {
        return new RetentionAttention(hiddenSize, nHeads, (int) (hiddenSize / nHeads), false);
    }

    public int nHeads() { return nHeads; }
    public int headDim() { return headDim; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, null)[0];
    }

    /**
     * @param pastS [B,H,D,D] or null
     * @return {out [B,T,C], newS}
     */
    public Tensor[] forwardCached(Tensor x, Tensor pastS) {
        long B = x.size(0);
        long T = x.size(1);

        Tensor q = q_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);
        Tensor k = k_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);
        Tensor v = v_proj.forward(x).view(B, T, nHeads, headDim).transpose(1, 2);

        Tensor S = pastS != null && pastS.defined()
                ? pastS
                : zeros(new long[]{B, nHeads, headDim, headDim}, x.options());

        Tensor out = zeros(new long[]{B, nHeads, T, headDim}, x.options());
        // gamma: [H] → [1,H,1,1]
        Tensor g = gamma.to(x.scalar_type()).reshape(1, nHeads, 1, 1);

        for (long t = 0; t < T; t++) {
            Tensor kt = k.select(2, t); // [B,H,D]
            Tensor vt = v.select(2, t);
            Tensor qt = q.select(2, t);
            Tensor outer = kt.unsqueeze(-1).mul(vt.unsqueeze(-2)); // [B,H,D,D]
            S = S.mul(g).add(outer);
            Tensor y = qt.unsqueeze(-2).matmul(S).squeeze(-2); // [B,H,D]
            out.select(2, t).copy_(y);
        }

        Tensor y = out.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), S};
    }
}
