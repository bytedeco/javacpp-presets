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
import org.bytedeco.pytorch.llm.modules.attn.AttentionOps;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.elu;
import static org.bytedeco.pytorch.global.torch.sigmoid;
import static org.bytedeco.pytorch.global.torch.zeros;

/**
 * Infini-attention (Munkhdalai et al., 2024) — <b>lite didactic</b> port.
 *
 * <p>Combines a local sliding-window softmax attention with a compressive
 * linear-attention memory updated across segments:
 * <pre>
 *   mem' = mem + φ(K)ᵀ V          // compressive state [B,H,D,D]
 *   z'   = z   + φ(K)             // normalizer [B,H,D]
 *   y_mem = φ(Q) mem / (φ(Q)·z)
 *   y_loc = softmax_window(Q,K,V)
 *   out = σ(β) · y_mem + (1−σ(β)) · y_loc
 * </pre>
 *
 * <p>Cache payload: {@code {localK, localV, mem, z}} — {@link #forwardCached}
 * returns {@code {out, newK, newV, mem, z}} (5-tensor for full state; callers
 * that only need local KV may ignore mem/z).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class InfiniAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;
    /** Per-head gate β before sigmoid. */
    public final LinearImpl beta_proj;

    private final int nHeads;
    private final int nKvHeads;
    private final int headDim;
    private final double ropeTheta;
    private final boolean useRope;
    private final int window;
    private final double eps;

    public InfiniAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                           double ropeTheta, boolean useRope, boolean qkvBias,
                           int window, double eps) {
        super("InfiniAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads);
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = Math.max(1, hd);
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.window = Math.max(1, window);
        this.eps = eps > 0 ? eps : 1e-6;

        long qDim = (long) nHeads * this.headDim;
        long kvDim = (long) nKvHeads * this.headDim;
        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(qDim, hiddenSize).bias(false)));
        this.beta_proj = register_module("beta_proj",
                new LinearImpl(new LinearOptions(hiddenSize, nHeads).bias(true)));
    }

    public static InfiniAttention gqa(long hiddenSize, int nHeads, int nKvHeads,
                                      double ropeTheta, int window) {
        return new InfiniAttention(hiddenSize, nHeads, nKvHeads, (int) (hiddenSize / nHeads),
                ropeTheta, true, false, window, 1e-6);
    }

    public static InfiniAttention paperDefault(long hiddenSize, int nHeads, double ropeTheta) {
        return gqa(hiddenSize, nHeads, nHeads, ropeTheta, 16);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public int window() { return window; }

    static Tensor phi(Tensor x) {
        return elu(x).add(new Scalar(1.0));
    }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null, null, null)[0];
    }

    /**
     * @return {out, newK, newV, mem, z}
     */
    public Tensor[] forwardCached(Tensor x, long positionOffset,
                                  Tensor pastK, Tensor pastV,
                                  Tensor pastMem, Tensor pastZ) {
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

        // Local window softmax
        Tensor mask = pastLen == 0
                ? AttentionOps.causalMask(T, window)
                : AttentionOps.causalMaskCached(pastLen, T, window);
        double sc = AttentionOps.scale(headDim);
        Tensor yLoc = AttentionOps.denseSdpa(q, k, v, mask, sc, 0.0, false)[0]; // [B,H,T,D]

        // Compressive memory update from current segment K/V
        Tensor kf = phi(newK); // [B,H,T,D]
        Tensor mem = pastMem != null && pastMem.defined()
                ? pastMem
                : zeros(new long[]{B, nHeads, headDim, headDim}, x.options());
        Tensor z = pastZ != null && pastZ.defined()
                ? pastZ
                : zeros(new long[]{B, nHeads, headDim}, x.options());

        // Aggregate segment into memory: Σ_t outer(k_t, v_t)
        for (long t = 0; t < T; t++) {
            Tensor kt = kf.select(2, t);
            Tensor vt = newV.select(2, t);
            mem = mem.add(kt.unsqueeze(-1).mul(vt.unsqueeze(-2)));
            z = z.add(kt);
        }

        Tensor qf = phi(q); // [B,H,T,D]
        Tensor yMem = zeros(new long[]{B, nHeads, T, headDim}, x.options());
        for (long t = 0; t < T; t++) {
            Tensor qt = qf.select(2, t); // [B,H,D]
            Tensor num = qt.unsqueeze(-2).matmul(mem).squeeze(-2);
            Tensor den = qt.mul(z).sum(new long[]{-1L}, true, new org.bytedeco.pytorch.ScalarTypeOptional()).add(new Scalar(eps));
            yMem.select(2, t).copy_(num.div(den));
        }

        // Gate mix per head: β from x → [B,T,H] → [B,H,T,1]
        Tensor beta = sigmoid(beta_proj.forward(x)).transpose(1, 2).unsqueeze(-1);
        Tensor y = yMem.mul(beta).add(yLoc.mul(beta.neg().add(new Scalar(1.0))));
        y = y.transpose(1, 2).contiguous().view(B, T, (long) nHeads * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV, mem, z};
    }
}
