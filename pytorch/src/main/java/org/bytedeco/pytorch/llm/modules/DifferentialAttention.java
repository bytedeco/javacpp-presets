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
import org.bytedeco.pytorch.llm.modules.attn.AttentionOps;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.softmax;

/**
 * Differential Transformer attention (Ye et al., Microsoft, 2024).
 *
 * <p>Each logical head is split into two groups; attention is
 * {@code softmax(q1 k1ᵀ) − λ · softmax(q2 k2ᵀ)}, followed by a group RMSNorm.
 * Amplifies noise cancellation vs standard softmax attention.
 *
 * <pre>
 *   [q1|q2] = Wq x;  [k1|k2]=Wk x;  v shared or split
 *   A = softmax(q1 k1ᵀ/√d) − λ softmax(q2 k2ᵀ/√d)
 *   y = GroupNorm(A v);  out = Wo y
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DifferentialAttention extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl q_proj;
    public final LinearImpl k_proj;
    public final LinearImpl v_proj;
    public final LinearImpl o_proj;
    public final RMSNorm group_norm;

    private final int nHeads;      // logical heads (each uses 2 sub-heads)
    private final int nKvHeads;
    private final int headDim;     // per sub-head dim
    private final double ropeTheta;
    private final boolean useRope;
    private final boolean isCausal;
    private final double lambdaInit;

    public DifferentialAttention(long hiddenSize, int nHeads, int nKvHeads, int headDim,
                                 double ropeTheta, boolean useRope, boolean qkvBias,
                                 boolean isCausal, double lambdaInit, double rmsNormEps) {
        super("DifferentialAttention");
        if (nHeads <= 0 || nKvHeads <= 0 || nHeads % nKvHeads != 0) {
            throw new IllegalArgumentException("invalid heads");
        }
        // Project 2 * nHeads sub-heads for Q/K differential pairs
        int hd = headDim > 0 ? headDim : (int) (hiddenSize / nHeads / 2);
        if (hd <= 0) {
            hd = Math.max(1, (int) (hiddenSize / (nHeads * 2)));
        }
        this.nHeads = nHeads;
        this.nKvHeads = nKvHeads;
        this.headDim = hd;
        this.ropeTheta = ropeTheta;
        this.useRope = useRope;
        this.isCausal = isCausal;
        this.lambdaInit = lambdaInit;

        long qDim = (long) nHeads * 2 * hd;      // two groups
        long kvDim = (long) nKvHeads * 2 * hd;
        long vDim = (long) nHeads * 2 * hd;      // V also dual for simplicity
        long oIn = (long) nHeads * 2 * hd;

        this.q_proj = register_module("q_proj",
                new LinearImpl(new LinearOptions(hiddenSize, qDim).bias(qkvBias)));
        this.k_proj = register_module("k_proj",
                new LinearImpl(new LinearOptions(hiddenSize, kvDim).bias(qkvBias)));
        this.v_proj = register_module("v_proj",
                new LinearImpl(new LinearOptions(hiddenSize, vDim).bias(qkvBias)));
        this.o_proj = register_module("o_proj",
                new LinearImpl(new LinearOptions(oIn, hiddenSize).bias(false)));
        this.group_norm = register_module("group_norm", new RMSNorm(2L * hd, rmsNormEps));
    }

    public static DifferentialAttention paperDefault(long hiddenSize, int nHeads, double ropeTheta) {
        int kv = Math.max(1, nHeads / 2);
        if (nHeads % kv != 0) {
            kv = nHeads;
        }
        return new DifferentialAttention(hiddenSize, nHeads, kv, 0,
                ropeTheta, true, false, true, 0.8, 1e-6);
    }

    public static DifferentialAttention gqa(long hiddenSize, int nHeads, int nKvHeads, double ropeTheta) {
        return new DifferentialAttention(hiddenSize, nHeads, nKvHeads, 0,
                ropeTheta, true, false, true, 0.8, 1e-6);
    }

    public int nHeads() { return nHeads; }
    public int nKvHeads() { return nKvHeads; }
    public int headDim() { return headDim; }
    public double lambdaInit() { return lambdaInit; }

    @Override
    public Tensor forward(Tensor x) {
        return forwardCached(x, 0L, null, null)[0];
    }

    public Tensor[] forwardCached(Tensor x, long positionOffset, Tensor pastK, Tensor pastV) {
        long B = x.size(0);
        long T = x.size(1);
        int subH = nHeads * 2;
        int subKv = nKvHeads * 2;

        Tensor q = q_proj.forward(x).view(B, T, subH, headDim).transpose(1, 2);
        Tensor k = k_proj.forward(x).view(B, T, subKv, headDim).transpose(1, 2);
        Tensor v = v_proj.forward(x).view(B, T, subH, headDim).transpose(1, 2);

        if (useRope) {
            q = RotaryEmbedding.apply(q, ropeTheta, positionOffset);
            k = RotaryEmbedding.apply(k, ropeTheta, positionOffset);
        }
        int nRep = subH / subKv;
        k = AttentionOps.repeatKv(k, nRep);
        // v already at subH

        Tensor newK = k;
        Tensor newV = v;
        long pastLen = 0L;
        if (pastK != null && pastK.defined() && pastK.dim() >= 3) {
            Tensor[] merged = AttentionOps.mergePast(pastK, pastV, k, v, B);
            k = merged[0];
            v = merged[1];
            pastLen = k.size(2) - T;
        }

        // Split into group1 / group2 along head dim: even/odd heads
        // Layout: heads [0..nH) = group1, [nH..2nH) = group2
        Tensor q1 = q.narrow(1, 0, nHeads);
        Tensor q2 = q.narrow(1, nHeads, nHeads);
        Tensor k1 = k.narrow(1, 0, nHeads);
        Tensor k2 = k.narrow(1, nHeads, nHeads);
        Tensor v1 = v.narrow(1, 0, nHeads);
        Tensor v2 = v.narrow(1, nHeads, nHeads);

        Tensor mask = null;
        if (isCausal) {
            mask = pastLen == 0
                    ? AttentionOps.causalMask(T, -1)
                    : AttentionOps.causalMaskCached(pastLen, T, -1);
        }
        double sc = AttentionOps.scale(headDim);
        Tensor[] a1 = AttentionOps.denseSdpa(q1, k1, v1, mask, sc, 0.0, false);
        Tensor[] a2 = AttentionOps.denseSdpa(q2, k2, v2, mask, sc, 0.0, false);
        // Differential: y1 - λ y2, then concat groups back
        Tensor y1 = a1[0];
        Tensor y2 = a2[0];
        Tensor yDiff = y1.sub(y2.mul(new Scalar(lambdaInit))); // [B, nH, T, D]
        // Pair with y2 path residual-style: concat [yDiff, y1] or paper uses both V groups
        // Standard Diff Transformer: attend V with (A1 - λ A2) on shared V.
        // We recompute with shared v1 as V:
        Tensor s1 = matmul(q1, k1.transpose(-2, -1)).mul(new Scalar(sc));
        Tensor s2 = matmul(q2, k2.transpose(-2, -1)).mul(new Scalar(sc));
        if (mask != null) {
            s1 = s1.add(mask);
            s2 = s2.add(mask);
        }
        Tensor A = softmax(s1, -1L).sub(softmax(s2, -1L).mul(new Scalar(lambdaInit)));
        // Use concatenated V [v1;v2] reduced: apply A on v1 and v2 then concat
        Tensor o1 = matmul(A, v1);
        Tensor o2 = matmul(A, v2);
        Tensor y = cat(new TensorVector(o1, o2), 1); // [B, 2*nH, T, D]
        y = y.transpose(1, 2).contiguous(); // [B,T,2nH,D]
        // Group norm over last 2*headDim per logical head pair — flatten heads
        y = y.view(B, T, nHeads, 2L * headDim);
        y = group_norm.forward(y);
        y = y.view(B, T, (long) nHeads * 2 * headDim);
        return new Tensor[]{o_proj.forward(y), newK, newV};
    }
}
