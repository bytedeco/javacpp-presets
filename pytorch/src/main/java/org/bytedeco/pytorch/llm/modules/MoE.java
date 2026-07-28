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
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.topk;
import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Sparse Mixture-of-Experts FFN (DeepSeek / Mixtral / Qwen-MoE style).
 *
 * <p>Each expert is a {@link Mlp.SwiGLU}. A linear router produces logits over
 * {@code numExperts}; top-k experts are selected per token and their outputs
 * are weighted-summed.
 *
 * <p>This is a correct reference implementation (dense dispatch over experts
 * with a per-token mask), suitable for building small MoE networks and tests.
 * Production serving would use token-grouped GEMMs; the module API stays the same.
 *
 * <p>Optional shared expert (DeepSeek-V2/V3 style) is added when
 * {@code sharedExpert=true}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MoE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl gate;
    public final List<Mlp.SwiGLU> experts = new ArrayList<>();
    /** DeepSeek shared expert (always-on dense FFN); null when disabled. */
    public final Mlp.SwiGLU shared_expert;
    public final LinearImpl shared_expert_gate; // optional sigmoid gate for shared

    private final int numExperts;
    private final int topK;
    private final long hiddenSize;
    private final long intermediateSize;
    private final boolean normTopkProb;
    private final boolean hasShared;

    public MoE(long hiddenSize, long intermediateSize, int numExperts, int topK,
               boolean normTopkProb, boolean sharedExpert) {
        super("MoE");
        if (numExperts < 1) {
            throw new IllegalArgumentException("numExperts must be >= 1");
        }
        if (topK < 1 || topK > numExperts) {
            throw new IllegalArgumentException("topK must be in [1, numExperts]");
        }
        this.hiddenSize = hiddenSize;
        this.intermediateSize = intermediateSize;
        this.numExperts = numExperts;
        this.topK = topK;
        this.normTopkProb = normTopkProb;
        this.hasShared = sharedExpert;

        this.gate = register_module("gate",
                new LinearImpl(new LinearOptions(hiddenSize, numExperts).bias(false)));
        for (int i = 0; i < numExperts; i++) {
            experts.add(register_module("experts/" + i,
                    new Mlp.SwiGLU(hiddenSize, intermediateSize)));
        }
        if (sharedExpert) {
            this.shared_expert = register_module("shared_expert",
                    new Mlp.SwiGLU(hiddenSize, intermediateSize));
            this.shared_expert_gate = register_module("shared_expert_gate",
                    new LinearImpl(new LinearOptions(hiddenSize, 1).bias(false)));
        } else {
            this.shared_expert = null;
            this.shared_expert_gate = null;
        }
    }

    public MoE(long hiddenSize, long intermediateSize, int numExperts, int topK) {
        this(hiddenSize, intermediateSize, numExperts, topK, true, false);
    }

    /** DeepSeek-style: routed experts + shared expert. */
    public static MoE deepseek(long hiddenSize, long intermediateSize,
                               int numExperts, int topK) {
        return new MoE(hiddenSize, intermediateSize, numExperts, topK, true, true);
    }

    /** Mixtral-style: 8 experts, top-2, no shared. */
    public static MoE mixtral(long hiddenSize, long intermediateSize) {
        return new MoE(hiddenSize, intermediateSize, 8, 2, true, false);
    }

    public int numExperts() { return numExperts; }
    public int topK() { return topK; }
    public long hiddenSize() { return hiddenSize; }
    public long intermediateSize() { return intermediateSize; }
    public boolean hasSharedExpert() { return hasShared; }

    @Override
    public Tensor forward(Tensor x) {
        // x: [B, T, H] or [N, H]
        long[] origShape = new long[(int) x.dim()];
        for (int i = 0; i < origShape.length; i++) {
            origShape[i] = x.size(i);
        }
        Tensor flat = x.dim() == 2 ? x : x.reshape(-1, hiddenSize); // [N, H]
        long N = flat.size(0);

        Tensor routerLogits = gate.forward(flat); // [N, E]
        Tensor routerProb = softmax(routerLogits, -1L);

        T_TensorTensor_T top = topk(routerProb, topK, -1L, true, true);
        Tensor topW = top.get0().clone(); // [N, K] values
        Tensor topI = top.get1().clone(); // [N, K] indices (long)

        if (normTopkProb && topK > 1) {
            // keepdim via unsqueeze — JavaCPP sum(long[], keepdim, dtype) needs all 3 args
            Tensor denom = topW.sum(new long[]{-1L}).unsqueeze(-1).clamp_min(new Scalar(1e-9));
            topW = topW.div(denom);
        }

        Tensor out = zeros_like(flat);
        // Dense reference dispatch: for each expert, mask-scale all-token expert FFN.
        for (int e = 0; e < numExperts; e++) {
            Tensor eq = topI.eq(new Scalar((long) e)); // bool [N,K]
            Tensor w = topW.mul(eq.to(topW.scalar_type())).sum(new long[]{-1L}); // [N]
            if (w.gt(new Scalar(0.0)).any().item_bool()) {
                Tensor expertOut = experts.get(e).forward(flat); // [N, H]
                out = out.add(expertOut.mul(w.unsqueeze(-1)));
            }
        }

        if (hasShared) {
            Tensor sharedOut = shared_expert.forward(flat);
            Tensor sg = shared_expert_gate.forward(flat).sigmoid(); // [N,1]
            out = out.add(sharedOut.mul(sg));
        }

        if (x.dim() == 2) {
            return out;
        }
        return out.reshape(origShape);
    }

    /**
     * Auxiliary load-balancing loss (Switch Transformer style).
     * Call with the same input as the last forward for an approximate signal,
     * or pass router probabilities explicitly.
     *
     * <p>{@code loss = E * sum_e (f_e * P_e)} where f is fraction of tokens
     * dispatched to expert e and P is mean router prob.
     */
    public Tensor loadBalancingLoss(Tensor x) {
        Tensor flat = x.dim() == 2 ? x : x.reshape(-1, hiddenSize);
        Tensor routerLogits = gate.forward(flat);
        Tensor routerProb = softmax(routerLogits, -1L); // [N, E]
        T_TensorTensor_T top = topk(routerProb, topK, -1L, true, true);
        Tensor topI = top.get1(); // [N, K]
        long N = flat.size(0);
        // one-hot-ish frequency
        Tensor fre = zeros_like(routerProb); // [N, E]
        // scatter add 1/k for each selected expert
        Tensor ones = org.bytedeco.pytorch.global.torch.ones_like(topI)
                .to(routerProb.scalar_type().intern())
                .div(new Scalar((double) topK));
        fre = fre.scatter_add(-1L, topI, ones);
        Tensor f = fre.mean(new long[]{0L}); // [E]
        Tensor P = routerProb.mean(new long[]{0L}); // [E]
        return f.mul(P).sum().mul(new Scalar((double) numExperts));
    }
}
