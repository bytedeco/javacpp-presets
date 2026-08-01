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
package org.bytedeco.pytorch.llm.ktransformers.moe;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.T_TensorTensor_T;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.ktransformers.config.KtMoEConfig;
import org.bytedeco.pytorch.llm.ktransformers.util.DeviceBudget;
import org.bytedeco.pytorch.llm.modules.Mlp;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.softmax;
import static org.bytedeco.pytorch.global.torch.topk;

/**
 * Schedulable Mixture-of-Experts combining {@code modules.MoE} semantics with
 * CPU–GPU expert residency ({@link ExpertPool} + {@link ExpertScheduler}).
 *
 * <p>Router (gate) stays on the compute device; experts are dispatched via
 * {@link TokenDispatcher}. Optional shared expert matches DeepSeek-style MoE.
 *
 * <p>Use this in KT inference/SFT paths instead of bare {@code modules.MoE}
 * when heterogeneous placement matters. For pure dense reference tests,
 * {@code modules.MoE} remains preferred.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class RoutedMoE extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public final LinearImpl gate;
    public final Mlp.SwiGLU sharedExpert;
    public final LinearImpl sharedExpertGate;

    private final ExpertPool pool;
    private final ExpertScheduler scheduler;
    private final TokenDispatcher dispatcher;
    private final NumaAwarePlacement numa;
    private final long hiddenSize;
    private final int numExperts;
    private final int topK;
    private final boolean normTopkProb;
    private final boolean hasShared;

    public RoutedMoE(long hiddenSize, long intermediateSize, KtMoEConfig moe,
                     DeviceBudget budget) {
        super("RoutedMoE");
        Objects.requireNonNull(moe, "moe");
        this.hiddenSize = hiddenSize;
        this.numExperts = moe.numExperts();
        this.topK = moe.topK();
        this.normTopkProb = moe.normTopkProb();
        this.hasShared = moe.sharedExpert();
        this.numa = NumaAwarePlacement.from(moe);
        this.pool = ExpertPool.createSwiGLU(hiddenSize, intermediateSize, moe, numa);
        // Register expert modules so parameters are visible to optimizers / state_dict
        for (int i = 0; i < pool.numExperts(); i++) {
            ExpertSpec spec = pool.get(i);
            register_module("experts/" + i, spec.module());
        }
        this.scheduler = new ExpertScheduler(pool, moe, budget);
        this.dispatcher = new TokenDispatcher(pool, scheduler);

        this.gate = register_module("gate",
                new LinearImpl(new LinearOptions(hiddenSize, numExperts).bias(false)));
        if (hasShared) {
            this.sharedExpert = register_module("shared_expert",
                    new Mlp.SwiGLU(hiddenSize, intermediateSize));
            this.sharedExpertGate = register_module("shared_expert_gate",
                    new LinearImpl(new LinearOptions(hiddenSize, 1).bias(false)));
        } else {
            this.sharedExpert = null;
            this.sharedExpertGate = null;
        }
    }

    public RoutedMoE(long hiddenSize, long intermediateSize, KtMoEConfig moe) {
        this(hiddenSize, intermediateSize, moe, null);
    }

    /** Mini demo: 4 experts, top-2, 2 GPU slots, shared expert. */
    public static RoutedMoE mini(long hidden, long intermediate) {
        KtMoEConfig cfg = KtMoEConfig.builder()
                .numExperts(4).topK(2).sharedExpert(true)
                .schedule(KtMoEConfig.SchedulePolicy.BALANCED)
                .gpuExpertSlots(2)
                .migrateCooldownSteps(4)
                .migrateHotThreshold(0.10)
                .build();
        return new RoutedMoE(hidden, intermediate, cfg, DeviceBudget.mini());
    }

    public ExpertPool pool() { return pool; }
    public ExpertScheduler scheduler() { return scheduler; }
    public TokenDispatcher dispatcher() { return dispatcher; }
    public NumaAwarePlacement numa() { return numa; }
    public int numExperts() { return numExperts; }
    public int topK() { return topK; }
    public long hiddenSize() { return hiddenSize; }
    public boolean hasSharedExpert() { return hasShared; }
    public ExpertLoadBalanceMetrics metrics() { return pool.metrics(); }

    @Override
    public Tensor forward(Tensor x) {
        long[] origShape = new long[(int) x.dim()];
        for (int i = 0; i < origShape.length; i++) {
            origShape[i] = x.size(i);
        }
        Tensor flat = x.dim() == 2 ? x : x.reshape(-1, hiddenSize);

        Tensor routerLogits = gate.forward(flat);
        Tensor routerProb = softmax(routerLogits, -1L);
        T_TensorTensor_T top = topk(routerProb, topK, -1L, true, true);
        Tensor topW = top.get0().clone();
        Tensor topI = top.get1().clone();

        if (normTopkProb && topK > 1) {
            Tensor denom = topW.sum(new long[]{-1L}).unsqueeze(-1).clamp_min(new Scalar(1e-9));
            topW = topW.div(denom);
        }

        Tensor out = dispatcher.dispatch(flat, topW, topI, topK);

        if (hasShared) {
            Tensor sharedOut = sharedExpert.forward(flat);
            Tensor sg = sharedExpertGate.forward(flat).sigmoid();
            out = out.add(sharedOut.mul(sg));
        }

        if (x.dim() == 2) {
            return out;
        }
        return out.reshape(origShape);
    }

    /**
     * Switch-Transformer style load-balancing auxiliary loss on the gate.
     */
    public Tensor loadBalancingLoss(Tensor x) {
        Tensor flat = x.dim() == 2 ? x : x.reshape(-1, hiddenSize);
        Tensor routerLogits = gate.forward(flat);
        Tensor routerProb = softmax(routerLogits, -1L);
        T_TensorTensor_T top = topk(routerProb, topK, -1L, true, true);
        Tensor topI = top.get1();
        Tensor fre = org.bytedeco.pytorch.global.torch.zeros_like(routerProb);
        Tensor ones = org.bytedeco.pytorch.global.torch.ones_like(topI)
                .to(routerProb.scalar_type().intern())
                .div(new Scalar((double) topK));
        fre = fre.scatter_add(-1L, topI, ones);
        Tensor f = fre.mean(new long[]{0L});
        Tensor P = routerProb.mean(new long[]{0L});
        return f.mul(P).sum().mul(new Scalar((double) numExperts));
    }

    @Override
    public void close() {
        try {
            pool.close();
        } catch (Throwable ignored) {
        }
        super.close();
    }
}
