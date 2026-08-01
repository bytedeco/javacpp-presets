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
package org.bytedeco.pytorch.llm.llamafactory.train;

import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.model.ModelLoader.LoadedModel;
import org.bytedeco.pytorch.llm.llamafactory.model.ValueHead;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.RewardTrainer;
import org.bytedeco.pytorch.llm.trl.config.RewardConfig;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Tensor;

/**
 * RM (reward model) stage bridge → {@link RewardTrainer}.
 *
 * <p>When the batch lacks precomputed {@code chosen_rewards}/{@code rejected_rewards},
 * a {@link ValueHead} on last-token hidden states is used via {@link RewardTrainer.RewardForward}.
 * For the pure-Java CausalLM path (logits only, no hidden API), prefer precomputed rewards
 * or pairwise log-prob proxies supplied by the host collator.
 */
public final class RmTrainerBridge {
    private RmTrainerBridge() {}

    public static RewardTrainer create(FactoryArgs args, LoadedModel loaded, int maxSteps) {
        BaseTrainer t = TrainerFactory.create(args, loaded, maxSteps);
        if (t instanceof RewardTrainer rm) {
            return rm;
        }
        Optimizer opt = TrainerFactory.buildOptimizer(args, loaded);
        RewardConfig cfg = TrainerFactory.rewardConfig(args, maxSteps);
        return new RewardTrainer(loaded.module(), opt, cfg);
    }

    /**
     * Build a reward forward that scores sequences by mean-pooling LM logits as a
     * cheap proxy (offline / tiny models). Production hosts should attach a real
     * {@link ValueHead} on hidden states.
     */
    public static RewardTrainer createWithValueHead(
            FactoryArgs args, LoadedModel loaded, ValueHead head, int maxSteps) {
        Optimizer opt = TrainerFactory.buildOptimizer(args, loaded);
        RewardConfig cfg = TrainerFactory.rewardConfig(args, maxSteps);
        CausalLM causal = loaded.causalLM();
        RewardTrainer.RewardForward fwd = (ids, mask) -> {
            // CausalLM exposes logits [B,T,V]; use mean logit energy as scalar proxy
            Tensor logits = causal.forward(ids);
            // mean over vocab then over time → [B]
            Tensor perToken = logits.mean(/*dim*/ new long[]{logits.dim() - 1});
            if (mask != null && mask.defined()) {
                Tensor m = mask.to(perToken.dtype());
                Tensor summed = perToken.mul(m).sum(new long[]{perToken.dim() - 1});
                Tensor denom = m.sum(new long[]{m.dim() - 1}).clamp_min(new org.bytedeco.pytorch.Scalar(1.0));
                return summed.div(denom);
            }
            return perToken.mean(new long[]{perToken.dim() - 1});
        };
        return new RewardTrainer(loaded.module(), fwd, opt, cfg);
    }
}
