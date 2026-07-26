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
package org.bytedeco.pytorch.llm.trl;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.RewardConfig;
import org.bytedeco.pytorch.llm.trl.loss.RewardModelLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;

/**
 * Reward-model trainer (HF TRL {@code RewardTrainer} subset).
 *
 * <p>Trains a scalar reward head with Bradley-Terry preference loss.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>precomputed {@code chosen_rewards} / {@code rejected_rewards} {@code [B]}, or</li>
 *   <li>{@code chosen_input_ids} / {@code rejected_input_ids} (+ optional masks)
 *       together with a {@link RewardForward} that maps sequences → scalar rewards.</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RewardTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    /** Maps {@code (input_ids, attention_mask)} → scalar rewards {@code [B]}. */
    @FunctionalInterface
    public interface RewardForward {
        Tensor forward(Tensor inputIds, Tensor attentionMask);
    }

    private final Module model;
    private final RewardForward rewardForward; // may be null if only precomputed rewards
    private final RewardConfig rewardConfig;
    private final TensorVector params;

    public RewardTrainer(
            Module model,
            RewardForward rewardForward,
            Optimizer optimizer,
            RewardConfig config) {
        super(config, optimizer);
        this.model = Objects.requireNonNull(model, "model");
        this.rewardForward = rewardForward;
        this.rewardConfig = Objects.requireNonNull(config, "config");
        this.params = model.parameters();
    }

    /** Precomputed-rewards-only constructor. */
    public RewardTrainer(Module model, Optimizer optimizer, RewardConfig config) {
        this(model, null, optimizer, config);
    }

    public Module model() { return model; }
    public RewardConfig rewardConfig() { return rewardConfig; }

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        model.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        model.eval();
    }

    @Override
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor chosen;
        Tensor rejected;

        if (batch.containsKey("chosen_rewards")
                && batch.get("chosen_rewards") != null
                && batch.get("chosen_rewards").defined()) {
            chosen = batch.get("chosen_rewards");
            rejected = require(batch, "rejected_rewards");
        } else {
            if (rewardForward == null) {
                throw new IllegalStateException(
                        "batch missing chosen_rewards and no RewardForward was provided");
            }
            Tensor chosenIds = require(batch, "chosen_input_ids");
            Tensor rejectedIds = require(batch, "rejected_input_ids");
            Tensor chosenMask = batch.get("chosen_attention_mask");
            Tensor rejectedMask = batch.get("rejected_attention_mask");
            chosen = rewardForward.forward(chosenIds, chosenMask);
            rejected = rewardForward.forward(rejectedIds, rejectedMask);
        }

        if (rewardConfig.centerRewards()) {
            Tensor mean = chosen.add(rejected).mul(new Scalar(0.5)).mean();
            chosen = chosen.sub(mean);
            rejected = rejected.sub(mean);
        }

        double margin = rewardConfig.margin();
        if (margin != 0.0) {
            return log_sigmoid(chosen.sub(rejected).sub(new Scalar(margin))).neg().mean();
        }
        return RewardModelLoss.compute(chosen, rejected);
    }

    private static Tensor require(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        if (t == null || !t.defined()) {
            throw new IllegalArgumentException("batch missing required key: " + key);
        }
        return t;
    }
}
