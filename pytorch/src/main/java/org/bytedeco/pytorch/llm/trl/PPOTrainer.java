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
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.PPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.PPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Proximal Policy Optimization trainer for LLM RLHF (HF TRL {@code PPOTrainer} subset).
 *
 * <p>Two operating modes:
 * <ol>
 *   <li><b>Precomputed rollout</b> — batch already contains
 *       {@code old_logprobs}, {@code advantages}, {@code returns},
 *       {@code old_values}, {@code new_logprobs}, {@code values}, {@code entropy}.</li>
 *   <li><b>Online</b> — provide {@link PolicyValueForward} so the trainer can
 *       recompute log-probs / values from {@code input_ids} + {@code labels}.</li>
 * </ol>
 *
 * <p>Also exposes static {@link #computeGae} matching classic GAE-λ.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    /**
     * Policy + value head forward for online PPO.
     *
     * @param inputIds      {@code [B, T]}
     * @param attentionMask optional
     * @return {@link PolicyValueOutput}
     */
    @FunctionalInterface
    public interface PolicyValueForward {
        PolicyValueOutput forward(Tensor inputIds, Tensor attentionMask);
    }

    /** Bundle returned by {@link PolicyValueForward}. */
    public static final class PolicyValueOutput {
        /** Logits {@code [B, T, V]} (optional if logprobs already provided). */
        public final Tensor logits;
        /** Per-token or per-sequence values {@code [B, T]} or {@code [B]}. */
        public final Tensor values;
        /** Optional entropy {@code [B]} or {@code [B, T]}. */
        public final Tensor entropy;

        public PolicyValueOutput(Tensor logits, Tensor values, Tensor entropy) {
            this.logits = logits;
            this.values = values;
            this.entropy = entropy;
        }

        public PolicyValueOutput(Tensor logits, Tensor values) {
            this(logits, values, null);
        }
    }

    private final Module model;
    private final PolicyValueForward pvForward; // may be null for precomputed mode
    private final PPOConfig ppoConfig;
    private final TensorVector params;
    private final boolean normalizeAdvantages;

    public PPOTrainer(
            Module model,
            PolicyValueForward pvForward,
            Optimizer optimizer,
            PPOConfig config) {
        this(model, pvForward, optimizer, config, true);
    }

    public PPOTrainer(
            Module model,
            PolicyValueForward pvForward,
            Optimizer optimizer,
            PPOConfig config,
            boolean normalizeAdvantages) {
        super(config, optimizer);
        this.model = Objects.requireNonNull(model, "model");
        this.pvForward = pvForward;
        this.ppoConfig = Objects.requireNonNull(config, "config");
        this.params = model.parameters();
        this.normalizeAdvantages = normalizeAdvantages;
    }

    /** Precomputed-only constructor (no online recompute). */
    public PPOTrainer(Module model, Optimizer optimizer, PPOConfig config) {
        this(model, null, optimizer, config);
    }

    public Module model() { return model; }
    public PPOConfig ppoConfig() { return ppoConfig; }

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
        Tensor oldLogprobs = require(batch, "old_logprobs");
        Tensor advantages = require(batch, "advantages");
        Tensor returns = require(batch, "returns");
        Tensor oldValues = batch.get("old_values");

        Tensor newLogprobs;
        Tensor values;
        Tensor entropy;

        if (batch.containsKey("new_logprobs")
                && batch.get("new_logprobs") != null
                && batch.get("new_logprobs").defined()) {
            newLogprobs = batch.get("new_logprobs");
            values = require(batch, "values");
            entropy = orElse(batch.get("entropy"), zeros_like(newLogprobs));
        } else {
            if (pvForward == null) {
                throw new IllegalStateException(
                        "batch missing new_logprobs/values and no PolicyValueForward was provided");
            }
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            PolicyValueOutput out = pvForward.forward(inputIds, attentionMask);
            newLogprobs = LogProbUtils.sequenceLogProbs(out.logits, labels, attentionMask);
            values = out.values;
            // Reduce values to [B] if [B, T]
            if (values.dim() > 1) {
                values = values.mean(new long[]{values.dim() - 1});
            }
            entropy = out.entropy != null && out.entropy.defined()
                    ? out.entropy
                    : zeros_like(newLogprobs);
        }

        if (normalizeAdvantages) {
            advantages = normalize(advantages);
        }

        PPOLoss.Result result = PPOLoss.compute(
                newLogprobs,
                oldLogprobs,
                advantages,
                values,
                returns,
                oldValues,
                entropy,
                ppoConfig.clipRange(),
                ppoConfig.clipRangeVf(),
                ppoConfig.vfCoef(),
                ppoConfig.entCoef());
        return result.total;
    }

    /**
     * Generalized Advantage Estimation.
     *
     * @param rewards {@code [T]} or {@code [T, B]}
     * @param values  {@code [T+1]} or {@code [T+1, B]} (bootstrap value at T)
     * @param masks   {@code [T]} or {@code [T, B]} (1 = not done)
     * @param gamma   discount
     * @param lam     GAE λ
     * @return {@code {advantages, returns}} each shaped like rewards
     */
    public static Tensor[] computeGae(
            Tensor rewards, Tensor values, Tensor masks, double gamma, double lam) {
        long T = rewards.size(0);
        Tensor advantages = zeros_like(rewards);
        Tensor lastGae = org.bytedeco.pytorch.global.torch.zeros(
                new long[]{}, rewards.options());

        for (long t = T - 1; t >= 0; t--) {
            Tensor nextVal = values.select(0, t + 1);
            Tensor maskT = masks.select(0, t);
            Tensor delta = rewards.select(0, t)
                    .add(nextVal.mul(new Scalar(gamma)).mul(maskT))
                    .sub(values.select(0, t));
            lastGae = delta.add(lastGae.mul(new Scalar(gamma * lam)).mul(maskT));
            advantages.select(0, t).copy_(lastGae);
        }
        Tensor valueSlice = values.slice(0,
                new org.bytedeco.pytorch.LongOptional(0),
                new org.bytedeco.pytorch.LongOptional(T), 1);
        Tensor returns = advantages.add(valueSlice);
        return new Tensor[]{advantages, returns};
    }

    /** Convenience using {@link PPOConfig#gamma()} / {@link PPOConfig#gaeLambda()}. */
    public Tensor[] computeGae(Tensor rewards, Tensor values, Tensor masks) {
        return computeGae(rewards, values, masks, ppoConfig.gamma(), ppoConfig.gaeLambda());
    }

    private static Tensor normalize(Tensor x) {
        Tensor mean = x.mean();
        Tensor std = x.std().add(new Scalar(1e-8));
        return x.sub(mean).div(std);
    }

    private static Tensor orElse(Tensor a, Tensor b) {
        return a != null && a.defined() ? a : b;
    }

    private static Tensor require(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        if (t == null || !t.defined()) {
            throw new IllegalArgumentException("batch missing required key: " + key);
        }
        return t;
    }
}
