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
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.GRPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.GRPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Group Relative Policy Optimization trainer (DeepSeek-R1 / HF TRL GRPO subset).
 *
 * <p>No value network: for each prompt, {@link GRPOConfig#numGenerations()}
 * completions are scored and advantages are group-normalized rewards.
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code rewards} — {@code [B*G]} scalar rewards</li>
 *   <li>{@code old_logprobs} — {@code [B*G]} (for clipped variant)</li>
 *   <li>{@code completion_ids} + {@code prompt_ids} (or full {@code input_ids})
 *       with optional masks — for online logprob recompute</li>
 *   <li>or precomputed {@code logprobs} {@code [B*G]}</li>
 *   <li>optional {@code ref_logprobs} for KL penalty ({@code beta > 0})</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class GRPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;            // optional
    private final LlmForward referenceForward; // optional
    private final GRPOConfig grpoConfig;
    private final TensorVector params;
    private final boolean useClipping;

    public GRPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            GRPOConfig config) {
        this(policy, policyForward, reference, referenceForward, optimizer, config, true);
    }

    public GRPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            GRPOConfig config,
            boolean useClipping) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.grpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.useClipping = useClipping;
        if (reference != null) {
            freeze(reference);
        }
    }

    public GRPOTrainer(Module policy, LlmForward policyForward, Optimizer optimizer, GRPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public GRPOConfig grpoConfig() { return grpoConfig; }

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        policy.train(true);
        if (reference != null) {
            reference.eval();
        }
    }

    @Override
    public void eval() {
        super.eval();
        policy.eval();
        if (reference != null) {
            reference.eval();
        }
    }

    @Override
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor rewards = require(batch, "rewards");
        int groupSize = grpoConfig.numGenerations();

        Tensor newLogprobs;
        if (batch.containsKey("logprobs")
                && batch.get("logprobs") != null
                && batch.get("logprobs").defined()) {
            newLogprobs = batch.get("logprobs");
        } else {
            Tensor inputIds = require(batch, "input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            // completion_mask zeros out prompt tokens when provided
            Tensor completionMask = orElse(batch.get("completion_mask"), attentionMask);
            Tensor logits = policyForward.forward(inputIds, attentionMask);
            newLogprobs = LogProbUtils.sequenceLogProbs(logits, labels, completionMask);
        }

        Tensor oldLogprobs = batch.get("old_logprobs");
        Tensor refLogprobs = batch.get("ref_logprobs");

        // Online ref logprobs if needed and not precomputed
        if ((refLogprobs == null || !refLogprobs.defined())
                && grpoConfig.beta() > 0.0
                && referenceForward != null
                && batch.containsKey("input_ids")) {
            Tensor inputIds = batch.get("input_ids");
            Tensor attentionMask = batch.get("attention_mask");
            Tensor labels = orElse(batch.get("labels"), inputIds);
            Tensor completionMask = orElse(batch.get("completion_mask"), attentionMask);
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor refLogits = referenceForward.forward(inputIds, attentionMask);
                refLogprobs = LogProbUtils.sequenceLogProbs(refLogits, labels, completionMask).detach();
            }
        }

        if (useClipping
                && oldLogprobs != null
                && oldLogprobs.defined()
                && grpoConfig.clipRange() > 0.0) {
            Tensor clipped = GRPOLoss.computeClipped(
                    newLogprobs, oldLogprobs, rewards, groupSize, grpoConfig.clipRange());
            if (grpoConfig.beta() > 0.0 && refLogprobs != null && refLogprobs.defined()) {
                // Add KL on top of clipped surrogate
                Tensor kl = newLogprobs.sub(refLogprobs).mean()
                        .mul(new org.bytedeco.pytorch.Scalar(grpoConfig.beta()));
                return clipped.add(kl);
            }
            return clipped;
        }

        return GRPOLoss.compute(
                newLogprobs, rewards, groupSize, grpoConfig.beta(), refLogprobs);
    }

    /**
     * Group-normalize a flat reward vector (delegates to {@link GRPOLoss}).
     */
    public static Tensor groupNormalizeAdvantages(Tensor rewards, int groupSize) {
        return GRPOLoss.groupNormalize(rewards, groupSize);
    }

    private static void freeze(Module m) {
        TensorVector pv = m.parameters();
        for (long i = 0, n = pv.size(); i < n; i++) {
            Tensor p = pv.get(i);
            if (p != null && !p.isNull() && p.defined()) {
                p.requires_grad_(false);
            }
        }
        m.eval();
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
