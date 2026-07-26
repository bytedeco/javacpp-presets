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
import org.bytedeco.pytorch.llm.trl.config.DPOConfig;
import org.bytedeco.pytorch.llm.trl.loss.DPOLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

/**
 * Direct Preference Optimization trainer (HF TRL {@code DPOTrainer} subset).
 *
 * <p>Expected batch keys (all {@code [B, T]} unless noted):
 * <ul>
 *   <li>{@code chosen_input_ids}, {@code rejected_input_ids}</li>
 *   <li>{@code chosen_attention_mask}, {@code rejected_attention_mask} (optional)</li>
 *   <li>{@code chosen_labels}, {@code rejected_labels} (optional; default = input ids)</li>
 *   <li>or precomputed {@code policy_chosen_logps} / {@code policy_rejected_logps}
 *       / {@code ref_chosen_logps} / {@code ref_rejected_logps} as {@code [B]}</li>
 * </ul>
 *
 * <p>When a reference model is provided and {@link DPOConfig#referenceFree()} is
 * false, ref log-probs are computed under {@link NoGradGuard}. Reference-free
 * DPO uses zeros for the reference side.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DPOTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final Module policy;
    private final LlmForward policyForward;
    private final Module reference;          // may be null
    private final LlmForward referenceForward; // may be null
    private final DPOConfig dpoConfig;
    private final TensorVector params;
    private final boolean lengthNormalize;

    public DPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            DPOConfig config) {
        this(policy, policyForward, reference, referenceForward, optimizer, config, false);
    }

    public DPOTrainer(
            Module policy,
            LlmForward policyForward,
            Module reference,
            LlmForward referenceForward,
            Optimizer optimizer,
            DPOConfig config,
            boolean lengthNormalize) {
        super(config, optimizer);
        this.policy = Objects.requireNonNull(policy, "policy");
        this.policyForward = Objects.requireNonNull(policyForward, "policyForward");
        this.reference = reference;
        this.referenceForward = referenceForward;
        this.dpoConfig = Objects.requireNonNull(config, "config");
        this.params = policy.parameters();
        this.lengthNormalize = lengthNormalize;
        if (reference != null) {
            freeze(reference);
        }
    }

    /** Policy-only constructor (reference-free or external ref logps). */
    public DPOTrainer(Module policy, LlmForward policyForward, Optimizer optimizer, DPOConfig config) {
        this(policy, policyForward, null, null, optimizer, config);
    }

    public Module policy() { return policy; }
    public Module reference() { return reference; }
    public DPOConfig dpoConfig() { return dpoConfig; }

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
        // Fast path: precomputed log-probs
        if (batch.containsKey("policy_chosen_logps")
                && batch.get("policy_chosen_logps") != null
                && batch.get("policy_chosen_logps").defined()) {
            Tensor pC = batch.get("policy_chosen_logps");
            Tensor pR = require(batch, "policy_rejected_logps");
            Tensor rC = batch.get("ref_chosen_logps");
            Tensor rR = batch.get("ref_rejected_logps");
            if (dpoConfig.referenceFree() || rC == null || !rC.defined()) {
                rC = zerosLike(pC);
                rR = zerosLike(pR);
            }
            return DPOLoss.compute(pC, pR, rC, rR, dpoConfig.beta(), dpoConfig.lossType());
        }

        Tensor chosenIds = require(batch, "chosen_input_ids");
        Tensor rejectedIds = require(batch, "rejected_input_ids");
        Tensor chosenMask = batch.get("chosen_attention_mask");
        Tensor rejectedMask = batch.get("rejected_attention_mask");
        Tensor chosenLabels = orElse(batch.get("chosen_labels"), chosenIds);
        Tensor rejectedLabels = orElse(batch.get("rejected_labels"), rejectedIds);
        // Completion mask: use attention mask when present (prompt tokens should
        // already be zeroed by the collator — same contract as HF TRL).
        Tensor chosenCompMask = chosenMask;
        Tensor rejectedCompMask = rejectedMask;

        Tensor policyChosenLogits = policyForward.forward(chosenIds, chosenMask);
        Tensor policyRejectedLogits = policyForward.forward(rejectedIds, rejectedMask);
        Tensor policyChosenLp = logps(policyChosenLogits, chosenLabels, chosenCompMask);
        Tensor policyRejectedLp = logps(policyRejectedLogits, rejectedLabels, rejectedCompMask);

        Tensor refChosenLp;
        Tensor refRejectedLp;
        if (dpoConfig.referenceFree() || referenceForward == null) {
            refChosenLp = zerosLike(policyChosenLp);
            refRejectedLp = zerosLike(policyRejectedLp);
        } else {
            try (NoGradGuard guard = new NoGradGuard()) {
                Tensor refChosenLogits = referenceForward.forward(chosenIds, chosenMask);
                Tensor refRejectedLogits = referenceForward.forward(rejectedIds, rejectedMask);
                refChosenLp = logps(refChosenLogits, chosenLabels, chosenCompMask).detach();
                refRejectedLp = logps(refRejectedLogits, rejectedLabels, rejectedCompMask).detach();
            }
        }

        return DPOLoss.compute(
                policyChosenLp, policyRejectedLp,
                refChosenLp, refRejectedLp,
                dpoConfig.beta(), dpoConfig.lossType());
    }

    private Tensor logps(Tensor logits, Tensor labels, Tensor mask) {
        return lengthNormalize
                ? LogProbUtils.sequenceMeanLogProbs(logits, labels, mask)
                : LogProbUtils.sequenceLogProbs(logits, labels, mask);
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

    private static Tensor zerosLike(Tensor t) {
        return org.bytedeco.pytorch.global.torch.zeros_like(t);
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
