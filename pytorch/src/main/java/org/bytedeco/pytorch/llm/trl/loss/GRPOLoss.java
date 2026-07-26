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
package org.bytedeco.pytorch.llm.trl.loss;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.ScalarTypeOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.clamp;

/**
 * Group Relative Policy Optimization loss (Shao et al. / DeepSeek-R1 style).
 *
 * <p>For each prompt, {@code G} completions are scored; advantages are
 * group-normalized rewards (no critic). Optional KL penalty vs a reference
 * policy via {@code beta * (logπ − logπ_ref)}.
 *
 * <pre>
 *   adv_i = (r_i − mean_G(r)) / (std_G(r) + eps)
 *   loss  = −mean( adv * logπ )   [+ optional KL]
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class GRPOLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private GRPOLoss() {}

    /**
     * @param logprobs    {@code [B*G]} token-sum log-probs of completions under policy
     * @param rewards     {@code [B*G]} scalar rewards aligned with logprobs
     * @param groupSize   G — number of generations per prompt ({@code B*G == logprobs.numel()})
     * @param beta        KL coefficient (0 to disable; requires refLogprobs)
     * @param refLogprobs {@code [B*G]} optional reference log-probs
     * @return scalar loss
     */
    public static Tensor compute(
            Tensor logprobs,
            Tensor rewards,
            int groupSize,
            double beta,
            Tensor refLogprobs) {
        if (groupSize <= 0) {
            throw new IllegalArgumentException("groupSize must be > 0");
        }
        Tensor advantages = groupNormalize(rewards, groupSize);
        Tensor policyLoss = logprobs.mul(advantages).mean().neg();
        if (beta > 0.0 && refLogprobs != null && refLogprobs.defined()) {
            // Approx token-level KL surrogate: mean(logπ - logπ_ref) * beta
            Tensor kl = logprobs.sub(refLogprobs).mean().mul(new Scalar(beta));
            return policyLoss.add(kl);
        }
        return policyLoss;
    }

    public static Tensor compute(Tensor logprobs, Tensor rewards, int groupSize) {
        return compute(logprobs, rewards, groupSize, 0.0, null);
    }

    /**
     * Reshape rewards to {@code [B, G]}, normalize within each group, flatten.
     * Implemented with pure Tensor ops when possible; falls back to per-group
     * mean/std via view.
     */
    public static Tensor groupNormalize(Tensor rewards, int groupSize) {
        long n = rewards.numel();
        if (n % groupSize != 0) {
            throw new IllegalArgumentException(
                    "rewards length " + n + " not divisible by groupSize " + groupSize);
        }
        long batches = n / groupSize;
        Tensor grouped = rewards.reshape(batches, groupSize); // [B, G]
        // mean over G with keepdim
        Tensor mean = grouped.mean(new long[]{1L}, true, new ScalarTypeOptional()); // [B, 1]
        Tensor centered = grouped.sub(mean);
        Tensor var = centered.mul(centered).mean(new long[]{1L}, true, new ScalarTypeOptional());
        Tensor std = var.add(new Scalar(1e-8)).sqrt();
        Tensor normed = centered.div(std);
        return normed.reshape(n);
    }

    /**
     * Clipped GRPO variant: {@code -mean(min(r*adv, clip(r)*adv))} where
     * {@code r = exp(logπ - logπ_old)}.
     */
    public static Tensor computeClipped(
            Tensor newLogprobs,
            Tensor oldLogprobs,
            Tensor rewards,
            int groupSize,
            double clipRange) {
        Tensor advantages = groupNormalize(rewards, groupSize);
        Tensor ratio = newLogprobs.sub(oldLogprobs).exp();
        Tensor surr1 = ratio.mul(advantages);
        Tensor surr2 = clamp(ratio, new ScalarOptional(new Scalar(1.0 - clipRange)), new ScalarOptional(new Scalar(1.0 + clipRange))).mul(advantages);
        // min via 0.5*(a+b-|a-b|)
        Tensor diff = surr1.sub(surr2).abs();
        Tensor m = surr1.add(surr2).sub(diff).mul(new Scalar(0.5));
        return m.mean().neg();
    }
}
