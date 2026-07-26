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
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.clamp;
import static org.bytedeco.pytorch.global.torch.min;
import static org.bytedeco.pytorch.global.torch.stack;

/**
 * Proximal Policy Optimization loss (clipped surrogate + value + entropy).
 *
 * <pre>
 *   ratio   = exp(new_logp − old_logp)
 *   surr1   = ratio * adv
 *   surr2   = clip(ratio, 1−ε, 1+ε) * adv
 *   policy  = −mean(min(surr1, surr2))
 *   value   = mean((V − returns)²)   [optionally clipped]
 *   total   = policy + vfCoef * value − entCoef * entropy
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PPOLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private PPOLoss() {}

    public static final class Result {
        public final Tensor total;
        public final Tensor policy;
        public final Tensor value;
        public final Tensor entropy;

        public Result(Tensor total, Tensor policy, Tensor value, Tensor entropy) {
            this.total = total;
            this.policy = policy;
            this.value = value;
            this.entropy = entropy;
        }
    }

    /**
     * @param newLogprobs {@code [T]} log π_θ(a|s)
     * @param oldLogprobs {@code [T]} log π_old(a|s)
     * @param advantages  {@code [T]} GAE advantages (preferably normalized)
     * @param values      {@code [T]} current value predictions
     * @param returns     {@code [T]} GAE returns
     * @param oldValues   {@code [T]} value at rollout time (for clipped VF); may equal values
     * @param entropy     {@code [T]} or scalar policy entropy
     * @param clipRange   ε for policy
     * @param clipRangeVf ε for value (0 to disable value clipping)
     * @param vfCoef      value loss coefficient
     * @param entCoef     entropy bonus coefficient
     */
    public static Result compute(
            Tensor newLogprobs,
            Tensor oldLogprobs,
            Tensor advantages,
            Tensor values,
            Tensor returns,
            Tensor oldValues,
            Tensor entropy,
            double clipRange,
            double clipRangeVf,
            double vfCoef,
            double entCoef) {
        Tensor ratio = newLogprobs.sub(oldLogprobs).exp();
        Tensor surr1 = ratio.mul(advantages);
        Tensor ratioClipped = clamp(ratio, new ScalarOptional(new Scalar(1.0 - clipRange)), new ScalarOptional(new Scalar(1.0 + clipRange)));
        Tensor surr2 = ratioClipped.mul(advantages);
        // policy loss = -mean(min(surr1, surr2))
        Tensor policyLoss = min(surr1, surr2).mean().neg();

        Tensor valueLoss;
        if (clipRangeVf > 0.0 && oldValues != null && oldValues.defined()) {
            Tensor vClipped = oldValues.add(
                    clamp(values.sub(oldValues), new ScalarOptional(new Scalar(-clipRangeVf)), new ScalarOptional(new Scalar(clipRangeVf))));
            Tensor vf1 = values.sub(returns).pow(new Scalar(2.0));
            Tensor vf2 = vClipped.sub(returns).pow(new Scalar(2.0));
            // max(vf1, vf2) via stack+max if needed; use 0.5 * mean of elementwise max
            valueLoss = elementwiseMax(vf1, vf2).mean().mul(new Scalar(0.5));
        } else {
            valueLoss = values.sub(returns).pow(new Scalar(2.0)).mean().mul(new Scalar(0.5));
        }

        Tensor ent = entropy.mean();
        Tensor total = policyLoss
                .add(valueLoss.mul(new Scalar(vfCoef)))
                .sub(ent.mul(new Scalar(entCoef)));
        return new Result(total, policyLoss, valueLoss, ent);
    }

    private static Tensor elementwiseMax(Tensor a, Tensor b) {
        // max(a,b) = 0.5 * (a+b + |a-b|)
        Tensor diff = a.sub(b).abs();
        return a.add(b).add(diff).mul(new Scalar(0.5));
    }
}
