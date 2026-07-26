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
import static org.bytedeco.pytorch.global.torch.log;
import static org.bytedeco.pytorch.global.torch.sigmoid;

/**
 * Direct Preference Optimization loss (Rafailov et al.).
 *
 * <pre>
 *   π_logratios  = policy_chosen − policy_rejected
 *   ref_logratios = ref_chosen − ref_rejected
 *   logits      = π_logratios − ref_logratios
 *   sigmoid: −log σ(β · logits)
 *   hinge:   relu(1 − β · logits)
 *   ipo:     (logits − 1/(2β))²
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DPOLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private DPOLoss() {}

    /**
     * @param policyChosenLogps   {@code [B]} log-probs of chosen under policy
     * @param policyRejectedLogps {@code [B]} log-probs of rejected under policy
     * @param refChosenLogps      {@code [B]} under reference (use zeros if reference-free)
     * @param refRejectedLogps    {@code [B]} under reference
     * @param beta                KL strength
     * @param lossType            {@code sigmoid}, {@code hinge}, or {@code ipo}
     * @return scalar mean loss
     */
    public static Tensor compute(
            Tensor policyChosenLogps,
            Tensor policyRejectedLogps,
            Tensor refChosenLogps,
            Tensor refRejectedLogps,
            double beta,
            String lossType) {
        Tensor piLogratios = policyChosenLogps.sub(policyRejectedLogps);
        Tensor refLogratios = refChosenLogps.sub(refRejectedLogps);
        Tensor logits = piLogratios.sub(refLogratios);

        String type = lossType == null ? "sigmoid" : lossType.toLowerCase();
        Tensor losses;
        switch (type) {
            case "hinge": {
                // relu(1 - β * logits) = clamp(1 - β*logits, 0, +inf)
                Tensor t = logits.mul(new Scalar(-beta)).add(new Scalar(1.0));
                losses = clamp(t, new ScalarOptional(new Scalar(0.0)), new ScalarOptional(new Scalar(1e12)));
                break;
            }
            case "ipo": {
                double target = 1.0 / (2.0 * Math.max(beta, 1e-8));
                Tensor diff = logits.sub(new Scalar(target));
                losses = diff.mul(diff);
                break;
            }
            case "sigmoid":
            default: {
                Tensor scaled = logits.mul(new Scalar(beta));
                losses = log(sigmoid(scaled)).neg();
                break;
            }
        }
        return losses.mean();
    }

    public static Tensor compute(
            Tensor policyChosenLogps,
            Tensor policyRejectedLogps,
            Tensor refChosenLogps,
            Tensor refRejectedLogps,
            double beta) {
        return compute(policyChosenLogps, policyRejectedLogps,
                refChosenLogps, refRejectedLogps, beta, "sigmoid");
    }
}
