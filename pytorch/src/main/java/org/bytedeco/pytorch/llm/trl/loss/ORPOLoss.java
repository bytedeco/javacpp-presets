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
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;

/**
 * Odds Ratio Preference Optimization (Hong et al.) — reference-free DPO variant.
 *
 * <pre>
 *   log_odds = π_chosen − π_rejected
 *   ratio    = log_odds − log σ(log_odds)
 *   loss     = −log σ(β · log_odds) − β · log σ(ratio)
 * </pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ORPOLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private ORPOLoss() {}

    /**
     * @param policyChosenLogps   {@code [B]}
     * @param policyRejectedLogps {@code [B]}
     * @param beta                odds-ratio coefficient (HF default {@code 0.1})
     * @return scalar mean loss
     */
    public static Tensor compute(
            Tensor policyChosenLogps,
            Tensor policyRejectedLogps,
            double beta) {
        Tensor logOdds = policyChosenLogps.sub(policyRejectedLogps);
        Tensor ratio = logOdds.sub(log_sigmoid(logOdds));
        Tensor pref = log_sigmoid(logOdds.mul(new Scalar(beta))).neg();
        Tensor orTerm = log_sigmoid(ratio).neg();
        return pref.add(orTerm.mul(new Scalar(beta))).mean();
    }
}
