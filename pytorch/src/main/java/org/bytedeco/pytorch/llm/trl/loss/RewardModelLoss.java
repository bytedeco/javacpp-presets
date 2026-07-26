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
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.log_sigmoid;

/**
 * Bradley-Terry reward-model preference loss:
 * {@code −mean(log σ(r_chosen − r_rejected))}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class RewardModelLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private RewardModelLoss() {}

    /**
     * @param chosenRewards   {@code [B]} scalar rewards for preferred completions
     * @param rejectedRewards {@code [B]} scalar rewards for dispreferred completions
     * @return scalar mean BT loss
     */
    public static Tensor compute(Tensor chosenRewards, Tensor rejectedRewards) {
        return log_sigmoid(chosenRewards.sub(rejectedRewards)).neg().mean();
    }
}
