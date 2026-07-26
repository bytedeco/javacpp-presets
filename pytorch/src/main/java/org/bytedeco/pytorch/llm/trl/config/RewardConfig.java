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
package org.bytedeco.pytorch.llm.trl.config;

/** Reward-model trainer config (HF TRL {@code RewardConfig} subset). */
public final class RewardConfig extends TrainerConfig {
    private final double margin;
    private final boolean centerRewardsCoefficient;

    private RewardConfig(Builder b) {
        super(b);
        this.margin = b.margin;
        this.centerRewardsCoefficient = b.centerRewardsCoefficient;
    }

    /** Optional margin in BT loss: {@code −log σ(r_c − r_r − margin)}. */
    public double margin() { return margin; }

    /** If true, subtract batch-mean from rewards before the BT loss (stability). */
    public boolean centerRewards() { return centerRewardsCoefficient; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double margin = 0.0;
        private boolean centerRewardsCoefficient = false;

        public Builder margin(double v) { this.margin = v; return this; }
        public Builder centerRewards(boolean v) { this.centerRewardsCoefficient = v; return this; }

        @Override
        public RewardConfig build() { return new RewardConfig(this); }
    }
}
