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

/**
 * GRPO (Group Relative Policy Optimization) config.
 *
 * <p>Samples {@code numGenerations} completions per prompt and normalizes
 * rewards within each group (no value network).
 */
public final class GRPOConfig extends TrainerConfig {
    private final int numGenerations;
    private final double beta;
    private final double clipRange;
    private final double temperature;
    private final int maxCompletionLength;

    private GRPOConfig(Builder b) {
        super(b);
        this.numGenerations = b.numGenerations;
        this.beta = b.beta;
        this.clipRange = b.clipRange;
        this.temperature = b.temperature;
        this.maxCompletionLength = b.maxCompletionLength;
    }

    public int numGenerations() { return numGenerations; }
    public double beta() { return beta; }
    public double clipRange() { return clipRange; }
    public double temperature() { return temperature; }
    public int maxCompletionLength() { return maxCompletionLength; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private int numGenerations = 4;
        private double beta = 0.04;
        private double clipRange = 0.2;
        private double temperature = 0.9;
        private int maxCompletionLength = 256;

        public Builder numGenerations(int v) { this.numGenerations = v; return this; }
        public Builder beta(double v) { this.beta = v; return this; }
        public Builder clipRange(double v) { this.clipRange = v; return this; }
        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder maxCompletionLength(int v) { this.maxCompletionLength = v; return this; }

        @Override
        public GRPOConfig build() { return new GRPOConfig(this); }
    }
}
