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

/** PPO trainer config (TRL / classic PPO hyper-parameters). */
public final class PPOConfig extends TrainerConfig {
    private final double clipRange;
    private final double clipRangeVf;
    private final double vfCoef;
    private final double entCoef;
    private final double gamma;
    private final double gaeLambda;
    private final int ppoEpochs;
    private final int miniBatchSize;

    private PPOConfig(Builder b) {
        super(b);
        this.clipRange = b.clipRange;
        this.clipRangeVf = b.clipRangeVf;
        this.vfCoef = b.vfCoef;
        this.entCoef = b.entCoef;
        this.gamma = b.gamma;
        this.gaeLambda = b.gaeLambda;
        this.ppoEpochs = b.ppoEpochs;
        this.miniBatchSize = b.miniBatchSize;
    }

    public double clipRange() { return clipRange; }
    public double clipRangeVf() { return clipRangeVf; }
    public double vfCoef() { return vfCoef; }
    public double entCoef() { return entCoef; }
    public double gamma() { return gamma; }
    public double gaeLambda() { return gaeLambda; }
    public int ppoEpochs() { return ppoEpochs; }
    public int miniBatchSize() { return miniBatchSize; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double clipRange = 0.2;
        private double clipRangeVf = 0.2;
        private double vfCoef = 0.5;
        private double entCoef = 0.01;
        private double gamma = 0.99;
        private double gaeLambda = 0.95;
        private int ppoEpochs = 4;
        private int miniBatchSize = 64;

        public Builder clipRange(double v) { this.clipRange = v; return this; }
        public Builder clipRangeVf(double v) { this.clipRangeVf = v; return this; }
        public Builder vfCoef(double v) { this.vfCoef = v; return this; }
        public Builder entCoef(double v) { this.entCoef = v; return this; }
        public Builder gamma(double v) { this.gamma = v; return this; }
        public Builder gaeLambda(double v) { this.gaeLambda = v; return this; }
        public Builder ppoEpochs(int v) { this.ppoEpochs = v; return this; }
        public Builder miniBatchSize(int v) { this.miniBatchSize = v; return this; }

        @Override
        public PPOConfig build() { return new PPOConfig(this); }
    }
}
