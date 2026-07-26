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

/** Shared training hyper-parameters for TRL-style trainers. */
public class TrainerConfig {
    private final double learningRate;
    private final int maxSteps;
    private final int loggingSteps;
    private final int gradientAccumulationSteps;
    private final double maxGradNorm;
    private final boolean fp16;
    private final long seed;

    protected TrainerConfig(Builder<?> b) {
        this.learningRate = b.learningRate;
        this.maxSteps = b.maxSteps;
        this.loggingSteps = b.loggingSteps;
        this.gradientAccumulationSteps = b.gradientAccumulationSteps;
        this.maxGradNorm = b.maxGradNorm;
        this.fp16 = b.fp16;
        this.seed = b.seed;
    }

    public double learningRate() { return learningRate; }
    public int maxSteps() { return maxSteps; }
    public int loggingSteps() { return loggingSteps; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public double maxGradNorm() { return maxGradNorm; }
    public boolean fp16() { return fp16; }
    public long seed() { return seed; }

    @SuppressWarnings("unchecked")
    public static class Builder<B extends Builder<B>> {
        private double learningRate = 1e-5;
        private int maxSteps = 1000;
        private int loggingSteps = 10;
        private int gradientAccumulationSteps = 1;
        private double maxGradNorm = 1.0;
        private boolean fp16 = false;
        private long seed = 42L;

        public B learningRate(double v) { this.learningRate = v; return (B) this; }
        public B maxSteps(int v) { this.maxSteps = v; return (B) this; }
        public B loggingSteps(int v) { this.loggingSteps = v; return (B) this; }
        public B gradientAccumulationSteps(int v) { this.gradientAccumulationSteps = v; return (B) this; }
        public B maxGradNorm(double v) { this.maxGradNorm = v; return (B) this; }
        public B fp16(boolean v) { this.fp16 = v; return (B) this; }
        public B seed(long v) { this.seed = v; return (B) this; }

        public TrainerConfig build() { return new TrainerConfig(this); }
    }

    public static Builder<?> builder() { return new Builder<>(); }
}
