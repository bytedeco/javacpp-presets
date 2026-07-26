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
 * DPO trainer config (Hugging Face TRL {@code DPOConfig} subset).
 *
 * <p>{@code lossType}: {@code sigmoid} (default), {@code hinge}, {@code ipo}.
 */
public final class DPOConfig extends TrainerConfig {
    private final double beta;
    private final String lossType;
    private final boolean referenceFree;
    private final double labelSmoothing;

    private DPOConfig(Builder b) {
        super(b);
        this.beta = b.beta;
        this.lossType = b.lossType;
        this.referenceFree = b.referenceFree;
        this.labelSmoothing = b.labelSmoothing;
    }

    public double beta() { return beta; }
    public String lossType() { return lossType; }
    public boolean referenceFree() { return referenceFree; }
    public double labelSmoothing() { return labelSmoothing; }

    public static Builder builder() { return new Builder(); }

    public static final class Builder extends TrainerConfig.Builder<Builder> {
        private double beta = 0.1;
        private String lossType = "sigmoid";
        private boolean referenceFree = false;
        private double labelSmoothing = 0.0;

        public Builder beta(double v) { this.beta = v; return this; }
        public Builder lossType(String v) { this.lossType = v != null ? v : "sigmoid"; return this; }
        public Builder referenceFree(boolean v) { this.referenceFree = v; return this; }
        public Builder labelSmoothing(double v) { this.labelSmoothing = v; return this; }

        @Override
        public DPOConfig build() { return new DPOConfig(this); }
    }
}
