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
package org.bytedeco.pytorch.llm.peft;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * LoRA configuration (mirrors Hugging Face {@code LoraConfig}).
 *
 * <p>Default targets match common attention projections:
 * {@code q_proj}, {@code v_proj}, {@code k_proj}, {@code o_proj}, {@code linear}.
 */
public final class LoraConfig extends PeftConfig {
    private final int r;
    private final double alpha;
    private final double dropout;
    private final List<String> targetModules;
    private final boolean freezeBase;
    private final boolean useRslora;
    private final String bias; // "none" | "all" | "lora_only"

    protected LoraConfig(Builder b) {
        super(b);
        if (b.r <= 0) {
            throw new IllegalArgumentException("r must be > 0");
        }
        this.r = b.r;
        this.alpha = b.alpha;
        this.dropout = b.dropout;
        this.targetModules = Collections.unmodifiableList(new ArrayList<>(b.targetModules));
        this.freezeBase = b.freezeBase;
        this.useRslora = b.useRslora;
        this.bias = b.bias;
    }

    public int r() {
        return r;
    }

    public double alpha() {
        return alpha;
    }

    public double dropout() {
        return dropout;
    }

    public List<String> targetModules() {
        return targetModules;
    }

    public boolean freezeBase() {
        return freezeBase;
    }

    public boolean useRslora() {
        return useRslora;
    }

    public String bias() {
        return bias;
    }

    /** {@code alpha / r} or {@code alpha / sqrt(r)} when rsLoRA is enabled. */
    public double scaling() {
        return useRslora ? alpha / Math.sqrt(r) : alpha / (double) r;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends PeftConfig.Builder<Builder> {
        private int r = 8;
        private double alpha = 16.0;
        private double dropout = 0.0;
        private List<String> targetModules = new ArrayList<>(
                Arrays.asList("q_proj", "v_proj", "k_proj", "o_proj", "linear", "lin"));
        private boolean freezeBase = true;
        private boolean useRslora = false;
        private String bias = "none";

        public Builder() {
            peftType(PeftType.LORA);
        }

        public Builder r(int r) {
            this.r = r;
            return this;
        }

        public Builder alpha(double alpha) {
            this.alpha = alpha;
            return this;
        }

        public Builder dropout(double dropout) {
            this.dropout = dropout;
            return this;
        }

        public Builder targetModules(String... modules) {
            this.targetModules = new ArrayList<>(Arrays.asList(modules));
            return this;
        }

        public Builder targetModules(List<String> modules) {
            this.targetModules = new ArrayList<>(Objects.requireNonNull(modules));
            return this;
        }

        public Builder freezeBase(boolean freezeBase) {
            this.freezeBase = freezeBase;
            return this;
        }

        public Builder useRslora(boolean useRslora) {
            this.useRslora = useRslora;
            return this;
        }

        public Builder bias(String bias) {
            this.bias = bias != null ? bias : "none";
            return this;
        }

        @Override
        public LoraConfig build() {
            return new LoraConfig(this);
        }
    }
}
