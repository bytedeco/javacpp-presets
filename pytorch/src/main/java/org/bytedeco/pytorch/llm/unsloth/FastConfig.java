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
package org.bytedeco.pytorch.llm.unsloth;

import org.bytedeco.pytorch.llm.peft.LoraConfig;
import org.bytedeco.pytorch.llm.peft.QLoRAConfig;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;

import java.util.ArrayList;
import java.util.List;

/**
 * Unsloth FastLanguageModel configuration (Java port).
 */
public final class FastConfig {
    private final int r;
    private final double loraAlpha;
    private final double loraDropout;
    private final boolean loadIn4bit;
    private final boolean loadIn8bit;
    private final boolean gradientCheckpointing;
    private final boolean useRslora;
    private final boolean useGradientCheckpointingUnsloth;
    private final int maxSeqLength;
    private final List<String> targetModules;
    private final String dtype;
    private final boolean fullFinetuning;
    private final boolean useGradientCheckpointing;
    private final String randomStateSeed;

    private FastConfig(Builder b) {
        this.r = b.r;
        this.loraAlpha = b.loraAlpha;
        this.loraDropout = b.loraDropout;
        this.loadIn4bit = b.loadIn4bit;
        this.loadIn8bit = b.loadIn8bit;
        this.gradientCheckpointing = b.gradientCheckpointing;
        this.useRslora = b.useRslora;
        this.useGradientCheckpointingUnsloth = b.useGradientCheckpointingUnsloth;
        this.maxSeqLength = b.maxSeqLength;
        this.targetModules = List.copyOf(b.targetModules);
        this.dtype = b.dtype;
        this.fullFinetuning = b.fullFinetuning;
        this.useGradientCheckpointing = b.useGradientCheckpointing;
        this.randomStateSeed = b.randomStateSeed;
        if (loadIn4bit && loadIn8bit) {
            throw new IllegalArgumentException("loadIn4bit and loadIn8bit are mutually exclusive");
        }
    }

    public static Builder builder() { return new Builder(); }

    public int r() { return r; }
    public double loraAlpha() { return loraAlpha; }
    public double loraDropout() { return loraDropout; }
    public boolean loadIn4bit() { return loadIn4bit; }
    public boolean loadIn8bit() { return loadIn8bit; }
    public boolean gradientCheckpointing() { return gradientCheckpointing; }
    public boolean useRslora() { return useRslora; }
    public boolean useGradientCheckpointingUnsloth() { return useGradientCheckpointingUnsloth; }
    public int maxSeqLength() { return maxSeqLength; }
    public List<String> targetModules() { return targetModules; }
    public String dtype() { return dtype; }
    public boolean fullFinetuning() { return fullFinetuning; }
    public boolean useGradientCheckpointing() { return useGradientCheckpointing; }
    public String randomStateSeed() { return randomStateSeed; }

    public LoraConfig toLoraConfig() {
        return LoraConfig.builder()
                .r(r)
                .alpha(loraAlpha)
                .dropout(loraDropout)
                .useRslora(useRslora)
                .targetModules(targetModules)
                .build();
    }

    public QLoRAConfig toQLoRAConfig() {
        return QLoRAConfig.builder()
                .r(r)
                .alpha(loraAlpha)
                .dropout(loraDropout)
                .targetModules(targetModules.toArray(new String[0]))
                .loadIn4bit(loadIn4bit)
                .bnb4bitQuantType("nf4")
                .bnb4bitUseDoubleQuant(true)
                .bnb4bitComputeDtype(dtype == null || dtype.isEmpty() ? "float16" : dtype)
                .build();
    }

    public BitsAndBytesConfig toBnbConfig() {
        BitsAndBytesConfig.Builder b = BitsAndBytesConfig.builder();
        if (loadIn4bit) {
            b.loadIn4Bit(true)
                    .bnb4BitQuantType("nf4")
                    .bnb4BitUseDoubleQuant(true)
                    .bnb4BitComputeDtype(dtype == null || "float32".equals(dtype) ? "bfloat16" : dtype);
        }
        if (loadIn8bit) {
            b.loadIn8Bit(true)
                    .llmInt8Threshold(6.0);
        }
        // Always skip lm_head for k-bit load (HF default behaviour)
        b.llm_int8_skip_modules("lm_head", "embed_tokens", "wte", "wpe");
        return b.build();
    }

    public static final class Builder {
        private int r = 16;
        private double loraAlpha = 16;
        private double loraDropout = 0.0;
        private boolean loadIn4bit = true;
        private boolean loadIn8bit = false;
        private boolean gradientCheckpointing = true;
        private boolean useRslora = false;
        private boolean useGradientCheckpointingUnsloth = true;
        private int maxSeqLength = 2048;
        private List<String> targetModules = new ArrayList<>(List.of(
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"));
        private String dtype = "float32";
        private boolean fullFinetuning = false;
        private boolean useGradientCheckpointing = true;
        private String randomStateSeed = "3407";

        public Builder r(int r) { this.r = r; return this; }
        public Builder loraAlpha(double loraAlpha) { this.loraAlpha = loraAlpha; return this; }
        public Builder loraDropout(double loraDropout) { this.loraDropout = loraDropout; return this; }
        public Builder loadIn4bit(boolean loadIn4bit) { this.loadIn4bit = loadIn4bit; return this; }
        public Builder loadIn8bit(boolean loadIn8bit) { this.loadIn8bit = loadIn8bit; return this; }
        public Builder gradientCheckpointing(boolean gradientCheckpointing) {
            this.gradientCheckpointing = gradientCheckpointing; return this;
        }
        public Builder useRslora(boolean useRslora) { this.useRslora = useRslora; return this; }
        public Builder useGradientCheckpointingUnsloth(boolean v) {
            this.useGradientCheckpointingUnsloth = v; return this;
        }
        public Builder maxSeqLength(int maxSeqLength) { this.maxSeqLength = maxSeqLength; return this; }
        public Builder targetModules(List<String> targetModules) {
            this.targetModules = new ArrayList<>(targetModules); return this;
        }
        public Builder dtype(String dtype) { this.dtype = dtype; return this; }
        public Builder fullFinetuning(boolean v) { this.fullFinetuning = v; return this; }
        public Builder useGradientCheckpointing(boolean v) { this.useGradientCheckpointing = v; return this; }
        public Builder randomStateSeed(String v) { this.randomStateSeed = v; return this; }
        public FastConfig build() { return new FastConfig(this); }
    }
}
