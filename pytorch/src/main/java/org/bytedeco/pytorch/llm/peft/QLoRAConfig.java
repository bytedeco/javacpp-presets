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

/**
 * QLoRA configuration: LoRA hyper-parameters plus 4-bit load metadata.
 *
 * <p><b>MVP note:</b> NF4 kernels are not available in this preset. Runtime
 * behaviour is identical to {@link LoraConfig}; the extra fields are retained
 * for API parity with Hugging Face PEFT and future quantized backends.
 */
public final class QLoRAConfig extends PeftConfig {
    private final LoraConfig lora;
    private final boolean loadIn4bit;
    private final String bnb4bitQuantType;
    private final boolean bnb4bitUseDoubleQuant;
    private final String bnb4bitComputeDtype;

    private QLoRAConfig(Builder b) {
        super(b);
        this.lora = b.loraBuilder.peftType(PeftType.QLORA).build();
        this.loadIn4bit = b.loadIn4bit;
        this.bnb4bitQuantType = b.bnb4bitQuantType;
        this.bnb4bitUseDoubleQuant = b.bnb4bitUseDoubleQuant;
        this.bnb4bitComputeDtype = b.bnb4bitComputeDtype;
    }

    /** Underlying LoRA hyper-parameters used to build {@link LoraLinear}. */
    public LoraConfig lora() {
        return lora;
    }

    public int r() {
        return lora.r();
    }

    public double alpha() {
        return lora.alpha();
    }

    public double scaling() {
        return lora.scaling();
    }

    public boolean loadIn4bit() {
        return loadIn4bit;
    }

    public String bnb4bitQuantType() {
        return bnb4bitQuantType;
    }

    public boolean bnb4bitUseDoubleQuant() {
        return bnb4bitUseDoubleQuant;
    }

    public String bnb4bitComputeDtype() {
        return bnb4bitComputeDtype;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends PeftConfig.Builder<Builder> {
        private final LoraConfig.Builder loraBuilder = LoraConfig.builder();
        private boolean loadIn4bit = true;
        private String bnb4bitQuantType = "nf4";
        private boolean bnb4bitUseDoubleQuant = true;
        private String bnb4bitComputeDtype = "float16";

        public Builder() {
            peftType(PeftType.QLORA);
        }

        public Builder r(int r) {
            loraBuilder.r(r);
            return this;
        }

        public Builder alpha(double alpha) {
            loraBuilder.alpha(alpha);
            return this;
        }

        public Builder dropout(double dropout) {
            loraBuilder.dropout(dropout);
            return this;
        }

        public Builder targetModules(String... modules) {
            loraBuilder.targetModules(modules);
            return this;
        }

        public Builder freezeBase(boolean freezeBase) {
            loraBuilder.freezeBase(freezeBase);
            return this;
        }

        public Builder loadIn4bit(boolean loadIn4bit) {
            this.loadIn4bit = loadIn4bit;
            return this;
        }

        public Builder bnb4bitQuantType(String bnb4bitQuantType) {
            this.bnb4bitQuantType = bnb4bitQuantType;
            return this;
        }

        public Builder bnb4bitUseDoubleQuant(boolean v) {
            this.bnb4bitUseDoubleQuant = v;
            return this;
        }

        public Builder bnb4bitComputeDtype(String dtype) {
            this.bnb4bitComputeDtype = dtype;
            return this;
        }

        @Override
        public QLoRAConfig build() {
            return new QLoRAConfig(this);
        }
    }
}
