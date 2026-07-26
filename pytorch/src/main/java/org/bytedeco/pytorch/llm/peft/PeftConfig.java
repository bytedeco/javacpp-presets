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
 * Base PEFT configuration (mirrors Hugging Face {@code PeftConfig}).
 */
public class PeftConfig {
    private final PeftType peftType;
    private final String taskType;
    private final boolean inferenceMode;

    protected PeftConfig(Builder<?> b) {
        this.peftType = b.peftType;
        this.taskType = b.taskType;
        this.inferenceMode = b.inferenceMode;
    }

    public PeftType peftType() {
        return peftType;
    }

    public String taskType() {
        return taskType;
    }

    public boolean inferenceMode() {
        return inferenceMode;
    }

    @SuppressWarnings("unchecked")
    public static class Builder<B extends Builder<B>> {
        private PeftType peftType = PeftType.LORA;
        private String taskType = "CAUSAL_LM";
        private boolean inferenceMode = false;

        public B peftType(PeftType peftType) {
            this.peftType = peftType;
            return (B) this;
        }

        public B taskType(String taskType) {
            this.taskType = taskType;
            return (B) this;
        }

        public B inferenceMode(boolean inferenceMode) {
            this.inferenceMode = inferenceMode;
            return (B) this;
        }

        public PeftConfig build() {
            return new PeftConfig(this);
        }
    }

    public static Builder<?> builder() {
        return new Builder<>();
    }
}
