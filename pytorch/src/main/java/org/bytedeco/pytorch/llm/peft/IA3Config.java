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
 * IA3 configuration (Infused Adapter by Inhibiting and Amplifying Inner Activations).
 *
 * <p>MVP: config + target module names; full {@code IA3Linear} injection can be added
 * alongside {@link LoraLinear}.
 */
public final class IA3Config extends PeftConfig {
    private final String[] targetModules;
    private final String[] feedforwardModules;
    private final boolean initIa3Weights;

    private IA3Config(Builder b) {
        super(b);
        this.targetModules = b.targetModules;
        this.feedforwardModules = b.feedforwardModules;
        this.initIa3Weights = b.initIa3Weights;
    }

    public String[] targetModules() {
        return targetModules.clone();
    }

    public String[] feedforwardModules() {
        return feedforwardModules.clone();
    }

    public boolean initIa3Weights() {
        return initIa3Weights;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder extends PeftConfig.Builder<Builder> {
        private String[] targetModules = new String[] {"k_proj", "v_proj", "down_proj"};
        private String[] feedforwardModules = new String[] {"down_proj"};
        private boolean initIa3Weights = true;

        public Builder() {
            peftType(PeftType.IA3);
        }

        public Builder targetModules(String... modules) {
            this.targetModules = modules;
            return this;
        }

        public Builder feedforwardModules(String... modules) {
            this.feedforwardModules = modules;
            return this;
        }

        public Builder initIa3Weights(boolean initIa3Weights) {
            this.initIa3Weights = initIa3Weights;
            return this;
        }

        @Override
        public IA3Config build() {
            return new IA3Config(this);
        }
    }
}
