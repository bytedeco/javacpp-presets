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

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.llm.bitsandbytes.QLoRA;
import org.bytedeco.pytorch.llm.transformers.CausalLM;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

/**
 * Unsloth-inspired fast QLoRA fine-tune facade (Java port).
 *
 * <p>Composes 4-bit base loading ({@code BitsAndBytes}), LoRA injection
 * ({@code PeftModel}), gradient checkpointing flags, and SFT training via
 * {@link UnslothTrainer}. Kernel fusions from the real Unsloth project are
 * configuration toggles + bookkeeping — the numeric path uses standard libtorch ops.
 *
 * <pre>{@code
 * FastLanguageModel fm = Unsloth.fastLanguageModel(PretrainedConfig.tinyGpt2(),
 *     FastConfig.builder().r(8).loadIn4bit(true).build());
 * fm.getPeftModel().trainStep(inputIds);
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Unsloth {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";

    private Unsloth() {}

    public static String version() { return VERSION; }

    public static FastLanguageModel fastLanguageModel(PretrainedConfig config, FastConfig fastConfig) {
        return FastLanguageModel.fromPretrained(config, fastConfig).getPeftModel();
    }

    public static FastLanguageModel fastLanguageModel(PretrainedConfig config) {
        return fastLanguageModel(config, FastConfig.builder().build());
    }

    /** @deprecated use {@link FastLanguageModel} directly; kept as type alias for MVP callers. */
    @Deprecated
    public static final class FastModel {
        private final FastLanguageModel delegate;

        public FastModel(PretrainedConfig config, FastConfig fastConfig) {
            this.delegate = FastLanguageModel.fromPretrained(config, fastConfig).getPeftModel();
        }

        public FastLanguageModel delegate() { return delegate; }

        public CausalLM model() { return delegate.model(); }
        /** @return QLoRA session when 4/8-bit + LoRA applied; may be null. */
        public QLoRA.Session qlora() {
            return delegate.qloraSession();
        }
        public FastConfig fastConfig() { return delegate.fastConfig(); }
        public PretrainedConfig config() { return delegate.config(); }
        public long stepCount() { return delegate.stepCount(); }
        public boolean checkpointingEnabled() { return delegate.checkpointingEnabled(); }
        public void enableGradientCheckpointing() { delegate.enableGradientCheckpointing(); }
        public void disableGradientCheckpointing() { delegate.disableGradientCheckpointing(); }
        public org.bytedeco.pytorch.Tensor forward(org.bytedeco.pytorch.Tensor inputIds) {
            return delegate.forward(inputIds);
        }
        public org.bytedeco.pytorch.Tensor trainStep(org.bytedeco.pytorch.Tensor inputIds) {
            return delegate.trainStep(inputIds);
        }
        public long trainableParameters() { return delegate.trainableParameters(); }
        public long totalParameters() { return delegate.totalParameters(); }
        public java.util.Map<String, Object> stats() { return delegate.stats(); }
        public int[] generate(int[] prompt, int maxNew) { return delegate.generate(prompt, maxNew); }
    }
}
