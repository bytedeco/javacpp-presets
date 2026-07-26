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
package org.bytedeco.pytorch.utils.transformers;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.tokenizers.Encoding;
import org.bytedeco.pytorch.utils.tokenizers.FastTokenizer;
import org.bytedeco.pytorch.utils.transformers.generation.GenerationConfig;
import org.bytedeco.pytorch.utils.transformers.generation.Generator;
import org.bytedeco.pytorch.utils.transformers.loading.WeightLoader;
import org.bytedeco.pytorch.utils.transformers.mapping.ModelRegistry;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * HuggingFace {@code AutoModel} / {@code AutoModelForCausalLM} façade.
 *
 * <p>Delegates to {@link AutoModelForCausalLM} which resolves architecture via
 * {@link ModelRegistry}, zero-copy-binds safetensors, and pairs tokenizer + chat.
 *
 * <pre>{@code
 * AutoModel.Bundle b = AutoModel.fromPretrained("Qwen/Qwen2-0.5B-Instruct", hub);
 * String out = b.generate("Hello", 32);
 * }</pre>
 */
public final class AutoModel {

    private AutoModel() {}

    /** Model + tokenizer + config triplet (backward-compatible wrapper). */
    public static final class Bundle {
        private final AutoModelForCausalLM.Bundle inner;

        public Bundle(AutoModelForCausalLM.Bundle inner) {
            this.inner = Objects.requireNonNull(inner);
        }

        public Module model() { return inner.model(); }
        public FastTokenizer tokenizer() { return inner.tokenizer(); }
        public PretrainedConfig config() { return inner.config(); }
        public Path snapshot() { return inner.snapshot(); }
        public WeightLoader.LoadReport loadReport() { return inner.loadReport(); }
        public AutoModelForCausalLM.Bundle causal() { return inner; }

        /** Encode text and run generate. */
        public String generate(String prompt, int maxNewTokens) {
            return inner.generate(prompt, maxNewTokens);
        }

        public String generate(String prompt, GenerationConfig gen) {
            return inner.generate(prompt, gen);
        }

        /** Low-level: encode → token ids → generate ids → decode. */
        public int[] generateIds(String prompt, int maxNewTokens) {
            Encoding enc = tokenizer().encode(prompt, true);
            GenerationConfig gen = GenerationConfig.builder()
                    .maxNewTokens(maxNewTokens)
                    .eosTokenId(config().eosTokenId())
                    .build();
            return Generator.generate(model(), enc.ids(), gen, config().maxPositionEmbeddings());
        }
    }

    public static Module fromConfig(PretrainedConfig config) {
        return ModelRegistry.create(config);
    }

    public static Bundle fromPretrained(String modelId, HfHub hub) throws IOException {
        return new Bundle(AutoModelForCausalLM.fromPretrained(modelId, hub));
    }

    public static Bundle fromPretrained(String modelId, HfHub hub,
                                        AutoModelForCausalLM.LoadOptions opts) throws IOException {
        return new Bundle(AutoModelForCausalLM.fromPretrained(modelId, hub, opts));
    }

    public static Bundle fromDirectory(Path dir) throws IOException {
        return new Bundle(AutoModelForCausalLM.fromDirectory(dir));
    }

    public static Bundle fromDirectory(Path dir, AutoModelForCausalLM.LoadOptions opts) throws IOException {
        return new Bundle(AutoModelForCausalLM.fromDirectory(dir, opts));
    }

    /** Build an offline tiny bundle without touching disk / weights. */
    public static Bundle tiny(String kind) {
        return new Bundle(AutoModelForCausalLM.tiny(kind));
    }
}
