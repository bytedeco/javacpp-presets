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
package org.bytedeco.pytorch.utils.transformers.pipeline;

import org.bytedeco.pytorch.utils.hub.HfHub;
import org.bytedeco.pytorch.utils.transformers.AutoModelForCausalLM;
import org.bytedeco.pytorch.utils.transformers.generation.GenerationConfig;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * High-level text-generation / chat pipeline (HF {@code pipeline("text-generation")} style).
 *
 * <pre>{@code
 * TextGenerationPipeline pipe = TextGenerationPipeline.fromPretrained(
 *     "Qwen/Qwen2-0.5B-Instruct", hub);
 * String out = pipe.chat(List.of(Map.of("role","user","content","Hello!")));
 * }</pre>
 */
public final class TextGenerationPipeline {

    private final AutoModelForCausalLM.Bundle bundle;

    public TextGenerationPipeline(AutoModelForCausalLM.Bundle bundle) {
        this.bundle = Objects.requireNonNull(bundle, "bundle");
    }

    public static TextGenerationPipeline fromPretrained(String modelId, HfHub hub) throws IOException {
        return new TextGenerationPipeline(AutoModelForCausalLM.fromPretrained(modelId, hub));
    }

    public static TextGenerationPipeline fromPretrained(String modelId, HfHub hub,
                                                          AutoModelForCausalLM.LoadOptions opts) throws IOException {
        return new TextGenerationPipeline(AutoModelForCausalLM.fromPretrained(modelId, hub, opts));
    }

    public static TextGenerationPipeline fromDirectory(Path dir) throws IOException {
        return new TextGenerationPipeline(AutoModelForCausalLM.fromDirectory(dir));
    }

    public static TextGenerationPipeline tiny(String kind) {
        return new TextGenerationPipeline(AutoModelForCausalLM.tiny(kind));
    }

    public AutoModelForCausalLM.Bundle bundle() {
        return bundle;
    }

    public String generate(String prompt) {
        return bundle.generate(prompt, bundle.generationConfig());
    }

    public String generate(String prompt, GenerationConfig gen) {
        return bundle.generate(prompt, gen);
    }

    public String generate(String prompt, int maxNewTokens) {
        return bundle.generate(prompt, maxNewTokens);
    }

    public String chat(List<Map<String, String>> messages) {
        return bundle.chat(messages);
    }

    public String chat(List<Map<String, String>> messages, GenerationConfig gen) {
        return bundle.chat(messages, gen);
    }
}
