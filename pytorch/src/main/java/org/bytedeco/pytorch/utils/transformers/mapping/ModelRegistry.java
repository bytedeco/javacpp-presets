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
package org.bytedeco.pytorch.utils.transformers.mapping;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.transformers.CausalLM;
import org.bytedeco.pytorch.utils.transformers.PretrainedConfig;
import org.bytedeco.pytorch.utils.transformers.modeling.LlamaForCausalLM;
import org.bytedeco.pytorch.utils.transformers.modeling.Qwen2ForCausalLM;

import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * Maps HuggingFace {@code model_type} / {@code architectures} to a Module factory
 * and a {@link WeightMap}.
 */
public final class ModelRegistry {

    public record Entry(
            String modelType,
            Function<PretrainedConfig, Module> factory,
            WeightMap weightMap
    ) {}

    private static final Map<String, Entry> REGISTRY = new ConcurrentHashMap<>();

    static {
        register("qwen2", Qwen2ForCausalLM::fromConfig, WeightMaps.qwen2());
        register("qwen", Qwen2ForCausalLM::fromConfig, WeightMaps.qwen2());
        register("llama", LlamaForCausalLM::fromConfig, WeightMaps.llama());
        register("mistral", LlamaForCausalLM::fromConfig, WeightMaps.mistral());
        // GPT-2 / generic fall back to the original CausalLM teaching model
        register("gpt2", CausalLM::fromConfig, WeightMaps.gpt2());
        register("gpt", CausalLM::fromConfig, WeightMaps.gpt2());
        register("generic", CausalLM::fromConfig, WeightMaps.identity());
    }

    private ModelRegistry() {}

    public static void register(String modelType,
                                Function<PretrainedConfig, Module> factory,
                                WeightMap weightMap) {
        Objects.requireNonNull(modelType, "modelType");
        Objects.requireNonNull(factory, "factory");
        String key = modelType.toLowerCase(Locale.ROOT);
        REGISTRY.put(key, new Entry(key, factory, weightMap == null ? WeightMap.identity() : weightMap));
    }

    public static Entry resolve(PretrainedConfig config) {
        Objects.requireNonNull(config, "config");
        // Prefer architectures[0] when present in extra
        Object archs = config.extra().get("architectures");
        if (archs instanceof java.util.List<?> list && !list.isEmpty()) {
            String a = String.valueOf(list.get(0)).toLowerCase(Locale.ROOT);
            if (a.contains("qwen2")) return must("qwen2");
            if (a.contains("llama")) return must("llama");
            if (a.contains("mistral")) return must("mistral");
            if (a.contains("gpt2")) return must("gpt2");
        }
        String mt = config.modelType() == null
                ? "generic"
                : config.modelType().name().toLowerCase(Locale.ROOT);
        // QWEN enum → qwen2 implementation
        if ("qwen".equals(mt)) mt = "qwen2";
        Entry e = REGISTRY.get(mt);
        if (e == null) e = REGISTRY.get("generic");
        return e;
    }

    public static Module create(PretrainedConfig config) {
        return resolve(config).factory().apply(config);
    }

    public static WeightMap weightMap(PretrainedConfig config) {
        return resolve(config).weightMap();
    }

    private static Entry must(String key) {
        Entry e = REGISTRY.get(key);
        if (e == null) throw new IllegalStateException("Missing registry entry: " + key);
        return e;
    }
}
