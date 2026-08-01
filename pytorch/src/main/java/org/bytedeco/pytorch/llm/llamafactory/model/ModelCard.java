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
package org.bytedeco.pytorch.llm.llamafactory.model;

import org.bytedeco.pytorch.llm.llamafactory.hparams.ModelArgs;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.util.LinkedHashMap;
import java.util.Map;

/** Lightweight identity card for a loaded model (board / logs / export meta). */
public final class ModelCard {
    private final String modelNameOrPath;
    private final String modelType;
    private final int vocabSize;
    private final int hiddenSize;
    private final int numLayers;
    private final int numHeads;
    private final int maxPositionEmbeddings;
    private final Map<String, Object> extra;

    private ModelCard(Builder b) {
        this.modelNameOrPath = b.modelNameOrPath == null ? "unknown" : b.modelNameOrPath;
        this.modelType = b.modelType == null ? "causal_lm" : b.modelType;
        this.vocabSize = b.vocabSize;
        this.hiddenSize = b.hiddenSize;
        this.numLayers = b.numLayers;
        this.numHeads = b.numHeads;
        this.maxPositionEmbeddings = b.maxPositionEmbeddings;
        this.extra = java.util.Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
    }

    public String modelNameOrPath() { return modelNameOrPath; }
    public String modelType() { return modelType; }
    public int vocabSize() { return vocabSize; }
    public int hiddenSize() { return hiddenSize; }
    public int numLayers() { return numLayers; }
    public int numHeads() { return numHeads; }
    public int maxPositionEmbeddings() { return maxPositionEmbeddings; }
    public Map<String, Object> extra() { return extra; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_name_or_path", modelNameOrPath);
        m.put("model_type", modelType);
        m.put("vocab_size", vocabSize);
        m.put("hidden_size", hiddenSize);
        m.put("num_layers", numLayers);
        m.put("num_heads", numHeads);
        m.put("max_position_embeddings", maxPositionEmbeddings);
        if (!extra.isEmpty()) m.put("extra", extra);
        return m;
    }

    public static ModelCard unknown() {
        return builder().modelNameOrPath("unknown").build();
    }

    public static ModelCard from(ModelArgs args, PretrainedConfig cfg) {
        Builder b = builder();
        if (args != null) {
            b.modelNameOrPath(args.modelNameOrPath());
        }
        if (cfg != null) {
            // Reflect common getters without hard-coding every field name variant
            b.vocabSize(intProp(cfg, "vocabSize", "vocab_size", 50257));
            b.hiddenSize(intProp(cfg, "hiddenSize", "nEmbd", 768));
            b.numLayers(intProp(cfg, "numHiddenLayers", "nLayer", 12));
            b.numHeads(intProp(cfg, "numAttentionHeads", "nHead", 12));
            b.maxPositionEmbeddings(intProp(cfg, "maxPositionEmbeddings", "nPositions", 1024));
            try {
                var mt = cfg.getClass().getMethod("modelType");
                Object v = mt.invoke(cfg);
                if (v != null) b.modelType(String.valueOf(v));
            } catch (ReflectiveOperationException ignored) {
                b.modelType("gpt2");
            }
        }
        return b.build();
    }

    private static int intProp(Object cfg, String camel, String alt, int def) {
        for (String name : new String[]{camel, alt}) {
            try {
                var m = cfg.getClass().getMethod(name);
                Object v = m.invoke(cfg);
                if (v instanceof Number n) return n.intValue();
            } catch (ReflectiveOperationException ignored) {
                // try next
            }
        }
        return def;
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String modelNameOrPath;
        private String modelType = "causal_lm";
        private int vocabSize;
        private int hiddenSize;
        private int numLayers;
        private int numHeads;
        private int maxPositionEmbeddings;
        private final Map<String, Object> extra = new LinkedHashMap<>();

        public Builder modelNameOrPath(String v) { this.modelNameOrPath = v; return this; }
        public Builder modelType(String v) { this.modelType = v; return this; }
        public Builder vocabSize(int v) { this.vocabSize = v; return this; }
        public Builder hiddenSize(int v) { this.hiddenSize = v; return this; }
        public Builder numLayers(int v) { this.numLayers = v; return this; }
        public Builder numHeads(int v) { this.numHeads = v; return this; }
        public Builder maxPositionEmbeddings(int v) { this.maxPositionEmbeddings = v; return this; }
        public Builder extra(String k, Object v) { this.extra.put(k, v); return this; }
        public ModelCard build() { return new ModelCard(this); }
    }
}
