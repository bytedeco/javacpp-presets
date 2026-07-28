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
package org.bytedeco.pytorch.llm.transformers;

import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * HuggingFace-style model configuration (mirrors {@code PretrainedConfig}).
 *
 * <p>Carries architecture hyper-parameters for GPT-2 / Llama / Qwen / Mistral
 * style causal LMs used by {@link CausalLM}.
 */
public final class PretrainedConfig {

    public enum ModelType {
        GPT2, LLAMA, QWEN, MISTRAL, GLM, BERT, GENERIC
    }

    private final ModelType modelType;
    private final int vocabSize;
    private final int hiddenSize;
    private final int intermediateSize;
    private final int numHiddenLayers;
    private final int numAttentionHeads;
    private final int numKeyValueHeads;
    /** Explicit head dim (Qwen3 may use head_dim ≠ hidden/heads). 0 → derive. */
    private final int headDim;
    private final int maxPositionEmbeddings;
    private final double rmsNormEps;
    private final double layerNormEps;
    private final double ropeTheta;
    private final boolean tieWordEmbeddings;
    private final boolean attentionBias;
    private final int bosTokenId;
    private final int eosTokenId;
    private final int padTokenId;
    private final String torchDtype;
    private final Map<String, Object> extra;

    private PretrainedConfig(Builder b) {
        this.modelType = b.modelType;
        this.vocabSize = b.vocabSize;
        this.hiddenSize = b.hiddenSize;
        this.intermediateSize = b.intermediateSize > 0 ? b.intermediateSize : 4 * b.hiddenSize;
        this.numHiddenLayers = b.numHiddenLayers;
        this.numAttentionHeads = b.numAttentionHeads;
        this.numKeyValueHeads = b.numKeyValueHeads > 0 ? b.numKeyValueHeads : b.numAttentionHeads;
        // Qwen3 publishes head_dim independently (e.g. 128 with hidden=1024, heads=16).
        // When unset, fall back to hidden/heads for Qwen2/Llama-style models.
        this.headDim = b.headDim > 0
                ? b.headDim
                : (b.numAttentionHeads > 0 ? b.hiddenSize / b.numAttentionHeads : 0);
        this.maxPositionEmbeddings = b.maxPositionEmbeddings;
        this.rmsNormEps = b.rmsNormEps;
        this.layerNormEps = b.layerNormEps;
        this.ropeTheta = b.ropeTheta;
        this.tieWordEmbeddings = b.tieWordEmbeddings;
        this.attentionBias = b.attentionBias;
        this.bosTokenId = b.bosTokenId;
        this.eosTokenId = b.eosTokenId;
        this.padTokenId = b.padTokenId;
        this.torchDtype = b.torchDtype;
        this.extra = Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
        // Only enforce divisibility when headDim was derived (not explicit).
        if (b.headDim <= 0 && numAttentionHeads > 0 && hiddenSize % numAttentionHeads != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by numAttentionHeads");
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Tiny GPT-2 style config for unit tests / benchmarks. */
    public static PretrainedConfig tinyGpt2() {
        return builder()
                .modelType(ModelType.GPT2)
                .vocabSize(256)
                .hiddenSize(64)
                .numHiddenLayers(2)
                .numAttentionHeads(4)
                .maxPositionEmbeddings(128)
                .bosTokenId(0).eosTokenId(0).padTokenId(0)
                .build();
    }

    public static PretrainedConfig tinyLlama() {
        return builder()
                .modelType(ModelType.LLAMA)
                .vocabSize(320)
                .hiddenSize(64)
                .intermediateSize(128)
                .numHiddenLayers(2)
                .numAttentionHeads(4)
                .numKeyValueHeads(2)
                .maxPositionEmbeddings(256)
                .rmsNormEps(1e-5)
                .ropeTheta(10000.0)
                .bosTokenId(1).eosTokenId(2).padTokenId(0)
                .build();
    }

    public static PretrainedConfig tinyQwen() {
        return builder()
                .modelType(ModelType.QWEN)
                .vocabSize(512)
                .hiddenSize(64)
                .intermediateSize(128)
                .numHiddenLayers(2)
                .numAttentionHeads(4)
                .numKeyValueHeads(2)
                .maxPositionEmbeddings(256)
                .rmsNormEps(1e-6)
                .ropeTheta(1000000.0)
                .attentionBias(true)
                .bosTokenId(151643).eosTokenId(151645).padTokenId(151643)
                .build();
    }

    /** Tiny Qwen3 (explicit head_dim + qk-norm; no attention bias). */
    public static PretrainedConfig tinyQwen3() {
        return builder()
                .modelType(ModelType.QWEN)
                .vocabSize(512)
                .hiddenSize(64)
                .intermediateSize(128)
                .numHiddenLayers(2)
                .numAttentionHeads(4)
                .numKeyValueHeads(2)
                .headDim(16)
                .maxPositionEmbeddings(256)
                .rmsNormEps(1e-6)
                .ropeTheta(1000000.0)
                .attentionBias(false)
                .tieWordEmbeddings(false)
                .bosTokenId(151643).eosTokenId(151645).padTokenId(151643)
                .extra("model_type", "qwen3")
                .build();
    }

    public static PretrainedConfig tinyMistral() {
        return builder()
                .modelType(ModelType.MISTRAL)
                .vocabSize(320)
                .hiddenSize(64)
                .intermediateSize(128)
                .numHiddenLayers(2)
                .numAttentionHeads(4)
                .numKeyValueHeads(2)
                .maxPositionEmbeddings(256)
                .rmsNormEps(1e-5)
                .ropeTheta(10000.0)
                .bosTokenId(1).eosTokenId(2).padTokenId(0)
                .build();
    }

    public static PretrainedConfig fromMap(Map<String, Object> m) {
        Builder b = builder();
        if (m.containsKey("model_type")) {
            b.modelType(parseType(String.valueOf(m.get("model_type"))));
        }
        if (m.containsKey("vocab_size")) b.vocabSize(asInt(m.get("vocab_size")));
        if (m.containsKey("hidden_size") || m.containsKey("n_embd")) {
            b.hiddenSize(asInt(m.containsKey("hidden_size") ? m.get("hidden_size") : m.get("n_embd")));
        }
        if (m.containsKey("intermediate_size") || m.containsKey("n_inner")) {
            Object v = m.containsKey("intermediate_size") ? m.get("intermediate_size") : m.get("n_inner");
            if (v != null) b.intermediateSize(asInt(v));
        }
        if (m.containsKey("num_hidden_layers") || m.containsKey("n_layer")) {
            b.numHiddenLayers(asInt(m.containsKey("num_hidden_layers") ? m.get("num_hidden_layers") : m.get("n_layer")));
        }
        if (m.containsKey("num_attention_heads") || m.containsKey("n_head")) {
            b.numAttentionHeads(asInt(m.containsKey("num_attention_heads") ? m.get("num_attention_heads") : m.get("n_head")));
        }
        if (m.containsKey("num_key_value_heads")) b.numKeyValueHeads(asInt(m.get("num_key_value_heads")));
        if (m.containsKey("head_dim")) b.headDim(asInt(m.get("head_dim")));
        if (m.containsKey("max_position_embeddings") || m.containsKey("n_positions")) {
            b.maxPositionEmbeddings(asInt(m.containsKey("max_position_embeddings")
                    ? m.get("max_position_embeddings") : m.get("n_positions")));
        }
        if (m.containsKey("rms_norm_eps")) b.rmsNormEps(asDouble(m.get("rms_norm_eps")));
        if (m.containsKey("layer_norm_eps") || m.containsKey("layer_norm_epsilon")) {
            b.layerNormEps(asDouble(m.containsKey("layer_norm_eps") ? m.get("layer_norm_eps") : m.get("layer_norm_epsilon")));
        }
        if (m.containsKey("rope_theta")) b.ropeTheta(asDouble(m.get("rope_theta")));
        if (m.containsKey("tie_word_embeddings")) b.tieWordEmbeddings(asBool(m.get("tie_word_embeddings")));
        if (m.containsKey("attention_bias")) {
            b.attentionBias(asBool(m.get("attention_bias")));
        } else {
            // HF defaults: Qwen2 attention_bias=True; Qwen3/Llama=False.
            String mt = m.containsKey("model_type")
                    ? String.valueOf(m.get("model_type")).toLowerCase(Locale.ROOT) : "";
            if (mt.equals("qwen2") || mt.equals("qwen")) {
                b.attentionBias(true);
            }
        }
        if (m.containsKey("bos_token_id") && m.get("bos_token_id") != null) {
            b.bosTokenId(asInt(firstNumber(m.get("bos_token_id"))));
        }
        if (m.containsKey("eos_token_id") && m.get("eos_token_id") != null) {
            b.eosTokenId(asInt(firstNumber(m.get("eos_token_id"))));
        }
        if (m.containsKey("pad_token_id") && m.get("pad_token_id") != null) {
            b.padTokenId(asInt(firstNumber(m.get("pad_token_id"))));
        }
        if (m.containsKey("torch_dtype") && m.get("torch_dtype") != null) {
            b.torchDtype(String.valueOf(m.get("torch_dtype")));
        }
        for (Map.Entry<String, Object> e : m.entrySet()) {
            b.extra(e.getKey(), e.getValue());
        }
        return b.build();
    }

    /** Parse {@code config.json} via {@link Json} (supports nested objects / arrays). */
    public static PretrainedConfig fromJson(String json) {
        if (json == null || json.isBlank()) return tinyGpt2();
        try {
            Map<String, Object> m = Json.decodeObject(json);
            // Multimodal VL configs nest LM hyperparams under text_config (Qwen2-VL / Qwen3-VL).
            m = flattenTextConfig(m);
            return fromMap(m);
        } catch (IOException e) {
            throw new IllegalArgumentException("Invalid config.json: " + e.getMessage(), e);
        }
    }

    /**
     * Promote nested {@code text_config} fields to the top level so
     * {@link #fromMap} can read hidden_size / layers / heads for VL models
     * ({@code qwen3_vl}, {@code qwen2_vl}, …). Top-level keys win on conflict.
     */
    @SuppressWarnings("unchecked")
    static Map<String, Object> flattenTextConfig(Map<String, Object> m) {
        if (m == null) return m;
        Object tc = m.get("text_config");
        if (!(tc instanceof Map<?, ?> raw)) return m;
        Map<String, Object> text = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : raw.entrySet()) {
            if (e.getKey() != null) text.put(String.valueOf(e.getKey()), e.getValue());
        }
        if (text.isEmpty()) return m;
        Map<String, Object> out = new LinkedHashMap<>(text);
        // Preserve original top-level (architectures, vision_*, tokens, …) over text defaults
        out.putAll(m);
        // Prefer text model_type when top is a VL umbrella type
        Object topMt = m.get("model_type");
        Object textMt = text.get("model_type");
        if (topMt != null && textMt != null) {
            String t = String.valueOf(topMt).toLowerCase(Locale.ROOT);
            if (t.contains("_vl") || t.contains("vl_") || t.equals("multi_modality")
                    || t.contains("conditional")) {
                out.put("model_type", textMt);
                out.put("vl_model_type", topMt);
            }
        }
        // dtype often only on text_config as "dtype" not "torch_dtype"
        if (!out.containsKey("torch_dtype") || out.get("torch_dtype") == null) {
            Object d = text.get("dtype");
            if (d == null) d = text.get("torch_dtype");
            if (d != null) out.put("torch_dtype", d);
        }
        return out;
    }

    public static PretrainedConfig fromFile(Path path) throws IOException {
        return fromJson(Files.readString(path, StandardCharsets.UTF_8));
    }

    public String toJson() {
        return Json.encode(toMap());
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_type", modelType.name().toLowerCase(Locale.ROOT));
        m.put("vocab_size", vocabSize);
        m.put("hidden_size", hiddenSize);
        m.put("intermediate_size", intermediateSize);
        m.put("num_hidden_layers", numHiddenLayers);
        m.put("num_attention_heads", numAttentionHeads);
        m.put("num_key_value_heads", numKeyValueHeads);
        m.put("head_dim", headDim);
        m.put("max_position_embeddings", maxPositionEmbeddings);
        m.put("rms_norm_eps", rmsNormEps);
        m.put("layer_norm_eps", layerNormEps);
        m.put("rope_theta", ropeTheta);
        m.put("tie_word_embeddings", tieWordEmbeddings);
        m.put("attention_bias", attentionBias);
        m.put("bos_token_id", bosTokenId);
        m.put("eos_token_id", eosTokenId);
        m.put("pad_token_id", padTokenId);
        m.put("torch_dtype", torchDtype);
        m.putAll(extra);
        return m;
    }

    /** Attention head dimension (explicit or derived as hidden/heads). */
    public int headDim() {
        return headDim > 0 ? headDim : (numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0);
    }

    public ModelType modelType() { return modelType; }
    public int vocabSize() { return vocabSize; }
    public int hiddenSize() { return hiddenSize; }
    public int intermediateSize() { return intermediateSize; }
    public int numHiddenLayers() { return numHiddenLayers; }
    public int numAttentionHeads() { return numAttentionHeads; }
    public int numKeyValueHeads() { return numKeyValueHeads; }
    public int maxPositionEmbeddings() { return maxPositionEmbeddings; }
    public double rmsNormEps() { return rmsNormEps; }
    public double layerNormEps() { return layerNormEps; }
    public double ropeTheta() { return ropeTheta; }
    public boolean tieWordEmbeddings() { return tieWordEmbeddings; }
    /** Qwen2 uses attention bias on q/k/v; Qwen3/Llama typically do not. */
    public boolean attentionBias() { return attentionBias; }
    public int bosTokenId() { return bosTokenId; }
    public int eosTokenId() { return eosTokenId; }
    public int padTokenId() { return padTokenId; }
    public String torchDtype() { return torchDtype; }
    public Map<String, Object> extra() { return extra; }

    /** True when config looks like Qwen3 (model_type=qwen3 or architectures contain Qwen3). */
    public boolean isQwen3() {
        Object mt = extra.get("model_type");
        if (mt != null && String.valueOf(mt).toLowerCase(Locale.ROOT).contains("qwen3")) return true;
        Object archs = extra.get("architectures");
        if (archs instanceof List<?> list) {
            for (Object a : list) {
                if (String.valueOf(a).toLowerCase(Locale.ROOT).contains("qwen3")) return true;
            }
        }
        return false;
    }

    private static ModelType parseType(String s) {
        if (s == null) return ModelType.GENERIC;
        String t = s.toLowerCase(Locale.ROOT);
        if (t.contains("gpt2") || t.equals("gpt")) return ModelType.GPT2;
        if (t.contains("llama")) return ModelType.LLAMA;
        if (t.contains("qwen")) return ModelType.QWEN;
        if (t.contains("mistral") || t.contains("mixtral")) return ModelType.MISTRAL;
        if (t.contains("glm") || t.contains("chatglm")) return ModelType.GLM;
        if (t.contains("bert")) return ModelType.BERT;
        try {
            return ModelType.valueOf(t.toUpperCase(Locale.ROOT));
        } catch (Exception e) {
            return ModelType.GENERIC;
        }
    }

    /** HF often uses int or [int,…] for special token ids — take the first. */
    private static Object firstNumber(Object o) {
        if (o instanceof List<?> list && !list.isEmpty()) return list.get(0);
        return o;
    }

    private static int asInt(Object o) {
        if (o instanceof Number n) return n.intValue();
        return Integer.parseInt(String.valueOf(o));
    }

    private static double asDouble(Object o) {
        if (o instanceof Number n) return n.doubleValue();
        return Double.parseDouble(String.valueOf(o));
    }

    private static boolean asBool(Object o) {
        if (o instanceof Boolean b) return b;
        return Boolean.parseBoolean(String.valueOf(o));
    }

    public static final class Builder {
        private ModelType modelType = ModelType.GPT2;
        private int vocabSize = 50257;
        private int hiddenSize = 768;
        private int intermediateSize = 0;
        private int numHiddenLayers = 12;
        private int numAttentionHeads = 12;
        private int numKeyValueHeads = 0;
        private int headDim = 0;
        private int maxPositionEmbeddings = 1024;
        private double rmsNormEps = 1e-6;
        private double layerNormEps = 1e-5;
        private double ropeTheta = 10000.0;
        private boolean tieWordEmbeddings = true;
        private boolean attentionBias = false;
        private int bosTokenId = 50256;
        private int eosTokenId = 50256;
        private int padTokenId = 50256;
        private String torchDtype = "float32";
        private final Map<String, Object> extra = new LinkedHashMap<>();

        public Builder modelType(ModelType modelType) { this.modelType = modelType; return this; }
        public Builder vocabSize(int vocabSize) { this.vocabSize = vocabSize; return this; }
        public Builder hiddenSize(int hiddenSize) { this.hiddenSize = hiddenSize; return this; }
        public Builder intermediateSize(int intermediateSize) { this.intermediateSize = intermediateSize; return this; }
        public Builder numHiddenLayers(int numHiddenLayers) { this.numHiddenLayers = numHiddenLayers; return this; }
        public Builder numAttentionHeads(int numAttentionHeads) { this.numAttentionHeads = numAttentionHeads; return this; }
        public Builder numKeyValueHeads(int numKeyValueHeads) { this.numKeyValueHeads = numKeyValueHeads; return this; }
        public Builder headDim(int headDim) { this.headDim = headDim; return this; }
        public Builder maxPositionEmbeddings(int maxPositionEmbeddings) { this.maxPositionEmbeddings = maxPositionEmbeddings; return this; }
        public Builder rmsNormEps(double rmsNormEps) { this.rmsNormEps = rmsNormEps; return this; }
        public Builder layerNormEps(double layerNormEps) { this.layerNormEps = layerNormEps; return this; }
        public Builder ropeTheta(double ropeTheta) { this.ropeTheta = ropeTheta; return this; }
        public Builder tieWordEmbeddings(boolean tieWordEmbeddings) { this.tieWordEmbeddings = tieWordEmbeddings; return this; }
        public Builder attentionBias(boolean attentionBias) { this.attentionBias = attentionBias; return this; }
        public Builder bosTokenId(int bosTokenId) { this.bosTokenId = bosTokenId; return this; }
        public Builder eosTokenId(int eosTokenId) { this.eosTokenId = eosTokenId; return this; }
        public Builder padTokenId(int padTokenId) { this.padTokenId = padTokenId; return this; }
        public Builder torchDtype(String torchDtype) { this.torchDtype = torchDtype; return this; }
        public Builder extra(String k, Object v) { this.extra.put(k, v); return this; }

        public PretrainedConfig build() {
            return new PretrainedConfig(this);
        }
    }

    @Override
    public String toString() {
        return "PretrainedConfig{" + modelType + ", d=" + hiddenSize
                + ", L=" + numHiddenLayers + ", H=" + numAttentionHeads
                + ", V=" + vocabSize + "}";
    }
}
