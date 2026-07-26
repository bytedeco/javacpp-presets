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
package org.bytedeco.pytorch.utils.transformers.generation;

import org.bytedeco.pytorch.utils.json.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace-style generation configuration.
 */
public final class GenerationConfig {

    public final boolean doSample;
    public final double temperature;
    public final int topK;
    public final double topP;
    public final double repetitionPenalty;
    public final int maxNewTokens;
    public final boolean eosStop;
    public final List<Integer> eosTokenIds;
    public final Integer padTokenId;
    public final Integer bosTokenId;

    private GenerationConfig(Builder b) {
        this.doSample = b.doSample;
        this.temperature = b.temperature;
        this.topK = b.topK;
        this.topP = b.topP;
        this.repetitionPenalty = b.repetitionPenalty;
        this.maxNewTokens = b.maxNewTokens;
        this.eosStop = b.eosStop;
        this.eosTokenIds = Collections.unmodifiableList(new ArrayList<>(b.eosTokenIds));
        this.padTokenId = b.padTokenId;
        this.bosTokenId = b.bosTokenId;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static GenerationConfig greedy() {
        return builder().doSample(false).temperature(1.0).maxNewTokens(64).build();
    }

    public static GenerationConfig sample(double temperature, int topK) {
        return builder().doSample(true).temperature(temperature).topK(topK).maxNewTokens(64).build();
    }

    public static GenerationConfig fromJson(String json) throws IOException {
        if (json == null || json.isBlank()) return greedy();
        Map<String, Object> m = Json.decodeObject(json);
        return fromMap(m);
    }

    public static GenerationConfig fromFile(Path path) throws IOException {
        return fromJson(Files.readString(path, StandardCharsets.UTF_8));
    }

    public static GenerationConfig fromMap(Map<String, Object> m) {
        Builder b = builder();
        if (m.containsKey("do_sample")) b.doSample(asBool(m.get("do_sample")));
        if (m.containsKey("temperature")) b.temperature(asDouble(m.get("temperature")));
        if (m.containsKey("top_k")) b.topK(asInt(m.get("top_k")));
        if (m.containsKey("top_p")) b.topP(asDouble(m.get("top_p")));
        if (m.containsKey("repetition_penalty")) b.repetitionPenalty(asDouble(m.get("repetition_penalty")));
        if (m.containsKey("max_new_tokens")) b.maxNewTokens(asInt(m.get("max_new_tokens")));
        else if (m.containsKey("max_length")) b.maxNewTokens(asInt(m.get("max_length")));
        if (m.containsKey("pad_token_id") && m.get("pad_token_id") != null) b.padTokenId(asInt(m.get("pad_token_id")));
        if (m.containsKey("bos_token_id") && m.get("bos_token_id") != null) b.bosTokenId(asInt(m.get("bos_token_id")));
        Object eos = m.get("eos_token_id");
        if (eos instanceof Number) {
            b.eosTokenId(asInt(eos));
        } else if (eos instanceof List<?> list) {
            for (Object o : list) b.eosTokenId(asInt(o));
        }
        return b.build();
    }

    public String toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("do_sample", doSample);
        m.put("temperature", temperature);
        m.put("top_k", topK);
        m.put("top_p", topP);
        m.put("repetition_penalty", repetitionPenalty);
        m.put("max_new_tokens", maxNewTokens);
        if (padTokenId != null) m.put("pad_token_id", padTokenId);
        if (bosTokenId != null) m.put("bos_token_id", bosTokenId);
        if (eosTokenIds.size() == 1) m.put("eos_token_id", eosTokenIds.get(0));
        else if (!eosTokenIds.isEmpty()) m.put("eos_token_id", eosTokenIds);
        return Json.encode(m);
    }

    public Builder toBuilder() {
        Builder b = builder()
                .doSample(doSample).temperature(temperature).topK(topK).topP(topP)
                .repetitionPenalty(repetitionPenalty).maxNewTokens(maxNewTokens).eosStop(eosStop);
        for (int id : eosTokenIds) b.eosTokenId(id);
        if (padTokenId != null) b.padTokenId(padTokenId);
        if (bosTokenId != null) b.bosTokenId(bosTokenId);
        return b;
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
        private boolean doSample;
        private double temperature = 1.0;
        private int topK;
        private double topP = 1.0;
        private double repetitionPenalty = 1.0;
        private int maxNewTokens = 64;
        private boolean eosStop = true;
        private final List<Integer> eosTokenIds = new ArrayList<>();
        private Integer padTokenId;
        private Integer bosTokenId;

        public Builder doSample(boolean v) { this.doSample = v; return this; }
        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder topP(double v) { this.topP = v; return this; }
        public Builder repetitionPenalty(double v) { this.repetitionPenalty = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder eosStop(boolean v) { this.eosStop = v; return this; }
        public Builder eosTokenId(int id) { this.eosTokenIds.add(id); return this; }
        public Builder eosTokenIds(List<Integer> ids) {
            this.eosTokenIds.clear();
            if (ids != null) this.eosTokenIds.addAll(ids);
            return this;
        }
        public Builder padTokenId(int id) { this.padTokenId = id; return this; }
        public Builder bosTokenId(int id) { this.bosTokenId = id; return this; }

        public GenerationConfig build() {
            return new GenerationConfig(this);
        }
    }
}
