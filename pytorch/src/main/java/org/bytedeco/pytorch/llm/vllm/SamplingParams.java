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
package org.bytedeco.pytorch.llm.vllm;

import org.bytedeco.pytorch.llm.transformers.generation.GenerationConfig;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * nano-vLLM / vLLM style sampling parameters for one generation request.
 *
 * <p>Maps cleanly to {@link GenerationConfig} when needed.
 */
public final class SamplingParams {

    public final double temperature;
    public final double topP;
    public final int topK;
    public final double repetitionPenalty;
    public final int maxTokens;
    public final boolean ignoreEos;
    public final List<Integer> stopTokenIds;
    public final Integer seed;
    public final boolean doSample;

    private SamplingParams(Builder b) {
        this.temperature = b.temperature;
        this.topP = b.topP;
        this.topK = b.topK;
        this.repetitionPenalty = b.repetitionPenalty;
        this.maxTokens = b.maxTokens;
        this.ignoreEos = b.ignoreEos;
        this.stopTokenIds = Collections.unmodifiableList(new ArrayList<>(b.stopTokenIds));
        this.seed = b.seed;
        this.doSample = b.doSample;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static SamplingParams greedy(int maxTokens) {
        return builder().temperature(0).doSample(false).maxTokens(maxTokens).build();
    }

    public static SamplingParams defaults() {
        return builder().build();
    }

    public static SamplingParams fromGenerationConfig(GenerationConfig gen) {
        Objects.requireNonNull(gen, "gen");
        Builder b = builder()
                .temperature(gen.temperature)
                .topP(gen.topP)
                .topK(gen.topK)
                .repetitionPenalty(gen.repetitionPenalty)
                .maxTokens(gen.maxNewTokens)
                .doSample(gen.doSample)
                .ignoreEos(!gen.eosStop);
        for (int id : gen.eosTokenIds) b.stopTokenId(id);
        return b.build();
    }

    public GenerationConfig toGenerationConfig() {
        GenerationConfig.Builder b = GenerationConfig.builder()
                .temperature(temperature)
                .topP(topP)
                .topK(topK)
                .repetitionPenalty(repetitionPenalty)
                .maxNewTokens(maxTokens)
                .doSample(doSample)
                .eosStop(!ignoreEos);
        for (int id : stopTokenIds) b.eosTokenId(id);
        return b.build();
    }

    public Builder toBuilder() {
        Builder b = builder()
                .temperature(temperature)
                .topP(topP)
                .topK(topK)
                .repetitionPenalty(repetitionPenalty)
                .maxTokens(maxTokens)
                .ignoreEos(ignoreEos)
                .doSample(doSample)
                .seed(seed);
        for (int id : stopTokenIds) b.stopTokenId(id);
        return b;
    }

    @Override
    public String toString() {
        return "SamplingParams{temp=" + temperature + ", topP=" + topP + ", topK=" + topK
                + ", maxTokens=" + maxTokens + ", doSample=" + doSample + "}";
    }

    public static final class Builder {
        private double temperature = 1.0;
        private double topP = 1.0;
        private int topK = 0;
        private double repetitionPenalty = 1.0;
        private int maxTokens = 64;
        private boolean ignoreEos = false;
        private final List<Integer> stopTokenIds = new ArrayList<>();
        private Integer seed = null;
        private boolean doSample = false;

        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topP(double v) { this.topP = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder repetitionPenalty(double v) { this.repetitionPenalty = v; return this; }
        public Builder maxTokens(int v) { this.maxTokens = v; return this; }
        public Builder ignoreEos(boolean v) { this.ignoreEos = v; return this; }
        public Builder stopTokenId(int id) { this.stopTokenIds.add(id); return this; }
        public Builder stopTokenIds(List<Integer> ids) {
            this.stopTokenIds.clear();
            if (ids != null) this.stopTokenIds.addAll(ids);
            return this;
        }
        public Builder seed(Integer v) { this.seed = v; return this; }
        public Builder doSample(boolean v) { this.doSample = v; return this; }

        public SamplingParams build() {
            if (maxTokens <= 0) throw new IllegalArgumentException("maxTokens must be > 0");
            if (temperature < 0) throw new IllegalArgumentException("temperature must be >= 0");
            // temperature==0 implies greedy
            if (temperature == 0) doSample = false;
            return new SamplingParams(this);
        }
    }
}
