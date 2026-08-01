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

package org.bytedeco.pytorch.llm.llamacpp;

import java.util.LinkedHashMap;
import java.util.Map;

/** Sampling knobs aligned with llama.cpp common sampler chain. */
public final class LlamaSamplingParams {
    private final int maxTokens;
    private final float temperature;
    private final int topK;
    private final float topP;
    private final float minP;
    private final float repeatPenalty;
    private final int repeatLastN;
    private final float presencePenalty;
    private final float frequencyPenalty;
    private final long seed;
    private final boolean greedy;

    private LlamaSamplingParams(Builder b) {
        this.maxTokens = Math.max(1, b.maxTokens);
        this.temperature = b.temperature;
        this.topK = Math.max(0, b.topK);
        this.topP = clamp01(b.topP);
        this.minP = clamp01(b.minP);
        this.repeatPenalty = b.repeatPenalty <= 0 ? 1.0f : b.repeatPenalty;
        this.repeatLastN = Math.max(0, b.repeatLastN);
        this.presencePenalty = b.presencePenalty;
        this.frequencyPenalty = b.frequencyPenalty;
        this.seed = b.seed;
        this.greedy = b.greedy || b.temperature <= 0f;
    }

    public static Builder builder() { return new Builder(); }

    public static LlamaSamplingParams greedy(int maxTokens) {
        return builder().maxTokens(maxTokens).greedy(true).temperature(0f).build();
    }

    public static LlamaSamplingParams defaults() {
        return builder().build();
    }

    public int maxTokens() { return maxTokens; }
    public float temperature() { return temperature; }
    public int topK() { return topK; }
    public float topP() { return topP; }
    public float minP() { return minP; }
    public float repeatPenalty() { return repeatPenalty; }
    public int repeatLastN() { return repeatLastN; }
    public float presencePenalty() { return presencePenalty; }
    public float frequencyPenalty() { return frequencyPenalty; }
    public long seed() { return seed; }
    public boolean greedy() { return greedy; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("max_tokens", maxTokens);
        m.put("temperature", temperature);
        m.put("top_k", topK);
        m.put("top_p", topP);
        m.put("min_p", minP);
        m.put("repeat_penalty", repeatPenalty);
        m.put("repeat_last_n", repeatLastN);
        m.put("presence_penalty", presencePenalty);
        m.put("frequency_penalty", frequencyPenalty);
        m.put("seed", seed);
        m.put("greedy", greedy);
        return m;
    }

    private static float clamp01(float v) {
        if (v < 0) return 0;
        if (v > 1) return 1;
        return v;
    }

    public static final class Builder {
        private int maxTokens = 256;
        private float temperature = 0.8f;
        private int topK = 40;
        private float topP = 0.95f;
        private float minP = 0.05f;
        private float repeatPenalty = 1.1f;
        private int repeatLastN = 64;
        private float presencePenalty = 0f;
        private float frequencyPenalty = 0f;
        private long seed = -1;
        private boolean greedy = false;

        public Builder maxTokens(int v) { this.maxTokens = v; return this; }
        public Builder temperature(float v) { this.temperature = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder topP(float v) { this.topP = v; return this; }
        public Builder minP(float v) { this.minP = v; return this; }
        public Builder repeatPenalty(float v) { this.repeatPenalty = v; return this; }
        public Builder repeatLastN(int v) { this.repeatLastN = v; return this; }
        public Builder presencePenalty(float v) { this.presencePenalty = v; return this; }
        public Builder frequencyPenalty(float v) { this.frequencyPenalty = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }
        public Builder greedy(boolean v) { this.greedy = v; return this; }
        public LlamaSamplingParams build() { return new LlamaSamplingParams(this); }
    }
}
