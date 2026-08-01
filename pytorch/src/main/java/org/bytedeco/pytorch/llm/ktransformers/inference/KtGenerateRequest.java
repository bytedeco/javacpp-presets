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
package org.bytedeco.pytorch.llm.ktransformers.inference;

import java.util.Arrays;
import java.util.Objects;
import java.util.UUID;

/**
 * Single generate request for {@link KtInferenceEngine}.
 */
public final class KtGenerateRequest {

    private final String requestId;
    private final int[] promptTokenIds;
    private final int maxNewTokens;
    private final double temperature;
    private final double topP;
    private final int topK;
    private final long seed;
    private final boolean usePrefixCache;

    private KtGenerateRequest(Builder b) {
        this.requestId = b.requestId != null ? b.requestId : UUID.randomUUID().toString();
        this.promptTokenIds = Objects.requireNonNull(b.promptTokenIds, "promptTokenIds").clone();
        if (this.promptTokenIds.length == 0) {
            throw new IllegalArgumentException("promptTokenIds must be non-empty");
        }
        this.maxNewTokens = Math.max(1, b.maxNewTokens);
        this.temperature = b.temperature;
        this.topP = b.topP;
        this.topK = Math.max(0, b.topK);
        this.seed = b.seed;
        this.usePrefixCache = b.usePrefixCache;
    }

    public String requestId() { return requestId; }
    public int[] promptTokenIds() { return promptTokenIds.clone(); }
    public int promptLength() { return promptTokenIds.length; }
    public int maxNewTokens() { return maxNewTokens; }
    public double temperature() { return temperature; }
    public double topP() { return topP; }
    public int topK() { return topK; }
    public long seed() { return seed; }
    public boolean usePrefixCache() { return usePrefixCache; }

    public static Builder builder() { return new Builder(); }

    public static KtGenerateRequest of(int[] prompt, int maxNew) {
        return builder().promptTokenIds(prompt).maxNewTokens(maxNew).build();
    }

    public static final class Builder {
        private String requestId;
        private int[] promptTokenIds;
        private int maxNewTokens = 16;
        private double temperature = 0.0;
        private double topP = 1.0;
        private int topK = 0;
        private long seed = 0L;
        private boolean usePrefixCache = true;

        public Builder requestId(String v) { this.requestId = v; return this; }
        public Builder promptTokenIds(int[] v) { this.promptTokenIds = v; return this; }
        public Builder maxNewTokens(int v) { this.maxNewTokens = v; return this; }
        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topP(double v) { this.topP = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder seed(long v) { this.seed = v; return this; }
        public Builder usePrefixCache(boolean v) { this.usePrefixCache = v; return this; }

        public KtGenerateRequest build() { return new KtGenerateRequest(this); }
    }

    @Override
    public String toString() {
        return "KtGenerateRequest{id=" + requestId
                + ", promptLen=" + promptTokenIds.length
                + ", maxNew=" + maxNewTokens
                + ", prompt=" + Arrays.toString(Arrays.copyOf(promptTokenIds,
                Math.min(8, promptTokenIds.length))) + "}";
    }
}
