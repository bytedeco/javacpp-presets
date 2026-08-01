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
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Generate result with timing and cache metrics.
 */
public final class KtGenerateOutput {

    private final String requestId;
    private final int[] tokenIds;
    private final int promptTokens;
    private final int newTokens;
    private final long prefillNanos;
    private final long decodeNanos;
    private final int prefixHitTokens;
    private final Map<String, Double> metrics;

    public KtGenerateOutput(String requestId, int[] tokenIds, int promptTokens, int newTokens,
                            long prefillNanos, long decodeNanos, int prefixHitTokens,
                            Map<String, Double> metrics) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.tokenIds = Objects.requireNonNull(tokenIds, "tokenIds").clone();
        this.promptTokens = promptTokens;
        this.newTokens = newTokens;
        this.prefillNanos = prefillNanos;
        this.decodeNanos = decodeNanos;
        this.prefixHitTokens = Math.max(0, prefixHitTokens);
        this.metrics = metrics == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(metrics));
    }

    public String requestId() { return requestId; }
    public int[] tokenIds() { return tokenIds.clone(); }
    public int promptTokens() { return promptTokens; }
    public int newTokens() { return newTokens; }
    public long prefillNanos() { return prefillNanos; }
    public long decodeNanos() { return decodeNanos; }
    public int prefixHitTokens() { return prefixHitTokens; }
    public Map<String, Double> metrics() { return metrics; }

    public double prefillTokensPerSec() {
        if (prefillNanos <= 0 || promptTokens <= 0) return 0.0;
        return promptTokens * 1_000_000_000.0 / prefillNanos;
    }

    public double decodeTokensPerSec() {
        if (decodeNanos <= 0 || newTokens <= 0) return 0.0;
        return newTokens * 1_000_000_000.0 / decodeNanos;
    }

    public double ttftMillis() {
        return prefillNanos / 1_000_000.0;
    }

    public int[] newTokenIds() {
        if (newTokens <= 0 || tokenIds.length < promptTokens + newTokens) {
            return Arrays.copyOfRange(tokenIds, promptTokens, tokenIds.length);
        }
        return Arrays.copyOfRange(tokenIds, promptTokens, promptTokens + newTokens);
    }

    @Override
    public String toString() {
        return String.format(
                "KtGenerateOutput{id=%s, prompt=%d, new=%d, prefixHit=%d, ttft=%.2fms, decode=%.1f tok/s}",
                requestId, promptTokens, newTokens, prefixHitTokens, ttftMillis(), decodeTokensPerSec());
    }
}
