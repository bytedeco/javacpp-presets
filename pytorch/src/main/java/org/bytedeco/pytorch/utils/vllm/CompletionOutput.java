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
package org.bytedeco.pytorch.utils.vllm;

import java.util.Arrays;
import java.util.Objects;

/** One completion candidate for a request (n=1 in Phase 1). */
public final class CompletionOutput {

    public final int index;
    public final String text;
    public final int[] tokenIds;
    public final String finishReason;
    public final double cumulativeLogprob;

    public CompletionOutput(int index, String text, int[] tokenIds, String finishReason) {
        this(index, text, tokenIds, finishReason, Double.NaN);
    }

    public CompletionOutput(int index, String text, int[] tokenIds,
                            String finishReason, double cumulativeLogprob) {
        this.index = index;
        this.text = text == null ? "" : text;
        this.tokenIds = tokenIds == null ? new int[0] : tokenIds.clone();
        this.finishReason = finishReason;
        this.cumulativeLogprob = cumulativeLogprob;
    }

    public int numTokens() { return tokenIds.length; }

    @Override
    public String toString() {
        return "CompletionOutput{index=" + index + ", tokens=" + tokenIds.length
                + ", finish=" + finishReason + ", text=" + preview(text) + "}";
    }

    private static String preview(String t) {
        if (t == null) return "";
        return t.length() <= 64 ? t : t.substring(0, 61) + "...";
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CompletionOutput that)) return false;
        return index == that.index && Arrays.equals(tokenIds, that.tokenIds)
                && Objects.equals(text, that.text)
                && Objects.equals(finishReason, that.finishReason);
    }

    @Override
    public int hashCode() {
        return Objects.hash(index, text, finishReason, Arrays.hashCode(tokenIds));
    }
}
