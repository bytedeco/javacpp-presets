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

import java.util.Collections;
import java.util.List;
import java.util.Objects;

/** Finished (or partial) result for one generation request. */
public final class RequestOutput {

    public final long requestId;
    public final String prompt;
    public final int[] promptTokenIds;
    public final List<CompletionOutput> outputs;
    public final boolean finished;
    public final double ttftMs;
    public final double latencyMs;
    public final int promptTokens;
    public final int generatedTokens;

    public RequestOutput(long requestId, String prompt, int[] promptTokenIds,
                         List<CompletionOutput> outputs, boolean finished,
                         double ttftMs, double latencyMs) {
        this.requestId = requestId;
        this.prompt = prompt == null ? "" : prompt;
        this.promptTokenIds = promptTokenIds == null ? new int[0] : promptTokenIds.clone();
        this.outputs = outputs == null ? List.of() : Collections.unmodifiableList(outputs);
        this.finished = finished;
        this.ttftMs = ttftMs;
        this.latencyMs = latencyMs;
        this.promptTokens = this.promptTokenIds.length;
        this.generatedTokens = this.outputs.stream().mapToInt(CompletionOutput::numTokens).sum();
    }

    public static RequestOutput fromSequence(Sequence seq, String decodedText) {
        Objects.requireNonNull(seq, "seq");
        CompletionOutput co = new CompletionOutput(
                0, decodedText, seq.outputTokenIdsArray(), seq.finishReason());
        return new RequestOutput(
                seq.requestId(),
                seq.promptText() == null ? "" : seq.promptText(),
                seq.promptTokenIds(),
                List.of(co),
                seq.isFinished(),
                seq.ttftMs(),
                seq.latencyMs());
    }

    /** Convenience: first completion text. */
    public String text() {
        return outputs.isEmpty() ? "" : outputs.get(0).text;
    }

    @Override
    public String toString() {
        return "RequestOutput{id=" + requestId + ", finished=" + finished
                + ", promptTokens=" + promptTokens + ", genTokens=" + generatedTokens
                + ", ttftMs=" + String.format("%.1f", ttftMs)
                + ", latencyMs=" + String.format("%.1f", latencyMs)
                + ", text=" + textPreview() + "}";
    }

    private String textPreview() {
        String t = text();
        return t.length() <= 48 ? t : t.substring(0, 45) + "...";
    }
}
