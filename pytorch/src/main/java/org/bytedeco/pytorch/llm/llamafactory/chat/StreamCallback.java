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
package org.bytedeco.pytorch.llm.llamafactory.chat;

/**
 * Streaming token / text callback for chat / OpenAI SSE responses.
 */
@FunctionalInterface
public interface StreamCallback {

    /**
     * Called for each generated text chunk (may be a single token or a span).
     *
     * @param chunk non-null text fragment (may be empty for heartbeat)
     * @return {@code false} to request cooperative cancel of generation
     */
    boolean onChunk(String chunk);

    /** Called once when generation finishes successfully. */
    default void onComplete(String fullText) {}

    /** Called when generation fails. */
    default void onError(Throwable error) {}

    /** No-op sink (benchmarks / dry-run). */
    static StreamCallback noop() {
        return chunk -> true;
    }

    /** Collects chunks into a StringBuilder. */
    static StreamCallback collecting(StringBuilder sink) {
        return chunk -> {
            if (chunk != null && sink != null) {
                sink.append(chunk);
            }
            return true;
        };
    }
}
