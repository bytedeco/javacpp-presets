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

import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Unified llama.cpp-style inference engine (in-process or process-server).
 */
public interface LlamaEngine extends AutoCloseable {

    LlamaBackend backend();

    LlamaRuntimeConfig config();

    /** Load model / spawn server. Idempotent after first success. */
    void load() throws Exception;

    boolean isLoaded();

    Optional<LlamaModel> model();

    Optional<LlamaHParams> hparams();

    /**
     * Autoregressive completion from a raw prompt string.
     * @return generated continuation (not including prompt)
     */
    String complete(String prompt, LlamaSamplingParams params) throws Exception;

    /** Token-level generate: returns full sequence (prompt tokens + new). */
    int[] generate(int[] promptTokens, LlamaSamplingParams params) throws Exception;

    /**
     * Chat-style completion. Messages are maps with {@code role}/{@code content}
     * (OpenAI shape). Implementations apply {@link LlamaChatFormatter}.
     */
    String chat(List<Map<String, String>> messages, LlamaSamplingParams params) throws Exception;

    default String chat(List<Map<String, String>> messages) throws Exception {
        return chat(messages, LlamaSamplingParams.defaults());
    }

    default String complete(String prompt) throws Exception {
        return complete(prompt, LlamaSamplingParams.defaults());
    }

    /** Reset KV / conversation state without unloading weights. */
    void reset();

    Map<String, Object> stats();

    void unload();

    @Override
    default void close() {
        unload();
    }
}
