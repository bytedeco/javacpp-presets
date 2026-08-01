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

package org.bytedeco.pytorch.llm.llamacpp.studio;

import org.bytedeco.pytorch.llm.llamacpp.LlamaBackend;
import org.bytedeco.pytorch.llm.llamacpp.LlamaCpp;
import org.bytedeco.pytorch.llm.llamacpp.LlamaEngine;
import org.bytedeco.pytorch.llm.llamacpp.LlamaRuntimeConfig;
import org.bytedeco.pytorch.llm.llamacpp.LlamaSamplingParams;
import org.bytedeco.pytorch.llm.unsloth.studio.hardware.GgufHardwareControls;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.gguf.GgufRuntime;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Bridges Studio {@link GgufRuntime} SPI to enterprise {@link LlamaEngine}. */
public final class StudioGgufRuntimeAdapter implements GgufRuntime {

    private final LlamaBackend preferred;
    private LlamaEngine engine;

    public StudioGgufRuntimeAdapter() {
        this(LlamaBackend.AUTO);
    }

    public StudioGgufRuntimeAdapter(LlamaBackend preferred) {
        this.preferred = preferred != null ? preferred : LlamaBackend.AUTO;
    }

    @Override
    public void load(Path model, GgufHardwareControls controls) throws Exception {
        unload();
        LlamaRuntimeConfig.Builder b = LlamaRuntimeConfig.builder()
                .modelPath(model)
                .backend(preferred)
                .serverPort(0);
        if (controls != null) b.fromStudioHardware(controls);
        engine = LlamaCpp.open(b.build());
        engine.load();
    }

    @Override
    public ChatCompletionResponse chat(ChatCompletionRequest request) throws Exception {
        if (engine == null || !engine.isLoaded()) {
            throw new IllegalStateException("GGUF runtime not loaded");
        }
        List<Map<String, String>> msgs = new ArrayList<>();
        for (ChatMessage m : request.messages()) {
            Map<String, String> row = new LinkedHashMap<>();
            row.put("role", m.role());
            row.put("content", m.content());
            msgs.add(row);
        }
        LlamaSamplingParams sp = LlamaSamplingParams.builder()
                .maxTokens(Math.max(1, request.maxTokens()))
                .temperature((float) request.temperature())
                .topP((float) request.topP())
                .greedy(request.temperature() <= 0)
                .build();
        String content = engine.chat(msgs, sp);
        String modelId = engine.model().map(m -> m.path().getFileName().toString()).orElse("gguf");
        return ChatCompletionResponse.of(modelId, content);
    }

    public LlamaEngine engine() { return engine; }

    @Override
    public void unload() {
        if (engine != null) {
            engine.unload();
            engine = null;
        }
    }
}
