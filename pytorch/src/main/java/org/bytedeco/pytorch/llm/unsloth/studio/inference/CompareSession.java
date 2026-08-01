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

package org.bytedeco.pytorch.llm.unsloth.studio.inference;

import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.LoadRequest;

import java.util.LinkedHashMap;
import java.util.Map;

/** Side-by-side comparison of two models on the same prompt. */
public final class CompareSession {

    private final InferenceOrchestrator orchestrator;
    private final String modelA;
    private final String modelB;

    public CompareSession(InferenceOrchestrator orchestrator, String modelA, String modelB) {
        this.orchestrator = orchestrator;
        this.modelA = modelA;
        this.modelB = modelB;
    }

    public Map<String, Object> run(ChatCompletionRequest request) throws Exception {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("model_a", modelA);
        out.put("model_b", modelB);

        orchestrator.load(LoadRequest.builder().modelPath(modelA).loadIn4bit(false).build());
        ChatCompletionRequest reqA = ChatCompletionRequest.builder()
                .model(modelA)
                .messages(request.messages())
                .temperature(request.temperature())
                .topP(request.topP())
                .maxTokens(request.maxTokens())
                .build();
        ChatCompletionResponse ra = orchestrator.chatCompletions(reqA);
        out.put("response_a", ra.toMap());
        out.put("content_a", ra.firstContent());

        orchestrator.load(LoadRequest.builder().modelPath(modelB).loadIn4bit(false).build());
        ChatCompletionRequest reqB = ChatCompletionRequest.builder()
                .model(modelB)
                .messages(request.messages())
                .temperature(request.temperature())
                .topP(request.topP())
                .maxTokens(request.maxTokens())
                .build();
        ChatCompletionResponse rb = orchestrator.chatCompletions(reqB);
        out.put("response_b", rb.toMap());
        out.put("content_b", rb.firstContent());

        out.put("same", ra.firstContent().equals(rb.firstContent()));
        return out;
    }
}
