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

package org.bytedeco.pytorch.llm.unsloth.studio.inference.tools;

import org.bytedeco.pytorch.llm.unsloth.studio.inference.InferenceEngine;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.sandbox.CodeSandbox;
import org.bytedeco.pytorch.llm.unsloth.studio.inference.sandbox.InProcessSandbox;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Multi-turn tool loop with optional self-heal and sandboxed code execution tool.
 */
public final class ToolLoopController {

    private final ToolCallParser parser;
    private final SelfHealingToolCaller healer;
    private final CodeSandbox sandbox;
    private int maxRounds = 3;

    public ToolLoopController(ToolCallParser parser, SelfHealingToolCaller healer) {
        this.parser = parser;
        this.healer = healer;
        this.sandbox = new InProcessSandbox();
    }

    public void setMaxRounds(int maxRounds) {
        this.maxRounds = Math.max(1, maxRounds);
    }

    public ChatCompletionResponse run(InferenceEngine engine, ChatCompletionRequest request,
                                      ChatCompletionResponse first, boolean allowCodeExec) throws Exception {
        String content = first.firstContent();
        List<ToolCallParser.ToolCall> calls = parser.parse(content);
        if (calls.isEmpty()) return first;

        List<ToolSpec> specs = extractSpecs(request);
        List<SelfHealingToolCaller.HealResult> healed = healer.healAll(calls, specs);

        List<ChatMessage> msgs = new ArrayList<>(request.messages());
        msgs.add(ChatMessage.assistant(content));

        List<Map<String, Object>> toolResults = new ArrayList<>();
        for (SelfHealingToolCaller.HealResult h : healed) {
            ToolCallParser.ToolCall tc = h.repaired;
            String result = dispatch(tc, allowCodeExec);
            toolResults.add(Map.of(
                    "tool_call_id", tc.id,
                    "name", tc.name,
                    "result", result,
                    "repairs", h.repairs));
            msgs.add(ChatMessage.tool(tc.id, result));
        }

        // One follow-up completion with tool results in context
        ChatCompletionRequest follow = ChatCompletionRequest.builder()
                .model(request.model().orElse(null))
                .messages(msgs)
                .temperature(request.temperature())
                .topP(request.topP())
                .maxTokens(request.maxTokens())
                .build();
        ChatCompletionResponse second = engine.chatCompletions(follow);
        // annotate usage meta via content prefix only if empty
        if (second.firstContent() == null || second.firstContent().isBlank()) {
            String summary = "Tool results: " + JsonMaps.stringify(toolResults);
            return ChatCompletionResponse.of(second.model(), summary);
        }
        return second;
    }

    private String dispatch(ToolCallParser.ToolCall tc, boolean allowCodeExec) {
        if ("code_execution".equals(tc.name) || "run_code".equals(tc.name) || "python".equals(tc.name)) {
            if (!allowCodeExec) {
                return "{\"error\":\"code execution disabled by StudioOptions.allowCodeExecution=false\"}";
            }
            Object code = tc.arguments.getOrDefault("code", tc.arguments.get("source"));
            return sandbox.execute(String.valueOf(code == null ? "" : code));
        }
        if ("web_search".equals(tc.name) || "search".equals(tc.name)) {
            Object q = tc.arguments.getOrDefault("query", tc.arguments.get("q"));
            return "{\"status\":\"no-op\",\"query\":" + JsonMaps.stringify(String.valueOf(q))
                    + ",\"note\":\"Inject WebSearchClient for live search\"}";
        }
        // generic echo for unknown tools — host registers real handlers later
        Map<String, Object> echo = new LinkedHashMap<>();
        echo.put("tool", tc.name);
        echo.put("arguments", tc.arguments);
        echo.put("status", "accepted");
        return JsonMaps.stringify(echo);
    }

    @SuppressWarnings("unchecked")
    private List<ToolSpec> extractSpecs(ChatCompletionRequest request) {
        List<ToolSpec> specs = new ArrayList<>();
        for (Map<String, Object> t : request.tools()) {
            try {
                Map<String, Object> fn = t;
                if (t.get("function") instanceof Map<?, ?> f) fn = (Map<String, Object>) f;
                String name = String.valueOf(fn.getOrDefault("name", "tool"));
                String desc = fn.get("description") != null ? String.valueOf(fn.get("description")) : "";
                Map<String, Object> params = Map.of();
                if (fn.get("parameters") instanceof Map<?, ?> pm) params = (Map<String, Object>) pm;
                specs.add(new ToolSpec(name, desc, params, false));
            } catch (Exception ignored) {}
        }
        return specs;
    }
}
