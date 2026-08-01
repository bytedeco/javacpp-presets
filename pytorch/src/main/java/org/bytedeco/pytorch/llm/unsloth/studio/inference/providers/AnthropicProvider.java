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

package org.bytedeco.pytorch.llm.unsloth.studio.inference.providers;

import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionRequest;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatCompletionResponse;
import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage;
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Anthropic Messages API provider ({@code /v1/messages} shape). */
public final class AnthropicProvider implements ExternalProvider {

    private final String apiKey;
    private final String baseUrl;
    private final HttpClient client;

    public AnthropicProvider(String apiKey) {
        this(apiKey, "https://api.anthropic.com");
    }

    public AnthropicProvider(String apiKey, String baseUrl) {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
    }

    @Override
    public String name() { return "anthropic"; }

    @Override
    public ChatCompletionResponse chatCompletions(ChatCompletionRequest request) throws Exception {
        Map<String, Object> body = new LinkedHashMap<>();
        String model = request.model().orElse("claude-sonnet-4-20250514");
        if (model.startsWith("anthropic:")) model = model.substring("anthropic:".length());
        body.put("model", model);
        body.put("max_tokens", Math.max(1, request.maxTokens()));
        String system = null;
        List<Map<String, Object>> msgs = new ArrayList<>();
        for (ChatMessage m : request.messages()) {
            if ("system".equals(m.role())) {
                system = m.content();
            } else {
                msgs.add(Map.of("role", m.role(), "content", m.content()));
            }
        }
        if (system != null) body.put("system", system);
        body.put("messages", msgs);
        HttpRequest http = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/v1/messages"))
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .header("x-api-key", apiKey != null ? apiKey : "")
                .header("anthropic-version", "2023-06-01")
                .POST(HttpRequest.BodyPublishers.ofString(JsonMaps.stringify(body)))
                .build();
        HttpResponse<String> resp = client.send(http, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) {
            throw new IllegalStateException("anthropic HTTP " + resp.statusCode() + ": " + resp.body());
        }
        Map<String, Object> m = JsonMaps.parseObject(resp.body());
        String content = "";
        Object c = m.get("content");
        if (c instanceof List<?> list) {
            StringBuilder sb = new StringBuilder();
            for (Object o : list) {
                if (o instanceof Map<?, ?> mm && mm.get("text") != null) {
                    if (sb.length() > 0) sb.append('\n');
                    sb.append(mm.get("text"));
                }
            }
            content = sb.toString();
        }
        return ChatCompletionResponse.of(model, content);
    }
}
