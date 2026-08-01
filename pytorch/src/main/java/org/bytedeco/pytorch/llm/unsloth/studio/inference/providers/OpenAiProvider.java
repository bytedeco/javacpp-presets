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
import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Map;

/** OpenAI-compatible HTTP chat provider (also works for vLLM / Ollama OpenAI mode). */
public final class OpenAiProvider implements ExternalProvider {

    private final String name;
    private final String baseUrl;
    private final String apiKey;
    private final HttpClient client;

    public OpenAiProvider(String name, String baseUrl, String apiKey) {
        this.name = name != null ? name : "openai";
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
        this.client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(30)).build();
    }

    public static OpenAiProvider openai(String apiKey) {
        return new OpenAiProvider("openai", "https://api.openai.com/v1", apiKey);
    }

    public static OpenAiProvider vllm(String baseUrl) {
        return new OpenAiProvider("vllm", baseUrl, null);
    }

    public static OpenAiProvider ollama(String baseUrl) {
        return new OpenAiProvider("ollama", baseUrl != null ? baseUrl : "http://127.0.0.1:11434/v1", null);
    }

    @Override
    public String name() { return name; }

    @Override
    @SuppressWarnings("unchecked")
    public ChatCompletionResponse chatCompletions(ChatCompletionRequest request) throws Exception {
        String body = JsonMaps.stringify(request.toMap());
        HttpRequest.Builder b = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/chat/completions"))
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body));
        if (apiKey != null && !apiKey.isBlank()) {
            b.header("Authorization", "Bearer " + apiKey);
        }
        HttpResponse<String> resp = client.send(b.build(), HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) {
            throw new IllegalStateException(name + " HTTP " + resp.statusCode() + ": " + resp.body());
        }
        Map<String, Object> m = JsonMaps.parseObject(resp.body());
        // Minimal decode
        String content = "";
        Object choices = m.get("choices");
        if (choices instanceof java.util.List<?> list && !list.isEmpty() && list.get(0) instanceof Map<?, ?> c0) {
            Object msg = c0.get("message");
            if (msg instanceof Map<?, ?> mm && mm.get("content") != null) {
                content = String.valueOf(mm.get("content"));
            } else if (c0.get("text") != null) {
                content = String.valueOf(c0.get("text"));
            }
        }
        String model = m.get("model") != null ? String.valueOf(m.get("model")) : request.model().orElse(name);
        return ChatCompletionResponse.of(model, content);
    }
}
