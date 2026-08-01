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

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal OpenAI-compatible client for llama-server ({@code /v1/chat/completions}, {@code /health}).
 */
public final class LlamaServerClient {

    private final String baseUrl;
    private final HttpClient http;

    public LlamaServerClient(String host, int port) {
        this("http://" + host + ":" + port);
    }

    public LlamaServerClient(String baseUrl) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    public String baseUrl() { return baseUrl; }

    public boolean healthy() {
        try {
            HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/health"))
                    .timeout(Duration.ofSeconds(5)).GET().build();
            HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            return resp.statusCode() >= 200 && resp.statusCode() < 500;
        } catch (Exception e) {
            // fallback: try /v1/models
            try {
                HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/v1/models"))
                        .timeout(Duration.ofSeconds(5)).GET().build();
                HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
                return resp.statusCode() == 200;
            } catch (Exception e2) {
                return false;
            }
        }
    }

    public String chatCompletions(List<Map<String, String>> messages, LlamaSamplingParams params) throws Exception {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("messages", messages);
        body.put("max_tokens", params.maxTokens());
        body.put("temperature", params.greedy() ? 0.0 : params.temperature());
        body.put("top_p", params.topP());
        body.put("stream", false);
        String json = toJson(body);
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/v1/chat/completions"))
                .timeout(Duration.ofSeconds(300))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) {
            throw new IllegalStateException("llama-server HTTP " + resp.statusCode() + ": " + resp.body());
        }
        return extractContent(resp.body());
    }

    public String completions(String prompt, LlamaSamplingParams params) throws Exception {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("prompt", prompt);
        body.put("n_predict", params.maxTokens());
        body.put("temperature", params.greedy() ? 0.0 : params.temperature());
        body.put("top_p", params.topP());
        body.put("top_k", params.topK());
        body.put("repeat_penalty", params.repeatPenalty());
        String json = toJson(body);
        // llama-server native completion endpoint
        HttpRequest req = HttpRequest.newBuilder(URI.create(baseUrl + "/completion"))
                .timeout(Duration.ofSeconds(300))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
        HttpResponse<String> resp = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (resp.statusCode() >= 400) {
            // fallback openai completions
            Map<String, Object> ob = new LinkedHashMap<>();
            ob.put("prompt", prompt);
            ob.put("max_tokens", params.maxTokens());
            ob.put("temperature", params.greedy() ? 0.0 : params.temperature());
            req = HttpRequest.newBuilder(URI.create(baseUrl + "/v1/completions"))
                    .timeout(Duration.ofSeconds(300))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(toJson(ob)))
                    .build();
            resp = http.send(req, HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() >= 400) {
                throw new IllegalStateException("llama-server completion HTTP " + resp.statusCode() + ": " + resp.body());
            }
        }
        String bodyStr = resp.body();
        String content = extractField(bodyStr, "content");
        if (content == null) content = extractContent(bodyStr);
        return content != null ? content : bodyStr;
    }

    private static String extractContent(String json) {
        // choices[0].message.content or choices[0].text
        String c = extractNested(json, "\"content\"");
        if (c != null) return c;
        return extractNested(json, "\"text\"");
    }

    private static String extractField(String json, String field) {
        return extractNested(json, "\"" + field + "\"");
    }

    private static String extractNested(String json, String key) {
        if (json == null) return null;
        int i = json.indexOf(key);
        if (i < 0) return null;
        int colon = json.indexOf(':', i + key.length());
        if (colon < 0) return null;
        int j = colon + 1;
        while (j < json.length() && Character.isWhitespace(json.charAt(j))) j++;
        if (j >= json.length()) return null;
        if (json.charAt(j) == '"') {
            j++;
            StringBuilder sb = new StringBuilder();
            while (j < json.length()) {
                char ch = json.charAt(j++);
                if (ch == '\\' && j < json.length()) {
                    char n = json.charAt(j++);
                    switch (n) {
                        case 'n' -> sb.append('\n');
                        case 't' -> sb.append('\t');
                        case 'r' -> sb.append('\r');
                        case '"' -> sb.append('"');
                        case '\\' -> sb.append('\\');
                        default -> sb.append(n);
                    }
                } else if (ch == '"') break;
                else sb.append(ch);
            }
            return sb.toString();
        }
        return null;
    }

    @SuppressWarnings("unchecked")
    private static String toJson(Object value) {
        // local minimal encoder (studio JsonMaps may not be on minimal classpath tests)
        if (value == null) return "null";
        if (value instanceof String s) return "\"" + escape(s) + "\"";
        if (value instanceof Number || value instanceof Boolean) return String.valueOf(value);
        if (value instanceof Map<?, ?> map) {
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> e : map.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                sb.append(toJson(String.valueOf(e.getKey()))).append(':').append(toJson(e.getValue()));
            }
            return sb.append('}').toString();
        }
        if (value instanceof List<?> list) {
            StringBuilder sb = new StringBuilder("[");
            boolean first = true;
            for (Object o : list) {
                if (!first) sb.append(',');
                first = false;
                sb.append(toJson(o));
            }
            return sb.append(']').toString();
        }
        return toJson(String.valueOf(value));
    }

    private static String escape(String s) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> sb.append(c);
            }
        }
        return sb.toString();
    }
}
