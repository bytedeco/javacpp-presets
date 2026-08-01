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
package org.bytedeco.pytorch.llm.llamafactory.api;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/**
 * OpenAI-compatible request / response DTOs (plain maps + helpers, no Jackson).
 *
 * <p>Wire format mirrors the subset used by LLaMA-Factory / vLLM:
 * {@code /v1/chat/completions}, {@code /v1/completions}, {@code /v1/models}.
 */
public final class OpenAiTypes {

    private OpenAiTypes() {}

    public static final class ChatMessage {
        public final String role;
        public final String content;

        public ChatMessage(String role, String content) {
            this.role = role == null ? "user" : role;
            this.content = content == null ? "" : content;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("role", role);
            m.put("content", content);
            return m;
        }

        @SuppressWarnings("unchecked")
        public static ChatMessage fromMap(Map<String, Object> m) {
            if (m == null) return new ChatMessage("user", "");
            return new ChatMessage(str(m.get("role"), "user"), str(m.get("content"), ""));
        }
    }

    public static final class ChatCompletionRequest {
        public final String model;
        public final List<ChatMessage> messages;
        public final double temperature;
        public final double topP;
        public final int maxTokens;
        public final boolean stream;
        public final String user;

        public ChatCompletionRequest(
                String model,
                List<ChatMessage> messages,
                double temperature,
                double topP,
                int maxTokens,
                boolean stream,
                String user) {
            this.model = model == null ? "default" : model;
            this.messages = messages == null ? List.of() : List.copyOf(messages);
            this.temperature = temperature;
            this.topP = topP;
            this.maxTokens = maxTokens;
            this.stream = stream;
            this.user = user;
        }

        @SuppressWarnings("unchecked")
        public static ChatCompletionRequest fromMap(Map<String, Object> body) {
            if (body == null) body = Map.of();
            String model = str(body.get("model"), "default");
            List<ChatMessage> msgs = new ArrayList<>();
            Object raw = body.get("messages");
            if (raw instanceof List<?> list) {
                for (Object o : list) {
                    if (o instanceof Map<?, ?> m) {
                        msgs.add(ChatMessage.fromMap((Map<String, Object>) m));
                    }
                }
            }
            double temp = dbl(body.get("temperature"), 0.95);
            double topP = dbl(body.get("top_p"), 0.7);
            int maxTok = (int) dbl(body.get("max_tokens"),
                    dbl(body.get("max_new_tokens"), 256));
            boolean stream = bool(body.get("stream"), false);
            String user = strOrNull(body.get("user"));
            return new ChatCompletionRequest(model, msgs, temp, topP, maxTok, stream, user);
        }
    }

    public static final class CompletionRequest {
        public final String model;
        public final String prompt;
        public final double temperature;
        public final double topP;
        public final int maxTokens;
        public final boolean stream;

        public CompletionRequest(
                String model, String prompt, double temperature, double topP,
                int maxTokens, boolean stream) {
            this.model = model == null ? "default" : model;
            this.prompt = prompt == null ? "" : prompt;
            this.temperature = temperature;
            this.topP = topP;
            this.maxTokens = maxTokens;
            this.stream = stream;
        }

        public static CompletionRequest fromMap(Map<String, Object> body) {
            if (body == null) body = Map.of();
            String prompt;
            Object p = body.get("prompt");
            if (p instanceof List<?> list && !list.isEmpty()) {
                prompt = String.valueOf(list.get(0));
            } else {
                prompt = str(p, "");
            }
            return new CompletionRequest(
                    str(body.get("model"), "default"),
                    prompt,
                    dbl(body.get("temperature"), 0.95),
                    dbl(body.get("top_p"), 0.7),
                    (int) dbl(body.get("max_tokens"), 256),
                    bool(body.get("stream"), false));
        }
    }

    public static Map<String, Object> chatCompletionResponse(
            String model, String content, int promptTokens, int completionTokens) {
        String id = "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", content == null ? "" : content);

        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", "stop");

        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", Math.max(0, promptTokens));
        usage.put("completion_tokens", Math.max(0, completionTokens));
        usage.put("total_tokens", Math.max(0, promptTokens) + Math.max(0, completionTokens));

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", id);
        resp.put("object", "chat.completion");
        resp.put("created", System.currentTimeMillis() / 1000L);
        resp.put("model", model == null ? "default" : model);
        resp.put("choices", List.of(choice));
        resp.put("usage", usage);
        return resp;
    }

    public static Map<String, Object> completionResponse(
            String model, String text, int promptTokens, int completionTokens) {
        String id = "cmpl-" + UUID.randomUUID().toString().replace("-", "").substring(0, 12);
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("text", text == null ? "" : text);
        choice.put("finish_reason", "stop");

        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", Math.max(0, promptTokens));
        usage.put("completion_tokens", Math.max(0, completionTokens));
        usage.put("total_tokens", Math.max(0, promptTokens) + Math.max(0, completionTokens));

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", id);
        resp.put("object", "text_completion");
        resp.put("created", System.currentTimeMillis() / 1000L);
        resp.put("model", model == null ? "default" : model);
        resp.put("choices", List.of(choice));
        resp.put("usage", usage);
        return resp;
    }

    public static Map<String, Object> modelsList(String modelId) {
        Map<String, Object> model = new LinkedHashMap<>();
        model.put("id", modelId == null ? "default" : modelId);
        model.put("object", "model");
        model.put("created", System.currentTimeMillis() / 1000L);
        model.put("owned_by", "llamafactory-java");
        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("object", "list");
        resp.put("data", List.of(model));
        return resp;
    }

    public static Map<String, Object> error(String message, String type, int code) {
        Map<String, Object> err = new LinkedHashMap<>();
        err.put("message", message == null ? "error" : message);
        err.put("type", type == null ? "invalid_request_error" : type);
        err.put("code", code);
        return Map.of("error", err);
    }

    /** SSE chunk for streaming chat (one delta). */
    public static Map<String, Object> chatChunk(String model, String delta, boolean done) {
        String id = "chatcmpl-stream";
        Map<String, Object> deltaMap = new LinkedHashMap<>();
        if (!done) {
            deltaMap.put("content", delta == null ? "" : delta);
            deltaMap.put("role", "assistant");
        }
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", deltaMap);
        choice.put("finish_reason", done ? "stop" : null);

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", id);
        resp.put("object", "chat.completion.chunk");
        resp.put("created", System.currentTimeMillis() / 1000L);
        resp.put("model", model == null ? "default" : model);
        resp.put("choices", List.of(choice));
        return resp;
    }

    static String str(Object o, String def) {
        return o == null ? def : String.valueOf(o);
    }

    static String strOrNull(Object o) {
        return o == null ? null : String.valueOf(o);
    }

    static double dbl(Object o, double def) {
        if (o instanceof Number n) return n.doubleValue();
        if (o == null) return def;
        try { return Double.parseDouble(String.valueOf(o)); }
        catch (Exception e) { return def; }
    }

    static boolean bool(Object o, boolean def) {
        if (o instanceof Boolean b) return b;
        if (o == null) return def;
        String s = String.valueOf(o).trim().toLowerCase();
        if ("true".equals(s) || "1".equals(s) || "yes".equals(s)) return true;
        if ("false".equals(s) || "0".equals(s) || "no".equals(s)) return false;
        return def;
    }
}
