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
package org.bytedeco.pytorch.llm.ktransformers.serve;

import org.bytedeco.pytorch.llm.ktransformers.KTransformersVersion;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtGenerateOutput;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtGenerateRequest;
import org.bytedeco.pytorch.llm.ktransformers.inference.KtInferenceEngine;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Minimal OpenAI-compatible chat/completions handler over {@link KtInferenceEngine}.
 *
 * <p>Not an HTTP server — host meshes call {@link #chatCompletions(ChatRequest)} and
 * serialize the returned map (factory.api style). Tokenization is a char-mod vocab
 * hash suitable for mini models; production hosts inject a real tokenizer.
 */
public final class KtOpenAiHandler {

    private final KtInferenceEngine engine;
    private final AtomicLong idSeq = new AtomicLong();
    private final String modelName;

    public KtOpenAiHandler(KtInferenceEngine engine) {
        this.engine = Objects.requireNonNull(engine, "engine");
        String n = engine.config().modelNameOrPath();
        this.modelName = n != null && !n.isBlank() ? n : "ktransformers-java";
    }

    public KtInferenceEngine engine() { return engine; }
    public String modelName() { return modelName; }

    /** OpenAI-ish chat request (messages + sampling). */
    public static final class ChatRequest {
        public final List<Map<String, String>> messages;
        public final int maxTokens;
        public final double temperature;
        public final String model;

        public ChatRequest(List<Map<String, String>> messages, int maxTokens,
                           double temperature, String model) {
            this.messages = messages != null ? messages : List.of();
            this.maxTokens = Math.max(1, maxTokens);
            this.temperature = temperature;
            this.model = model;
        }

        public static ChatRequest ofUser(String user, int maxTokens) {
            List<Map<String, String>> msgs = new ArrayList<>();
            Map<String, String> m = new LinkedHashMap<>();
            m.put("role", "user");
            m.put("content", user != null ? user : "");
            msgs.add(m);
            return new ChatRequest(msgs, maxTokens, 0.0, null);
        }
    }

    public Map<String, Object> chatCompletions(ChatRequest req) {
        Objects.requireNonNull(req, "req");
        String promptText = flattenMessages(req.messages);
        int[] prompt = tokenize(promptText, engine.config().vocabSize());
        int maxNew = req.maxTokens;
        KtGenerateOutput out = engine.generate(KtGenerateRequest.of(prompt, maxNew));
        String content = detokenize(out.tokenIds(), out.promptTokens());

        Map<String, Object> choiceMsg = new LinkedHashMap<>();
        choiceMsg.put("role", "assistant");
        choiceMsg.put("content", content);

        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", choiceMsg);
        choice.put("finish_reason", "stop");

        Map<String, Object> usage = new LinkedHashMap<>();
        usage.put("prompt_tokens", out.promptTokens());
        usage.put("completion_tokens", out.newTokens());
        usage.put("total_tokens", out.promptTokens() + out.newTokens());

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", "chatcmpl-kt-" + idSeq.incrementAndGet());
        resp.put("object", "chat.completion");
        resp.put("created", System.currentTimeMillis() / 1000L);
        resp.put("model", req.model != null ? req.model : modelName);
        resp.put("choices", List.of(choice));
        resp.put("usage", usage);
        resp.put("kt_version", KTransformersVersion.VERSION);
        resp.put("kt_metrics", out.metrics());
        return resp;
    }

    public Map<String, Object> chatCompletions(String userMessage, int maxTokens) {
        return chatCompletions(ChatRequest.ofUser(userMessage, maxTokens));
    }

    public Map<String, Object> models() {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("id", modelName);
        data.put("object", "model");
        data.put("owned_by", "ktransformers-java");
        Map<String, Object> root = new LinkedHashMap<>();
        root.put("object", "list");
        root.put("data", List.of(data));
        return root;
    }

    static String flattenMessages(List<Map<String, String>> messages) {
        if (messages == null || messages.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            if (m == null) continue;
            String role = m.getOrDefault("role", "user");
            String content = m.getOrDefault("content", "");
            sb.append(role).append(": ").append(content).append('\n');
        }
        return sb.toString();
    }

    /** Deterministic mini tokenizer: char codes mod vocab (CI-safe). */
    static int[] tokenize(String text, int vocabSize) {
        int V = Math.max(2, vocabSize);
        if (text == null || text.isEmpty()) {
            return new int[]{1};
        }
        int n = Math.min(text.length(), 64);
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) {
            ids[i] = Math.floorMod(text.charAt(i), V);
            if (ids[i] == 0) ids[i] = 1;
        }
        return ids;
    }

    static String detokenize(int[] tokenIds, int promptTokens) {
        if (tokenIds == null || tokenIds.length == 0) return "";
        int start = Math.min(Math.max(0, promptTokens), tokenIds.length);
        StringBuilder sb = new StringBuilder();
        sb.append("kt-tokens:");
        for (int i = start; i < tokenIds.length; i++) {
            if (i > start) sb.append(',');
            sb.append(tokenIds[i]);
        }
        if (start >= tokenIds.length) {
            // full sequence fallback
            sb.setLength(0);
            sb.append("kt-tokens:");
            for (int i = 0; i < tokenIds.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(tokenIds[i]);
            }
        }
        return sb.toString();
    }

    /** Pretty JSON-ish dump for demos without a JSON lib. */
    public static String toJsonish(Map<String, Object> map) {
        return stringify(map);
    }

    @SuppressWarnings("unchecked")
    private static String stringify(Object o) {
        if (o == null) return "null";
        if (o instanceof String) return "\"" + escape((String) o) + "\"";
        if (o instanceof Number || o instanceof Boolean) return String.valueOf(o);
        if (o instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) o;
            StringBuilder sb = new StringBuilder("{");
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                sb.append('"').append(escape(String.valueOf(e.getKey()))).append("\":");
                sb.append(stringify(e.getValue()));
            }
            sb.append('}');
            return sb.toString();
        }
        if (o instanceof List) {
            List<?> list = (List<?>) o;
            StringBuilder sb = new StringBuilder("[");
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(stringify(list.get(i)));
            }
            sb.append(']');
            return sb.toString();
        }
        return "\"" + escape(String.valueOf(o)) + "\"";
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"")
                .replace("\n", "\\n").replace("\r", "\\r");
    }

    @Override
    public String toString() {
        return String.format(Locale.ROOT, "KtOpenAiHandler{model=%s}", modelName);
    }
}
