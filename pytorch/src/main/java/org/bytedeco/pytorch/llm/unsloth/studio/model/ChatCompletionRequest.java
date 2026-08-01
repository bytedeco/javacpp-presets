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

package org.bytedeco.pytorch.llm.unsloth.studio.model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** OpenAI-style chat completion request. */
public final class ChatCompletionRequest {
    private final String model;
    private final List<ChatMessage> messages;
    private final double temperature;
    private final double topP;
    private final int maxTokens;
    private final boolean stream;
    private final double presencePenalty;
    private final double frequencyPenalty;
    private final List<Map<String, Object>> tools;
    private final String toolChoice;
    private final Map<String, Object> extra;

    private ChatCompletionRequest(Builder b) {
        this.model = b.model;
        this.messages = List.copyOf(b.messages);
        if (this.messages.isEmpty()) {
            throw new org.bytedeco.pytorch.llm.unsloth.studio.util.StudioValidationException(
                    "messages must not be empty");
        }
        this.temperature = b.temperature;
        this.topP = b.topP;
        this.maxTokens = b.maxTokens;
        this.stream = b.stream;
        this.presencePenalty = b.presencePenalty;
        this.frequencyPenalty = b.frequencyPenalty;
        this.tools = List.copyOf(b.tools);
        this.toolChoice = b.toolChoice;
        this.extra = Map.copyOf(b.extra);
    }

    public static Builder builder() { return new Builder(); }

    public static ChatCompletionRequest of(String system, String user) {
        Builder b = builder();
        if (system != null && !system.isBlank()) b.addMessage(ChatMessage.system(system));
        b.addMessage(ChatMessage.user(user));
        return b.build();
    }

    public Optional<String> model() { return Optional.ofNullable(model); }
    public List<ChatMessage> messages() { return messages; }
    public double temperature() { return temperature; }
    public double topP() { return topP; }
    public int maxTokens() { return maxTokens; }
    public boolean stream() { return stream; }
    public double presencePenalty() { return presencePenalty; }
    public double frequencyPenalty() { return frequencyPenalty; }
    public List<Map<String, Object>> tools() { return tools; }
    public Optional<String> toolChoice() { return Optional.ofNullable(toolChoice); }
    public Map<String, Object> extra() { return extra; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        if (model != null) m.put("model", model);
        List<Map<String, Object>> msgs = new ArrayList<>();
        for (ChatMessage msg : messages) msgs.add(msg.toMap());
        m.put("messages", msgs);
        m.put("temperature", temperature);
        m.put("top_p", topP);
        m.put("max_tokens", maxTokens);
        m.put("stream", stream);
        m.put("presence_penalty", presencePenalty);
        m.put("frequency_penalty", frequencyPenalty);
        if (!tools.isEmpty()) m.put("tools", tools);
        if (toolChoice != null) m.put("tool_choice", toolChoice);
        return m;
    }

    @SuppressWarnings("unchecked")
    public static ChatCompletionRequest fromMap(Map<String, Object> m) {
        Builder b = builder();
        if (m.get("model") != null) b.model(String.valueOf(m.get("model")));
        Object msgs = m.get("messages");
        if (msgs instanceof List<?> list) {
            for (Object o : list) {
                if (o instanceof Map<?, ?> mm) {
                    b.addMessage(ChatMessage.fromMap((Map<String, Object>) mm));
                }
            }
        }
        if (m.containsKey("temperature")) b.temperature(((Number) m.get("temperature")).doubleValue());
        if (m.containsKey("top_p")) b.topP(((Number) m.get("top_p")).doubleValue());
        if (m.containsKey("max_tokens")) b.maxTokens(((Number) m.get("max_tokens")).intValue());
        if (m.containsKey("stream")) b.stream(Boolean.TRUE.equals(m.get("stream")) || "true".equals(String.valueOf(m.get("stream"))));
        if (m.get("tools") instanceof List<?> tlist) {
            List<Map<String, Object>> tools = new ArrayList<>();
            for (Object o : tlist) if (o instanceof Map<?, ?> mm) tools.add((Map<String, Object>) mm);
            b.tools(tools);
        }
        return b.build();
    }

    public static final class Builder {
        private String model;
        private List<ChatMessage> messages = new ArrayList<>();
        private double temperature = 0.7;
        private double topP = 0.95;
        private int maxTokens = 256;
        private boolean stream = false;
        private double presencePenalty = 0;
        private double frequencyPenalty = 0;
        private List<Map<String, Object>> tools = List.of();
        private String toolChoice;
        private Map<String, Object> extra = Map.of();

        public Builder model(String v) { this.model = v; return this; }
        public Builder messages(List<ChatMessage> v) { this.messages = v != null ? new ArrayList<>(v) : new ArrayList<>(); return this; }
        public Builder addMessage(ChatMessage v) { this.messages.add(v); return this; }
        public Builder temperature(double v) { this.temperature = v; return this; }
        public Builder topP(double v) { this.topP = v; return this; }
        public Builder maxTokens(int v) { this.maxTokens = v; return this; }
        public Builder stream(boolean v) { this.stream = v; return this; }
        public Builder presencePenalty(double v) { this.presencePenalty = v; return this; }
        public Builder frequencyPenalty(double v) { this.frequencyPenalty = v; return this; }
        public Builder tools(List<Map<String, Object>> v) { this.tools = v != null ? v : List.of(); return this; }
        public Builder toolChoice(String v) { this.toolChoice = v; return this; }
        public Builder extra(Map<String, Object> v) { this.extra = v != null ? v : Map.of(); return this; }
        public ChatCompletionRequest build() { return new ChatCompletionRequest(this); }
    }
}
