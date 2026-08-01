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
package org.bytedeco.pytorch.llm.llamafactory.data.template;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Chat / instruction template (mirrors LLaMA-Factory {@code data/template.py}).
 *
 * <p>Formats role turns into a single training string (or token id list via an
 * external tokenizer). Templates are pure string formatters — tokenization is
 * left to {@link org.bytedeco.pytorch.llm.tokenizers.FastTokenizer}.
 */
public final class Template {

    /** One role message. */
    public static final class Message {
        private final String role;
        private final String content;

        public Message(String role, String content) {
            this.role = role == null ? "user" : role;
            this.content = content == null ? "" : content;
        }

        public String role() { return role; }
        public String content() { return content; }

        public static Message system(String c) { return new Message("system", c); }
        public static Message user(String c) { return new Message("user", c); }
        public static Message assistant(String c) { return new Message("assistant", c); }
        public static Message tool(String c) { return new Message("tool", c); }
    }

    private final String name;
    private final String systemPrefix;
    private final String systemSuffix;
    private final String userPrefix;
    private final String userSuffix;
    private final String assistantPrefix;
    private final String assistantSuffix;
    private final String toolPrefix;
    private final String toolSuffix;
    private final String defaultSystem;
    private final List<String> stopWords;
    private final String replaceEos;
    private final boolean efficientEos;
    private final String mmToken; // e.g. "<image>"

    private Template(Builder b) {
        this.name = Objects.requireNonNull(b.name, "name");
        this.systemPrefix = b.systemPrefix;
        this.systemSuffix = b.systemSuffix;
        this.userPrefix = b.userPrefix;
        this.userSuffix = b.userSuffix;
        this.assistantPrefix = b.assistantPrefix;
        this.assistantSuffix = b.assistantSuffix;
        this.toolPrefix = b.toolPrefix;
        this.toolSuffix = b.toolSuffix;
        this.defaultSystem = b.defaultSystem;
        this.stopWords = Collections.unmodifiableList(new ArrayList<>(b.stopWords));
        this.replaceEos = b.replaceEos;
        this.efficientEos = b.efficientEos;
        this.mmToken = b.mmToken;
    }

    public String name() { return name; }
    public String defaultSystem() { return defaultSystem; }
    public List<String> stopWords() { return stopWords; }
    public String mmToken() { return mmToken; }
    public boolean efficientEos() { return efficientEos; }

    /**
     * Render a full multi-turn conversation to a single training string.
     *
     * @param messages role/content turns (system optional as first)
     * @param systemOverride optional system prompt; falls back to defaultSystem
     */
    public String encodeOneline(List<Message> messages, String systemOverride) {
        Objects.requireNonNull(messages, "messages");
        StringBuilder sb = new StringBuilder();
        String system = systemOverride != null && !systemOverride.isEmpty()
                ? systemOverride
                : defaultSystem;
        boolean systemEmitted = false;
        if (system != null && !system.isEmpty()) {
            sb.append(systemPrefix).append(system).append(systemSuffix);
            systemEmitted = true;
        }
        for (Message m : messages) {
            String role = m.role().toLowerCase(Locale.ROOT);
            if ("system".equals(role)) {
                if (!systemEmitted) {
                    sb.append(systemPrefix).append(m.content()).append(systemSuffix);
                    systemEmitted = true;
                }
                continue;
            }
            switch (role) {
                case "user", "human" -> sb.append(userPrefix).append(m.content()).append(userSuffix);
                case "assistant", "gpt", "model" ->
                        sb.append(assistantPrefix).append(m.content()).append(assistantSuffix);
                case "tool", "function" ->
                        sb.append(toolPrefix).append(m.content()).append(toolSuffix);
                case "observation" ->
                        sb.append(toolPrefix).append(m.content()).append(toolSuffix);
                default -> sb.append(userPrefix).append(m.content()).append(userSuffix);
            }
        }
        return sb.toString();
    }

    public String encodeOneline(List<Message> messages) {
        return encodeOneline(messages, null);
    }

    /**
     * Encode prompt-only (drop trailing assistant content) for generation.
     * If the last message is assistant, it is omitted so the model continues.
     */
    public String encodePrompt(List<Message> messages, String systemOverride) {
        if (messages == null || messages.isEmpty()) {
            return encodeOneline(List.of(), systemOverride);
        }
        Message last = messages.get(messages.size() - 1);
        String role = last.role().toLowerCase(Locale.ROOT);
        List<Message> prompt = new ArrayList<>(messages);
        if ("assistant".equals(role) || "gpt".equals(role) || "model".equals(role)) {
            prompt = messages.subList(0, messages.size() - 1);
        }
        String base = encodeOneline(prompt, systemOverride);
        // Open the assistant turn for completion
        return base + assistantPrefix;
    }

    /**
     * Supervised pair: returns {@code [prompt, response]} strings where response
     * is the last assistant turn and prompt is everything before it (including
     * assistant prefix so labels can mask the prompt).
     */
    public String[] encodeSupervised(List<Message> messages, String systemOverride) {
        if (messages == null || messages.isEmpty()) {
            return new String[]{"", ""};
        }
        int lastAsst = -1;
        for (int i = messages.size() - 1; i >= 0; i--) {
            String r = messages.get(i).role().toLowerCase(Locale.ROOT);
            if ("assistant".equals(r) || "gpt".equals(r) || "model".equals(r)) {
                lastAsst = i;
                break;
            }
        }
        if (lastAsst < 0) {
            return new String[]{encodeOneline(messages, systemOverride), ""};
        }
        List<Message> promptMsgs = messages.subList(0, lastAsst);
        String prompt = encodeOneline(promptMsgs, systemOverride) + assistantPrefix;
        String response = messages.get(lastAsst).content() + assistantSuffix;
        return new String[]{prompt, response};
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("name", name);
        m.put("default_system", defaultSystem);
        m.put("stop_words", stopWords);
        m.put("mm_token", mmToken);
        m.put("efficient_eos", efficientEos);
        return m;
    }

    public static Builder builder(String name) {
        return new Builder(name);
    }

    public static final class Builder {
        private final String name;
        private String systemPrefix = "";
        private String systemSuffix = "\n";
        private String userPrefix = "Human: ";
        private String userSuffix = "\n";
        private String assistantPrefix = "Assistant: ";
        private String assistantSuffix = "\n";
        private String toolPrefix = "Tool: ";
        private String toolSuffix = "\n";
        private String defaultSystem = "";
        private final List<String> stopWords = new ArrayList<>();
        private String replaceEos = null;
        private boolean efficientEos = false;
        private String mmToken = "<image>";

        private Builder(String name) {
            this.name = name;
        }

        public Builder system(String prefix, String suffix) {
            this.systemPrefix = prefix == null ? "" : prefix;
            this.systemSuffix = suffix == null ? "" : suffix;
            return this;
        }

        public Builder user(String prefix, String suffix) {
            this.userPrefix = prefix == null ? "" : prefix;
            this.userSuffix = suffix == null ? "" : suffix;
            return this;
        }

        public Builder assistant(String prefix, String suffix) {
            this.assistantPrefix = prefix == null ? "" : prefix;
            this.assistantSuffix = suffix == null ? "" : suffix;
            return this;
        }

        public Builder tool(String prefix, String suffix) {
            this.toolPrefix = prefix == null ? "" : prefix;
            this.toolSuffix = suffix == null ? "" : suffix;
            return this;
        }

        public Builder defaultSystem(String s) {
            this.defaultSystem = s == null ? "" : s;
            return this;
        }

        public Builder stopWord(String w) {
            if (w != null && !w.isEmpty()) stopWords.add(w);
            return this;
        }

        public Builder stopWords(List<String> ws) {
            if (ws != null) stopWords.addAll(ws);
            return this;
        }

        public Builder replaceEos(String s) { this.replaceEos = s; return this; }
        public Builder efficientEos(boolean v) { this.efficientEos = v; return this; }
        public Builder mmToken(String t) { this.mmToken = t == null ? "<image>" : t; return this; }

        public Template build() {
            return new Template(this);
        }
    }
}
