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
package org.bytedeco.pytorch.llm.llamafactory.chat;

import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Multi-turn conversation state for {@link ChatEngine} / OpenAI chat API.
 *
 * <p>Roles follow ChatML / OpenAI: {@code system}, {@code user}, {@code assistant}, {@code tool}.
 */
public final class Conversation {

    private final List<Template.Message> messages = new ArrayList<>();
    private String system;
    private final int maxTurns;

    public Conversation() {
        this(null, 64);
    }

    public Conversation(String system) {
        this(system, 64);
    }

    public Conversation(String system, int maxTurns) {
        this.system = system;
        this.maxTurns = Math.max(1, maxTurns);
    }

    public String system() {
        return system;
    }

    public Conversation setSystem(String system) {
        this.system = system;
        return this;
    }

    public List<Template.Message> messages() {
        return Collections.unmodifiableList(new ArrayList<>(messages));
    }

    public int size() {
        return messages.size();
    }

    public Conversation clear() {
        messages.clear();
        return this;
    }

    public Conversation add(String role, String content) {
        Objects.requireNonNull(role, "role");
        String r = role.trim().toLowerCase();
        String c = content == null ? "" : content;
        Template.Message msg = switch (r) {
            case "system" -> Template.Message.system(c);
            case "assistant" -> Template.Message.assistant(c);
            case "tool", "function" -> Template.Message.tool(c);
            default -> Template.Message.user(c);
        };
        messages.add(msg);
        trim();
        return this;
    }

    public Conversation user(String content) {
        return add("user", content);
    }

    public Conversation assistant(String content) {
        return add("assistant", content);
    }

    public Conversation systemMessage(String content) {
        return add("system", content);
    }

    /** OpenAI-style message list: [{role, content}, …]. */
    public List<Map<String, String>> toOpenAiMessages() {
        List<Map<String, String>> out = new ArrayList<>();
        if (system != null && !system.isBlank()) {
            Map<String, String> sys = new LinkedHashMap<>();
            sys.put("role", "system");
            sys.put("content", system);
            out.add(sys);
        }
        for (Template.Message m : messages) {
            if ("system".equalsIgnoreCase(m.role()) && system != null && !system.isBlank()) {
                // already emitted default system; still allow explicit system turns
            }
            Map<String, String> row = new LinkedHashMap<>();
            row.put("role", m.role());
            row.put("content", m.content());
            out.add(row);
        }
        return out;
    }

    public static Conversation fromOpenAiMessages(List<Map<String, String>> msgs) {
        Conversation c = new Conversation();
        if (msgs == null) return c;
        for (Map<String, String> m : msgs) {
            if (m == null) continue;
            String role = m.get("role");
            String content = m.get("content");
            if (role == null) continue;
            if ("system".equalsIgnoreCase(role) && c.system == null) {
                c.system = content;
            } else {
                c.add(role, content);
            }
        }
        return c;
    }

    /** Render with a template for CausalLM prompting. */
    public String render(Template template) {
        Objects.requireNonNull(template, "template");
        List<Template.Message> all = new ArrayList<>();
        if (system != null && !system.isBlank()) {
            // system override handled by encodePrompt
        }
        all.addAll(messages);
        return template.encodePrompt(all, system);
    }

    private void trim() {
        // keep last maxTurns*2 messages (user+assistant pairs)
        int limit = maxTurns * 2;
        while (messages.size() > limit) {
            messages.remove(0);
        }
    }
}
