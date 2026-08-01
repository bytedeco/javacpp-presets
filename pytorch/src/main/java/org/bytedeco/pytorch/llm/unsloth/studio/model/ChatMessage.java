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

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public final class ChatMessage {
    private final String role;
    private final String content;
    private final String name;
    private final String toolCallId;

    public ChatMessage(String role, String content) {
        this(role, content, null, null);
    }

    public ChatMessage(String role, String content, String name, String toolCallId) {
        this.role = Objects.requireNonNull(role, "role");
        this.content = content != null ? content : "";
        this.name = name;
        this.toolCallId = toolCallId;
    }

    public static ChatMessage system(String content) { return new ChatMessage("system", content); }
    public static ChatMessage user(String content) { return new ChatMessage("user", content); }
    public static ChatMessage assistant(String content) { return new ChatMessage("assistant", content); }
    public static ChatMessage tool(String toolCallId, String content) {
        return new ChatMessage("tool", content, null, toolCallId);
    }

    public String role() { return role; }
    public String content() { return content; }
    public Optional<String> name() { return Optional.ofNullable(name); }
    public Optional<String> toolCallId() { return Optional.ofNullable(toolCallId); }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        if (name != null) m.put("name", name);
        if (toolCallId != null) m.put("tool_call_id", toolCallId);
        return m;
    }

    @SuppressWarnings("unchecked")
    public static ChatMessage fromMap(Map<String, Object> m) {
        String role = String.valueOf(m.getOrDefault("role", "user"));
        Object c = m.get("content");
        String content = c == null ? "" : String.valueOf(c);
        String name = m.get("name") != null ? String.valueOf(m.get("name")) : null;
        String tcid = m.get("tool_call_id") != null ? String.valueOf(m.get("tool_call_id")) : null;
        return new ChatMessage(role, content, name, tcid);
    }
}
