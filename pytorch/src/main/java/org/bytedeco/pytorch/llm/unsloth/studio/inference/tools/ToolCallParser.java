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

import org.bytedeco.pytorch.llm.unsloth.studio.util.JsonMaps;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Parses tool calls from assistant text: OpenAI JSON, Hermes XML, Qwen/function call blocks.
 */
public final class ToolCallParser {

    public static final class ToolCall {
        public final String id;
        public final String name;
        public final Map<String, Object> arguments;
        public final boolean wellFormed;
        public final String raw;

        public ToolCall(String id, String name, Map<String, Object> arguments, boolean wellFormed, String raw) {
            this.id = id;
            this.name = name;
            this.arguments = arguments != null ? Map.copyOf(arguments) : Map.of();
            this.wellFormed = wellFormed;
            this.raw = raw;
        }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("type", "function");
            Map<String, Object> fn = new LinkedHashMap<>();
            fn.put("name", name);
            fn.put("arguments", JsonMaps.stringify(arguments));
            m.put("function", fn);
            m.put("well_formed", wellFormed);
            return m;
        }
    }

    private static final Pattern HERMES = Pattern.compile(
            "<tool_call>\\s*\\{.*?\\}\\s*</tool_call>", Pattern.DOTALL);
    private static final Pattern FUNC = Pattern.compile(
            "```(?:json|tool)?\\s*(\\{.*?\\})\\s*```", Pattern.DOTALL);
    private static final Pattern NAME_ARGS = Pattern.compile(
            "\"name\"\\s*:\\s*\"([^\"]+)\".*?\"arguments\"\\s*:\\s*(\\{.*?\\}|\\[.*?\\]|\"(?:\\\\.|[^\"])*\")",
            Pattern.DOTALL);

    public List<ToolCall> parse(String assistantText) {
        List<ToolCall> out = new ArrayList<>();
        if (assistantText == null || assistantText.isBlank()) return out;

        // Hermes-style
        Matcher hm = HERMES.matcher(assistantText);
        while (hm.find()) {
            out.add(parseObjectBlob(hm.group(), "call_" + out.size()));
        }
        if (!out.isEmpty()) return out;

        // fenced json
        Matcher fm = FUNC.matcher(assistantText);
        while (fm.find()) {
            out.add(parseObjectBlob(fm.group(1), "call_" + out.size()));
        }
        if (!out.isEmpty()) return out;

        // raw JSON object with name/arguments
        String trimmed = assistantText.trim();
        if (trimmed.startsWith("{") && trimmed.contains("\"name\"")) {
            out.add(parseObjectBlob(trimmed, "call_0"));
            return out;
        }

        // name/arguments regex salvage
        Matcher nm = NAME_ARGS.matcher(assistantText);
        while (nm.find()) {
            String name = nm.group(1);
            String argsRaw = nm.group(2);
            Map<String, Object> args = Map.of();
            boolean ok = true;
            try {
                if (argsRaw.startsWith("\"")) {
                    Object inner = JsonMaps.parse(argsRaw);
                    if (inner instanceof String) {
                        Object parsed = JsonMaps.parse((String) inner);
                        if (parsed instanceof Map<?, ?> mm) {
                            @SuppressWarnings("unchecked")
                            Map<String, Object> cast = (Map<String, Object>) parsed;
                            args = cast;
                        }
                    }
                } else {
                    Object parsed = JsonMaps.parse(argsRaw);
                    if (parsed instanceof Map<?, ?> mm) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> cast = (Map<String, Object>) parsed;
                        args = cast;
                    }
                }
            } catch (Exception e) {
                ok = false;
                args = Map.of("_raw", argsRaw);
            }
            out.add(new ToolCall("call_" + out.size(), name, args, ok, nm.group()));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private ToolCall parseObjectBlob(String raw, String id) {
        try {
            String json = raw;
            int start = raw.indexOf('{');
            int end = raw.lastIndexOf('}');
            if (start >= 0 && end > start) json = raw.substring(start, end + 1);
            Map<String, Object> obj = JsonMaps.parseObject(json);
            String name = null;
            Map<String, Object> args = Map.of();
            if (obj.containsKey("name")) name = String.valueOf(obj.get("name"));
            if (obj.get("arguments") instanceof Map<?, ?> am) {
                args = (Map<String, Object>) am;
            } else if (obj.get("arguments") instanceof String s) {
                Object p = JsonMaps.parse(s);
                if (p instanceof Map<?, ?> am) args = (Map<String, Object>) p;
            } else if (obj.get("parameters") instanceof Map<?, ?> am) {
                args = (Map<String, Object>) am;
            } else if (obj.get("function") instanceof Map<?, ?> fn) {
                Map<String, Object> f = (Map<String, Object>) fn;
                if (f.get("name") != null) name = String.valueOf(f.get("name"));
                if (f.get("arguments") instanceof Map<?, ?> am) args = (Map<String, Object>) am;
                else if (f.get("arguments") instanceof String s) {
                    Object p = JsonMaps.parse(s);
                    if (p instanceof Map<?, ?> am) args = (Map<String, Object>) p;
                }
            }
            if (name == null && obj.containsKey("tool")) name = String.valueOf(obj.get("tool"));
            boolean ok = name != null && !name.isBlank();
            return new ToolCall(id, name != null ? name : "unknown", args, ok, raw);
        } catch (Exception e) {
            return new ToolCall(id, "unknown", Map.of("_error", e.getMessage()), false, raw);
        }
    }
}
