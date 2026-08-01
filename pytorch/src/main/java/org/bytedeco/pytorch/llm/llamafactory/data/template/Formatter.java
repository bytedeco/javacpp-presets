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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Helpers to normalize raw dataset rows into {@link Template.Message} lists
 * and apply a {@link Template}.
 */
public final class Formatter {

    private Formatter() {}

    /**
     * Build messages from a generic map that may already contain a
     * {@code messages} list or alpaca-style instruction/input/output fields.
     */
    @SuppressWarnings("unchecked")
    public static List<Template.Message> messagesFromRow(Map<String, Object> row) {
        Objects.requireNonNull(row, "row");
        Object msgs = row.get("messages");
        if (msgs instanceof List<?> list && !list.isEmpty()) {
            List<Template.Message> out = new ArrayList<>(list.size());
            for (Object o : list) {
                if (o instanceof Template.Message m) {
                    out.add(m);
                } else if (o instanceof Map<?, ?> m) {
                    String role = str(m.get("role"), str(m.get("from"), "user"));
                    String content = str(m.get("content"), str(m.get("value"), ""));
                    out.add(new Template.Message(role, content));
                } else if (o != null) {
                    out.add(Template.Message.user(String.valueOf(o)));
                }
            }
            return out;
        }
        // conversations sharegpt
        Object conv = row.get("conversations");
        if (conv instanceof List<?> list && !list.isEmpty()) {
            List<Template.Message> out = new ArrayList<>(list.size());
            for (Object o : list) {
                if (o instanceof Map<?, ?> m) {
                    String role = str(m.get("from"), str(m.get("role"), "human"));
                    String content = str(m.get("value"), str(m.get("content"), ""));
                    out.add(new Template.Message(normalizeRole(role), content));
                }
            }
            return out;
        }
        // alpaca
        List<Template.Message> out = new ArrayList<>(3);
        String system = str(row.get("system"), null);
        if (system != null && !system.isEmpty()) {
            out.add(Template.Message.system(system));
        }
        String instruction = str(row.get("instruction"), str(row.get("prompt"), null));
        String input = str(row.get("input"), str(row.get("query"), ""));
        String output = str(row.get("output"), str(row.get("response"),
                str(row.get("completion"), "")));
        if (instruction != null) {
            String user = input == null || input.isEmpty()
                    ? instruction
                    : instruction + "\n" + input;
            out.add(Template.Message.user(user));
        } else if (input != null && !input.isEmpty()) {
            out.add(Template.Message.user(input));
        }
        if (output != null && !output.isEmpty()) {
            out.add(Template.Message.assistant(output));
        }
        // plain text pretrain
        if (out.isEmpty()) {
            String text = str(row.get("text"), str(row.get("content"), ""));
            if (!text.isEmpty()) {
                out.add(Template.Message.assistant(text));
            }
        }
        return out;
    }

    public static String normalizeRole(String role) {
        if (role == null) return "user";
        String r = role.toLowerCase(Locale.ROOT);
        return switch (r) {
            case "human", "user", "prompter" -> "user";
            case "gpt", "assistant", "bot", "model" -> "assistant";
            case "system" -> "system";
            case "observation", "tool", "function" -> "tool";
            default -> r;
        };
    }

    /** Apply template to a row → full training string. */
    public static String formatRow(Template template, Map<String, Object> row) {
        List<Template.Message> msgs = messagesFromRow(row);
        String system = str(row.get("system"), null);
        return template.encodeOneline(msgs, system);
    }

    /** Prompt / response pair for SFT label masking. */
    public static String[] formatSupervised(Template template, Map<String, Object> row) {
        List<Template.Message> msgs = messagesFromRow(row);
        String system = str(row.get("system"), null);
        return template.encodeSupervised(msgs, system);
    }

    public static String str(Object o, String def) {
        if (o == null) return def;
        String s = String.valueOf(o);
        return s;
    }

    /** Shallow copy row with extra keys. */
    public static Map<String, Object> with(Map<String, Object> row, String k, Object v) {
        Map<String, Object> m = new LinkedHashMap<>(row);
        m.put(k, v);
        return m;
    }
}
