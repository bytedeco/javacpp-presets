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
package org.bytedeco.pytorch.llm.llamafactory.data.converter;

import org.bytedeco.pytorch.llm.llamafactory.data.template.Formatter;
import org.bytedeco.pytorch.llm.llamafactory.data.template.Template;
import org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Preference (chosen / rejected) converter for DPO / ORPO / RM.
 *
 * <p>Accepts:
 * <ul>
 *   <li>{@code chosen} / {@code rejected} as strings or message lists</li>
 *   <li>{@code conversations} + {@code chosen}/{@code rejected} assistant tails</li>
 *   <li>UltraFeedback-style nested maps with {@code content} / {@code value}</li>
 * </ul>
 */
public final class PreferenceConverter {

    private final Template template;

    public PreferenceConverter(Template template) {
        this.template = Objects.requireNonNull(template, "template");
    }

    public PreferenceConverter(String templateName) {
        this(TemplateRegistry.getOrDefault(templateName));
    }

    public static PreferenceConverter defaults() {
        return new PreferenceConverter("default");
    }

    public Map<String, Object> convert(Map<String, Object> raw) {
        Objects.requireNonNull(raw, "raw");
        String prompt = buildPrompt(raw);
        String chosen = asText(raw.get("chosen"));
        String rejected = asText(raw.get("rejected"));
        if (chosen.isEmpty()) {
            chosen = asText(raw.get("response_j"));
        }
        if (rejected.isEmpty()) {
            rejected = asText(raw.get("response_k"));
        }

        String chosenFull = prompt + chosen;
        String rejectedFull = prompt + rejected;

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("prompt", prompt);
        out.put("chosen", chosen);
        out.put("rejected", rejected);
        out.put("chosen_text", chosenFull);
        out.put("rejected_text", rejectedFull);
        // For tokenizers that operate on full strings later
        out.put("text", chosenFull);
        return out;
    }

    public List<Map<String, Object>> convertAll(List<Map<String, Object>> rows) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) out.add(convert(r));
        return out;
    }

    private String buildPrompt(Map<String, Object> raw) {
        Object p = raw.get("prompt");
        if (p instanceof String s && !s.isEmpty()) {
            // Ensure assistant prefix is open for completion-style pairs
            if (!s.contains(template.name())) {
                // already a rendered prompt — use as-is if it looks complete
            }
            return s.endsWith("\n") || s.contains("Assistant") || s.contains("assistant")
                    ? s
                    : s;
        }
        // instruction + input style
        String instruction = Formatter.str(raw.get("instruction"),
                Formatter.str(raw.get("question"), ""));
        String input = Formatter.str(raw.get("input"), "");
        List<Template.Message> msgs = new ArrayList<>();
        String system = Formatter.str(raw.get("system"), null);
        if (system != null && !system.isEmpty()) {
            msgs.add(Template.Message.system(system));
        }
        // sharegpt conversations without final assistant
        Object conv = raw.get("conversations");
        if (conv instanceof List<?> list && !list.isEmpty()) {
            msgs.addAll(Formatter.messagesFromRow(Map.of("conversations", list)));
            // drop trailing assistant if present — prompt only
            if (!msgs.isEmpty()) {
                String role = msgs.get(msgs.size() - 1).role().toLowerCase();
                if ("assistant".equals(role) || "gpt".equals(role)) {
                    msgs = new ArrayList<>(msgs.subList(0, msgs.size() - 1));
                }
            }
            return template.encodePrompt(msgs, system);
        }
        String user = input.isEmpty() ? instruction
                : (instruction.isEmpty() ? input : instruction + "\n" + input);
        if (!user.isEmpty()) {
            msgs.add(Template.Message.user(user));
        }
        return template.encodePrompt(msgs, system);
    }

    @SuppressWarnings("unchecked")
    private static String asText(Object o) {
        if (o == null) return "";
        if (o instanceof String s) return s;
        if (o instanceof Map<?, ?> m) {
            Object c = m.get("content");
            if (c == null) c = m.get("value");
            if (c == null) c = m.get("text");
            return c == null ? "" : String.valueOf(c);
        }
        if (o instanceof List<?> list) {
            StringBuilder sb = new StringBuilder();
            for (Object item : list) {
                if (item instanceof Map<?, ?> m) {
                    String role = Formatter.str(m.get("role"), Formatter.str(m.get("from"), "assistant"));
                    String content = Formatter.str(m.get("content"), Formatter.str(m.get("value"), ""));
                    if ("assistant".equalsIgnoreCase(Formatter.normalizeRole(role))
                            || "gpt".equalsIgnoreCase(role)) {
                        if (sb.length() > 0) sb.append('\n');
                        sb.append(content);
                    }
                } else if (item != null) {
                    if (sb.length() > 0) sb.append('\n');
                    sb.append(item);
                }
            }
            return sb.toString();
        }
        return String.valueOf(o);
    }
}
