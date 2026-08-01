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
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Tool-calling helpers (function call parse / format) for multi-turn agent data.
 *
 * <p>Supports a minimal OpenAI-style and GLM/Qwen parallel function-call subset
 * used in LLaMA-Factory tool datasets — pure string level, no network.
 */
public final class ToolUtils {

    private static final Pattern OPENAI_CALL = Pattern.compile(
            "\\{\\s*\"name\"\\s*:\\s*\"([^\"]+)\"\\s*,\\s*\"arguments\"\\s*:\\s*(\\{.*?\\}|\"\\{.*?\\}\")\\s*\\}",
            Pattern.DOTALL);
    private static final Pattern FN_TAG = Pattern.compile(
            "<tool_call>\\s*([\\s\\S]*?)\\s*</tool_call>", Pattern.CASE_INSENSITIVE);

    private ToolUtils() {}

    /** Format a tool result message body. */
    public static String formatToolResult(String name, String content) {
        String n = name == null ? "tool" : name;
        String c = content == null ? "" : content;
        return "{\"name\":\"" + escape(n) + "\",\"content\":\"" + escape(c) + "\"}";
    }

    /** Format an assistant tool call body (OpenAI parallel-call JSON list). */
    public static String formatToolCall(String name, String argumentsJson) {
        String n = name == null ? "fn" : name;
        String a = argumentsJson == null || argumentsJson.isBlank() ? "{}" : argumentsJson.trim();
        return "{\"name\":\"" + escape(n) + "\",\"arguments\":" + a + "}";
    }

    /**
     * Parse tool calls from an assistant message. Returns list of
     * {@code {name, arguments}} maps (arguments as raw JSON string).
     */
    public static List<Map<String, String>> parseToolCalls(String text) {
        List<Map<String, String>> out = new ArrayList<>();
        if (text == null || text.isBlank()) {
            return out;
        }
        Matcher tag = FN_TAG.matcher(text);
        String search = text;
        if (tag.find()) {
            search = tag.group(1);
        }
        Matcher m = OPENAI_CALL.matcher(search);
        while (m.find()) {
            Map<String, String> call = new LinkedHashMap<>();
            call.put("name", m.group(1));
            String args = m.group(2);
            if (args != null && args.startsWith("\"") && args.endsWith("\"")) {
                args = args.substring(1, args.length() - 1).replace("\\\"", "\"");
            }
            call.put("arguments", args);
            out.add(call);
        }
        // Fallback: whole text is a single bare call
        if (out.isEmpty() && text.contains("\"name\"") && text.contains("\"arguments\"")) {
            Matcher once = OPENAI_CALL.matcher(text);
            if (once.find()) {
                Map<String, String> call = new LinkedHashMap<>();
                call.put("name", once.group(1));
                call.put("arguments", once.group(2));
                out.add(call);
            }
        }
        return out;
    }

    /** True if content looks like a tool call. */
    public static boolean isToolCall(String text) {
        return text != null && (text.contains("<tool_call>")
                || (text.contains("\"name\"") && text.contains("\"arguments\"")));
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n");
    }
}
