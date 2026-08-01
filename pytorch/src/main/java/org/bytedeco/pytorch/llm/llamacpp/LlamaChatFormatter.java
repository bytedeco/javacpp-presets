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

package org.bytedeco.pytorch.llm.llamacpp;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Chat template rendering for GGUF / llama.cpp families.
 * Mirrors common llama.cpp built-in templates (chatml, llama3, gemma, phi, mistral).
 */
public final class LlamaChatFormatter {

    private final LlamaArchitecture architecture;
    private final String overrideTemplate;

    public LlamaChatFormatter(LlamaArchitecture architecture) {
        this(architecture, null);
    }

    public LlamaChatFormatter(LlamaArchitecture architecture, String overrideTemplate) {
        this.architecture = architecture != null ? architecture : LlamaArchitecture.UNKNOWN;
        this.overrideTemplate = overrideTemplate;
    }

    public String format(List<Map<String, String>> messages) {
        Objects.requireNonNull(messages, "messages");
        if (overrideTemplate != null && !overrideTemplate.isBlank()) {
            return applyPlain(overrideTemplate, messages);
        }
        return switch (architecture) {
            case LLAMA, MISTRAL, MIXTRAL -> renderLlama3(messages);
            case QWEN2, QWEN3 -> renderChatMl(messages);
            case GEMMA, GEMMA2, GEMMA3 -> renderGemma(messages);
            case PHI3, PHI3V -> renderPhi(messages);
            case GPT2, GPTNEOX -> renderPlain(messages);
            default -> renderChatMl(messages);
        };
    }

    public static String role(Map<String, String> m) {
        if (m == null) return "user";
        String r = m.get("role");
        return r != null ? r : "user";
    }

    public static String content(Map<String, String> m) {
        if (m == null) return "";
        String c = m.get("content");
        return c != null ? c : "";
    }

    private String renderLlama3(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|begin_of_text|>");
        for (Map<String, String> m : messages) {
            sb.append("<|start_header_id|>").append(role(m)).append("<|end_header_id|>\n\n");
            sb.append(content(m)).append("<|eot_id|>");
        }
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.toString();
    }

    private String renderChatMl(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            sb.append("<|im_start|>").append(role(m)).append('\n');
            sb.append(content(m)).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    private String renderGemma(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            String r = role(m);
            if ("assistant".equals(r)) r = "model";
            if ("system".equals(r)) {
                sb.append("<start_of_turn>user\n[System]\n").append(content(m)).append("<end_of_turn>\n");
            } else {
                sb.append("<start_of_turn>").append(r).append('\n');
                sb.append(content(m)).append("<end_of_turn>\n");
            }
        }
        sb.append("<start_of_turn>model\n");
        return sb.toString();
    }

    private String renderPhi(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            String r = role(m);
            if ("system".equals(r)) {
                sb.append("<|system|>\n").append(content(m)).append("<|end|>\n");
            } else if ("user".equals(r)) {
                sb.append("<|user|>\n").append(content(m)).append("<|end|>\n");
            } else if ("assistant".equals(r)) {
                sb.append("<|assistant|>\n").append(content(m)).append("<|end|>\n");
            }
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    private String renderPlain(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> m : messages) {
            sb.append(role(m)).append(": ").append(content(m)).append('\n');
        }
        sb.append("assistant:");
        return sb.toString();
    }

    private String applyPlain(String template, List<Map<String, String>> messages) {
        // Not full Jinja — if template has no placeholders, append transcript.
        if (!template.contains("{{") && !template.contains("{%")) {
            StringBuilder sb = new StringBuilder(template);
            if (!template.endsWith("\n")) sb.append('\n');
            for (Map<String, String> m : messages) {
                sb.append(role(m)).append(": ").append(content(m)).append('\n');
            }
            sb.append("assistant:");
            return sb.toString();
        }
        return renderChatMl(messages);
    }

    public static LlamaArchitecture guessFromModelId(String modelId) {
        if (modelId == null) return LlamaArchitecture.UNKNOWN;
        String s = modelId.toLowerCase(Locale.ROOT);
        return LlamaArchitecture.fromMetadata(s);
    }
}
