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

package org.bytedeco.pytorch.llm.unsloth.studio.inference;

import org.bytedeco.pytorch.llm.unsloth.studio.model.ChatMessage;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Chat template rendering for common families (Llama3 / Qwen / Gemma / Mistral / Phi / ChatML).
 * Prefer factory TemplateRegistry when present; otherwise use built-in compact templates.
 */
public final class ChatTemplateService {

    private final Map<String, String> overrides = new ConcurrentHashMap<>();

    public void setOverride(String modelId, String jinjaOrText) {
        if (modelId != null && jinjaOrText != null) {
            overrides.put(modelId, jinjaOrText);
        }
    }

    public String render(String modelId, List<ChatMessage> messages, String override) {
        if (override != null && !override.isBlank()) {
            return applySimple(override, messages);
        }
        if (modelId != null && overrides.containsKey(modelId)) {
            return applySimple(overrides.get(modelId), messages);
        }
        // Try factory TemplateRegistry reflectively
        String viaFactory = tryFactory(modelId, messages);
        if (viaFactory != null) return viaFactory;
        return renderBuiltin(guessFamily(modelId), messages);
    }

    public String renderBuiltin(String family, List<ChatMessage> messages) {
        String f = family == null ? "chatml" : family.toLowerCase(Locale.ROOT);
        return switch (f) {
            case "llama", "llama3" -> renderLlama3(messages);
            case "qwen", "qwen2", "qwen3" -> renderQwen(messages);
            case "gemma", "gemma2", "gemma3" -> renderGemma(messages);
            case "mistral", "mixtral" -> renderMistral(messages);
            case "phi", "phi3", "phi4" -> renderPhi(messages);
            default -> renderChatMl(messages);
        };
    }

    private String renderLlama3(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|begin_of_text|>");
        for (ChatMessage m : messages) {
            sb.append("<|start_header_id|>").append(m.role()).append("<|end_header_id|>\n\n");
            sb.append(m.content()).append("<|eot_id|>");
        }
        sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return sb.toString();
    }

    private String renderQwen(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        for (ChatMessage m : messages) {
            sb.append("<|im_start|>").append(m.role()).append('\n');
            sb.append(m.content()).append("<|im_end|>\n");
        }
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    private String renderGemma(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        for (ChatMessage m : messages) {
            String role = "assistant".equals(m.role()) ? "model" : m.role();
            if ("system".equals(role)) {
                sb.append("<start_of_turn>user\n[System]\n").append(m.content()).append("<end_of_turn>\n");
            } else {
                sb.append("<start_of_turn>").append(role).append('\n');
                sb.append(m.content()).append("<end_of_turn>\n");
            }
        }
        sb.append("<start_of_turn>model\n");
        return sb.toString();
    }

    private String renderMistral(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        String system = null;
        for (ChatMessage m : messages) {
            if ("system".equals(m.role())) system = m.content();
        }
        boolean firstUser = true;
        for (ChatMessage m : messages) {
            if ("system".equals(m.role())) continue;
            if ("user".equals(m.role())) {
                sb.append("[INST] ");
                if (firstUser && system != null) {
                    sb.append(system).append("\n\n");
                    firstUser = false;
                }
                sb.append(m.content()).append(" [/INST]");
            } else if ("assistant".equals(m.role())) {
                sb.append(m.content()).append("</s>");
            }
        }
        return sb.toString();
    }

    private String renderPhi(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder();
        for (ChatMessage m : messages) {
            if ("system".equals(m.role())) {
                sb.append("<|system|>\n").append(m.content()).append("<|end|>\n");
            } else if ("user".equals(m.role())) {
                sb.append("<|user|>\n").append(m.content()).append("<|end|>\n");
            } else if ("assistant".equals(m.role())) {
                sb.append("<|assistant|>\n").append(m.content()).append("<|end|>\n");
            }
        }
        sb.append("<|assistant|>\n");
        return sb.toString();
    }

    private String renderChatMl(List<ChatMessage> messages) {
        return renderQwen(messages);
    }

    /** Very small mustache-like: replace {{role}}/{{content}} loops are not full jinja. */
    private String applySimple(String template, List<ChatMessage> messages) {
        if (!template.contains("{%") && !template.contains("{{")) {
            // plain prefix + concatenated messages
            StringBuilder sb = new StringBuilder(template);
            if (!template.endsWith("\n")) sb.append('\n');
            for (ChatMessage m : messages) {
                sb.append(m.role()).append(": ").append(m.content()).append('\n');
            }
            sb.append("assistant:");
            return sb.toString();
        }
        // Fallback: ignore jinja structure, use chatml (full jinja needs host engine)
        return renderChatMl(messages);
    }

    private String tryFactory(String modelId, List<ChatMessage> messages) {
        try {
            Class<?> reg = Class.forName("org.bytedeco.pytorch.llm.llamafactory.data.template.TemplateRegistry");
            Object tmpl = null;
            try {
                tmpl = reg.getMethod("get", String.class).invoke(null, modelId != null ? modelId : "default");
            } catch (NoSuchMethodException e) {
                Object inst = reg.getMethod("getInstance").invoke(null);
                tmpl = inst.getClass().getMethod("get", String.class).invoke(inst, modelId);
            }
            if (tmpl != null) {
                try {
                    return (String) tmpl.getClass().getMethod("render", List.class).invoke(tmpl, messages);
                } catch (NoSuchMethodException e2) {
                    return null;
                }
            }
        } catch (Throwable ignored) {}
        return null;
    }

    static String guessFamily(String modelId) {
        if (modelId == null) return "chatml";
        String s = modelId.toLowerCase(Locale.ROOT);
        if (s.contains("llama")) return "llama";
        if (s.contains("qwen")) return "qwen";
        if (s.contains("gemma")) return "gemma";
        if (s.contains("mistral") || s.contains("mixtral")) return "mistral";
        if (s.contains("phi")) return "phi";
        return "chatml";
    }
}
