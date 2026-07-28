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
package org.bytedeco.pytorch.llm.transformers.tokenization;

import org.bytedeco.pytorch.utils.json.Json;
import org.bytedeco.pytorch.llm.transformers.PretrainedConfig;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal chat templates for Instruct models (no full Jinja engine).
 *
 * <p>Supported flavors:
 * <ul>
 *   <li>{@code qwen} — ChatML {@code <|im_start|>role\n…<|im_end|>}</li>
 *   <li>{@code llama3} — Llama-3 header style</li>
 *   <li>{@code glm} — GLM-Edge {@code <|system|>}/{@code <|user|>}/{@code <|assistant|>}</li>
 *   <li>{@code raw} — concatenate content only</li>
 * </ul>
 */
public final class ChatTemplate {

    public enum Flavor { QWEN, LLAMA3, MISTRAL, GLM, RAW }

    private final Flavor flavor;

    public ChatTemplate(Flavor flavor) {
        this.flavor = flavor == null ? Flavor.RAW : flavor;
    }

    public static ChatTemplate qwen() { return new ChatTemplate(Flavor.QWEN); }
    public static ChatTemplate llama3() { return new ChatTemplate(Flavor.LLAMA3); }
    public static ChatTemplate mistral() { return new ChatTemplate(Flavor.MISTRAL); }
    public static ChatTemplate glm() { return new ChatTemplate(Flavor.GLM); }
    public static ChatTemplate raw() { return new ChatTemplate(Flavor.RAW); }

    public static ChatTemplate forModelType(PretrainedConfig.ModelType type) {
        if (type == null) return raw();
        return switch (type) {
            case QWEN -> qwen();
            case LLAMA -> llama3();
            case MISTRAL -> mistral();
            case GLM -> glm();
            default -> raw();
        };
    }

    /** Detect from tokenizer_config.json chat_template string or model type. */
    public static ChatTemplate detect(Path dir, PretrainedConfig cfg) {
        Path tc = dir.resolve("tokenizer_config.json");
        if (Files.isRegularFile(tc)) {
            try {
                String raw = Files.readString(tc, StandardCharsets.UTF_8);
                Map<String, Object> m = Json.decodeObject(raw);
                Object ct = m.get("chat_template");
                if (ct != null) {
                    String s = String.valueOf(ct).toLowerCase(Locale.ROOT);
                    if (s.contains("im_start") || s.contains("chatml")) return qwen();
                    if (s.contains("<|user|>") || s.contains("<|assistant|>") || s.contains("glm")) return glm();
                    if (s.contains("start_header_id") || s.contains("llama")) return llama3();
                    if (s.contains("[INST]") || s.contains("mistral")) return mistral();
                }
            } catch (IOException ignored) {}
        }
        return forModelType(cfg == null ? null : cfg.modelType());
    }

    /**
     * @param messages list of {@code {role, content}} maps
     * @param addGenerationPrompt append assistant header for generation
     */
    public String apply(List<Map<String, String>> messages, boolean addGenerationPrompt) {
        Objects.requireNonNull(messages, "messages");
        return switch (flavor) {
            case QWEN -> applyQwen(messages, addGenerationPrompt);
            case LLAMA3 -> applyLlama3(messages, addGenerationPrompt);
            case MISTRAL -> applyMistral(messages, addGenerationPrompt);
            case GLM -> applyGlm(messages, addGenerationPrompt);
            case RAW -> applyRaw(messages);
        };
    }

    public String apply(List<Map<String, String>> messages) {
        return apply(messages, true);
    }

    private static String roleOf(Map<String, String> m) {
        String r = m.get("role");
        return r == null ? "user" : r;
    }

    private static String contentOf(Map<String, String> m) {
        String c = m.get("content");
        return c == null ? "" : c;
    }

    private static String applyQwen(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> msg : messages) {
            sb.append("<|im_start|>").append(roleOf(msg)).append('\n')
              .append(contentOf(msg)).append("<|im_end|>\n");
        }
        if (addGen) {
            sb.append("<|im_start|>assistant\n");
        }
        return sb.toString();
    }

    private static String applyLlama3(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        sb.append("<|begin_of_text|>");
        for (Map<String, String> msg : messages) {
            sb.append("<|start_header_id|>").append(roleOf(msg)).append("<|end_header_id|>\n\n")
              .append(contentOf(msg)).append("<|eot_id|>");
        }
        if (addGen) {
            sb.append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        }
        return sb.toString();
    }

    private static String applyMistral(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        String system = null;
        for (Map<String, String> msg : messages) {
            if ("system".equals(roleOf(msg))) {
                system = contentOf(msg);
                break;
            }
        }
        for (Map<String, String> msg : messages) {
            String role = roleOf(msg);
            if ("system".equals(role)) continue;
            if ("user".equals(role)) {
                sb.append("[INST] ");
                if (system != null) {
                    sb.append(system).append("\n\n");
                    system = null; // only once
                }
                sb.append(contentOf(msg)).append(" [/INST]");
            } else if ("assistant".equals(role)) {
                sb.append(' ').append(contentOf(msg)).append("</s>");
            }
        }
        // addGen: mistral expects model to continue after [/INST]
        return sb.toString();
    }

    /**
     * GLM-Edge / ChatGLM chat format from tokenizer chat_template:
     * {@code <|system|>\n…\n<|user|>\n…\n<|assistant|>\n…}
     */
    private static String applyGlm(List<Map<String, String>> messages, boolean addGen) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> msg : messages) {
            String role = roleOf(msg);
            String tag = switch (role) {
                case "system" -> "<|system|>";
                case "assistant" -> "<|assistant|>";
                case "observation" -> "<|observation|>";
                default -> "<|user|>";
            };
            sb.append(tag).append('\n').append(contentOf(msg));
        }
        if (addGen) {
            sb.append("<|assistant|>\n");
        }
        return sb.toString();
    }

    private static String applyRaw(List<Map<String, String>> messages) {
        StringBuilder sb = new StringBuilder();
        for (Map<String, String> msg : messages) {
            if (sb.length() > 0) sb.append('\n');
            sb.append(contentOf(msg));
        }
        return sb.toString();
    }

    public Flavor flavor() {
        return flavor;
    }
}
