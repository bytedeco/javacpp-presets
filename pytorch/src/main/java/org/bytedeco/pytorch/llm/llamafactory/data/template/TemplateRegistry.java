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

import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of named chat templates (alpaca / sharegpt / llama3 / qwen / chatml / glm4 / …).
 *
 * <p>Mirrors LLaMA-Factory {@code data/template.py} registrations. Hosts may
 * {@link #register(String, Template)} extra templates at runtime.
 */
public final class TemplateRegistry {

    private static final Map<String, Template> REGISTRY = new ConcurrentHashMap<>();

    static {
        register("empty", Template.builder("empty")
                .system("", "")
                .user("", "")
                .assistant("", "")
                .build());

        register("default", Template.builder("default")
                .system("### System:\n", "\n\n")
                .user("### Human:\n", "\n\n")
                .assistant("### Assistant:\n", "\n\n")
                .defaultSystem("")
                .build());

        register("alpaca", Template.builder("alpaca")
                .system("### Instruction:\n", "\n\n")
                .user("", "\n\n")
                .assistant("### Response:\n", "\n\n")
                .defaultSystem(
                        "Below is an instruction that describes a task. "
                                + "Write a response that appropriately completes the request.")
                .build());

        // ShareGPT-style often uses the same default role tags after conversion
        register("sharegpt", Template.builder("sharegpt")
                .system("### System:\n", "\n\n")
                .user("### Human:\n", "\n\n")
                .assistant("### Assistant:\n", "\n\n")
                .build());

        register("llama2", Template.builder("llama2")
                .system("<<SYS>>\n", "\n<</SYS>>\n\n")
                .user("[INST] ", " [/INST]")
                .assistant(" ", " </s>")
                .defaultSystem("You are a helpful assistant.")
                .stopWord("</s>")
                .build());

        register("llama3", Template.builder("llama3")
                .system("<|start_header_id|>system<|end_header_id|>\n\n", "<|eot_id|>")
                .user("<|start_header_id|>user<|end_header_id|>\n\n", "<|eot_id|>")
                .assistant("<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>")
                .defaultSystem("You are a helpful assistant.")
                .stopWord("<|eot_id|>")
                .stopWord("<|end_of_text|>")
                .efficientEos(true)
                .build());

        register("qwen", Template.builder("qwen")
                .system("<|im_start|>system\n", "<|im_end|>\n")
                .user("<|im_start|>user\n", "<|im_end|>\n")
                .assistant("<|im_start|>assistant\n", "<|im_end|>\n")
                .defaultSystem("You are a helpful assistant.")
                .stopWord("<|im_end|>")
                .stopWord("<|endoftext|>")
                .efficientEos(true)
                .build());

        register("qwen2", get("qwen"));
        register("qwen3", get("qwen"));

        register("chatml", Template.builder("chatml")
                .system("<|im_start|>system\n", "<|im_end|>\n")
                .user("<|im_start|>user\n", "<|im_end|>\n")
                .assistant("<|im_start|>assistant\n", "<|im_end|>\n")
                .stopWord("<|im_end|>")
                .efficientEos(true)
                .build());

        register("glm4", Template.builder("glm4")
                .system("<|system|>\n", "")
                .user("<|user|>\n", "")
                .assistant("<|assistant|>\n", "")
                .defaultSystem("你是一个乐于助人的助手。")
                .stopWord("<|user|>")
                .stopWord("<|observation|>")
                .efficientEos(true)
                .build());

        register("gemma", Template.builder("gemma")
                .system("<start_of_turn>system\n", "<end_of_turn>\n")
                .user("<start_of_turn>user\n", "<end_of_turn>\n")
                .assistant("<start_of_turn>model\n", "<end_of_turn>\n")
                .stopWord("<end_of_turn>")
                .efficientEos(true)
                .build());

        register("mistral", Template.builder("mistral")
                .system("", "")
                .user("[INST] ", " [/INST]")
                .assistant(" ", "</s>")
                .stopWord("</s>")
                .build());

        register("phi", Template.builder("phi")
                .system("<|system|>\n", "<|end|>\n")
                .user("<|user|>\n", "<|end|>\n")
                .assistant("<|assistant|>\n", "<|end|>\n")
                .stopWord("<|end|>")
                .efficientEos(true)
                .build());

        register("deepseek", Template.builder("deepseek")
                .system("", "")
                .user("User: ", "\n\n")
                .assistant("Assistant: ", "<｜end▁of▁sentence｜>")
                .stopWord("<｜end▁of▁sentence｜>")
                .build());

        register("vicuna", Template.builder("vicuna")
                .system("", "\n")
                .user("USER: ", "\n")
                .assistant("ASSISTANT: ", "</s>\n")
                .defaultSystem("A chat between a curious user and an artificial intelligence assistant. "
                        + "The assistant gives helpful, detailed, and polite answers to the user's questions.")
                .stopWord("</s>")
                .build());

        // Multimodal-friendly aliases (same text template; collator adds pixels)
        register("llava", get("vicuna"));
        register("qwen2_vl", get("qwen"));
        register("qwen3_vl", get("qwen"));
    }

    private TemplateRegistry() {}

    public static void register(String name, Template template) {
        if (name == null || template == null) {
            throw new IllegalArgumentException("name/template required");
        }
        REGISTRY.put(name.toLowerCase(Locale.ROOT), template);
    }

    public static Template get(String name) {
        if (name == null || name.isBlank()) {
            return REGISTRY.get("default");
        }
        Template t = REGISTRY.get(name.toLowerCase(Locale.ROOT).trim());
        if (t == null) {
            throw new IllegalArgumentException(
                    "Unknown template '" + name + "'; known=" + REGISTRY.keySet());
        }
        return t;
    }

    /** Resolve with fallback to default (never throws). */
    public static Template getOrDefault(String name) {
        try {
            return get(name);
        } catch (IllegalArgumentException e) {
            return REGISTRY.get("default");
        }
    }

    public static Set<String> names() {
        return Collections.unmodifiableSet(REGISTRY.keySet());
    }

    public static boolean contains(String name) {
        return name != null && REGISTRY.containsKey(name.toLowerCase(Locale.ROOT).trim());
    }
}
