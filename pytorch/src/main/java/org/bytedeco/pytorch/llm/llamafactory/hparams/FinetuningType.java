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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.Locale;

/**
 * Fine-tuning mode (mirrors LLaMA-Factory {@code finetuning_type}).
 *
 * <ul>
 *   <li>{@link #FULL} — full parameter fine-tuning (no adapters)</li>
 *   <li>{@link #FREEZE} — freeze base model, tune only selected layers / heads</li>
 *   <li>{@link #LORA} — LoRA (rank &amp; alpha configurable)</li>
 *   <li>{@link #QLORA} — QLoRA (4-bit base + LoRA)</li>
 *   <li>{@link #IA3} — IA3 (scaling vectors on attention)</li>
 *   <li>{@link #PROMPT} — prefix / prompt tuning</li>
 *   <li>{@link #PREFIX} — prefix tuning</li>
 * </ul>
 */
public enum FinetuningType {
    FULL,
    FREEZE,
    LORA,
    QLORA,
    IA3,
    PROMPT,
    PREFIX;

    /** Parse name; aliases {@code freeze→FREEZE}, {@code lora→LORA}, … */
    public static FinetuningType parse(String raw) {
        if (raw == null || raw.isBlank()) {
            return LORA;
        }
        String s = raw.trim().toLowerCase(Locale.ROOT).replace('-', '_').replace(" ", "");
        return switch (s) {
            case "full", "full_ft" -> FULL;
            case "freeze", "frozen" -> FREEZE;
            case "lora" -> LORA;
            case "qlora", "q_lora" -> QLORA;
            case "ia3" -> IA3;
            case "prompt", "prompt_tuning" -> PROMPT;
            case "prefix", "prefix_tuning" -> PREFIX;
            default -> {
                try {
                    yield valueOf(s.toUpperCase(Locale.ROOT));
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(
                            "Unknown finetuning type '" + raw + "'; expected one of "
                                    + java.util.Arrays.toString(values()), e);
                }
            }
        };
    }

    /** Whether this mode requires a PEFT wrapper. */
    public boolean needsPeft() {
        return this != FULL && this != FREEZE;
    }

    /** LLaMA-Factory wire name. */
    public String wireName() {
        return name().toLowerCase(Locale.ROOT);
    }
}
