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

/**
 * Training type labels aligned with upstream Studio UI
 * ({@code LoRA/QLoRA}, {@code Full Finetuning}, {@code Continued Pretraining}).
 */
public enum TrainingType {
    LORA_QLORA("LoRA/QLoRA"),
    FULL_FINETUNING("Full Finetuning"),
    CONTINUED_PRETRAINING("Continued Pretraining"),
    /** RL facade (GRPO / DPO / PPO) — Studio extension over upstream SFT-first UI. */
    REINFORCEMENT_LEARNING("Reinforcement Learning");

    private final String label;

    TrainingType(String label) {
        this.label = label;
    }

    public String label() {
        return label;
    }

    public boolean isLoraFamily() {
        return this == LORA_QLORA;
    }

    public boolean isFull() {
        return this == FULL_FINETUNING || this == CONTINUED_PRETRAINING;
    }

    public static TrainingType fromLabel(String raw) {
        if (raw == null || raw.isBlank()) {
            throw new IllegalArgumentException("training_type is required");
        }
        String t = raw.trim();
        for (TrainingType v : values()) {
            if (v.label.equalsIgnoreCase(t) || v.name().equalsIgnoreCase(t)
                    || v.name().replace('_', '/').equalsIgnoreCase(t)
                    || v.name().replace('_', ' ').equalsIgnoreCase(t)) {
                return v;
            }
        }
        // common aliases
        String lower = t.toLowerCase();
        if (lower.contains("lora") || lower.contains("qlora")) return LORA_QLORA;
        if (lower.contains("full")) return FULL_FINETUNING;
        if (lower.contains("pretrain")) return CONTINUED_PRETRAINING;
        if (lower.contains("rl") || lower.contains("grpo") || lower.contains("dpo") || lower.contains("ppo")) {
            return REINFORCEMENT_LEARNING;
        }
        throw new IllegalArgumentException("Unknown training_type: " + raw);
    }
}
