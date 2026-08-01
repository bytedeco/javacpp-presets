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
 * Training stage (mirrors LLaMA-Factory {@code stage} hyper-parameter).
 *
 * <ul>
 *   <li>{@link #PT} — (continuous) pre-training / causal LM on plain text</li>
 *   <li>{@link #SFT} — supervised fine-tuning (instruction / multi-turn)</li>
 *   <li>{@link #RM} — reward model (pairwise ranking)</li>
 *   <li>{@link #PPO} — proximal policy optimization (RLHF)</li>
 *   <li>{@link #DPO} — direct preference optimization</li>
 *   <li>{@link #KTO} — Kahneman-Tversky optimization</li>
 *   <li>{@link #ORPO} — odds-ratio preference optimization</li>
 *   <li>{@link #GRPO} — group-relative policy optimization</li>
 * </ul>
 */
public enum Stage {
    PT,
    SFT,
    RM,
    PPO,
    DPO,
    KTO,
    ORPO,
    GRPO;

    /** Parse case-insensitive name; aliases: {@code pretrain→PT}, {@code sft→SFT}, … */
    public static Stage parse(String raw) {
        if (raw == null || raw.isBlank()) {
            return SFT;
        }
        String s = raw.trim().toLowerCase(Locale.ROOT).replace('-', '_');
        return switch (s) {
            case "pt", "pretrain", "pre_train", "pre-training", "cpt" -> PT;
            case "sft", "supervised", "finetune", "fine_tune" -> SFT;
            case "rm", "reward", "reward_model" -> RM;
            case "ppo", "rlhf" -> PPO;
            case "dpo" -> DPO;
            case "kto" -> KTO;
            case "orpo" -> ORPO;
            case "grpo" -> GRPO;
            default -> {
                try {
                    yield Stage.valueOf(s.toUpperCase(Locale.ROOT));
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(
                            "Unknown stage '" + raw + "'; expected one of "
                                    + java.util.Arrays.toString(values()), e);
                }
            }
        };
    }

    /** Whether this stage consumes pairwise (chosen/rejected) data. */
    public boolean pairwise() {
        return this == RM || this == DPO || this == ORPO;
    }

    /** Whether this stage needs a reference / frozen policy. */
    public boolean needsReference() {
        return this == DPO || this == KTO || this == PPO;
    }

    /** Whether this stage needs a separate reward model. */
    public boolean needsRewardModel() {
        return this == PPO;
    }

    /** LLaMA-Factory wire name (lowercase). */
    public String wireName() {
        return name().toLowerCase(Locale.ROOT);
    }
}
