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
package org.bytedeco.pytorch.llm.ktransformers.config;

/**
 * Model families with published KTransformers / kt-kernel support trajectories.
 *
 * <p>Names mirror upstream Day0 / tutorial coverage (DeepSeek, Kimi, MiniMax, GLM,
 * Qwen-MoE, Mixtral, LLaMA 4, SmallThinker). {@link #GENERIC} is the safe default
 * when only a manual layer map is supplied.
 */
public enum KtModelFamily {
    GENERIC,
    DEEPSEEK_V2,
    DEEPSEEK_V3,
    DEEPSEEK_R1,
    DEEPSEEK_V4_FLASH,
    KIMI_K2,
    KIMI_K2_THINKING,
    KIMI_K2_5,
    MINIMAX_M2,
    MINIMAX_M2_1,
    MINIMAX_M2_5,
    MINIMAX_M3,
    GLM4_MOE,
    GLM5,
    GLM5_2,
    QWEN3_MOE,
    QWEN3_NEXT,
    MIXTRAL,
    LLAMA4,
    SMALLTHINKER;

    /** Whether this family is expected to use routed MoE experts. */
    public boolean isMoE() {
        switch (this) {
            case DEEPSEEK_V2:
            case DEEPSEEK_V3:
            case DEEPSEEK_R1:
            case DEEPSEEK_V4_FLASH:
            case KIMI_K2:
            case KIMI_K2_THINKING:
            case KIMI_K2_5:
            case MINIMAX_M2:
            case MINIMAX_M2_1:
            case MINIMAX_M2_5:
            case MINIMAX_M3:
            case GLM4_MOE:
            case GLM5:
            case GLM5_2:
            case QWEN3_MOE:
            case QWEN3_NEXT:
            case MIXTRAL:
            case SMALLTHINKER:
                return true;
            default:
                return false;
        }
    }

    /** Whether MLA-style compressed KV is the default attention path. */
    public boolean prefersMla() {
        switch (this) {
            case DEEPSEEK_V2:
            case DEEPSEEK_V3:
            case DEEPSEEK_R1:
            case DEEPSEEK_V4_FLASH:
            case KIMI_K2:
            case KIMI_K2_THINKING:
            case KIMI_K2_5:
                return true;
            default:
                return false;
        }
    }

    public static KtModelFamily fromString(String name) {
        if (name == null || name.isEmpty()) {
            return GENERIC;
        }
        String n = name.trim().toLowerCase().replace('-', '_').replace(' ', '_');
        if (n.contains("deepseek") && n.contains("v4")) return DEEPSEEK_V4_FLASH;
        if (n.contains("deepseek") && (n.contains("r1") || n.contains("reasoner"))) return DEEPSEEK_R1;
        if (n.contains("deepseek") && n.contains("v3")) return DEEPSEEK_V3;
        if (n.contains("deepseek")) return DEEPSEEK_V2;
        if (n.contains("kimi") && n.contains("thinking")) return KIMI_K2_THINKING;
        if (n.contains("kimi") && n.contains("k2.5") || n.contains("kimi_k2_5")) return KIMI_K2_5;
        if (n.contains("kimi")) return KIMI_K2;
        if (n.contains("minimax") && n.contains("m3")) return MINIMAX_M3;
        if (n.contains("minimax") && n.contains("m2.5")) return MINIMAX_M2_5;
        if (n.contains("minimax") && n.contains("m2.1")) return MINIMAX_M2_1;
        if (n.contains("minimax")) return MINIMAX_M2;
        if (n.contains("glm") && n.contains("5.2")) return GLM5_2;
        if (n.contains("glm") && n.contains("5")) return GLM5;
        if (n.contains("glm")) return GLM4_MOE;
        if (n.contains("qwen3") && n.contains("next")) return QWEN3_NEXT;
        if (n.contains("qwen")) return QWEN3_MOE;
        if (n.contains("mixtral")) return MIXTRAL;
        if (n.contains("llama4") || n.contains("llama_4")) return LLAMA4;
        if (n.contains("smallthinker")) return SMALLTHINKER;
        try {
            return KtModelFamily.valueOf(name.trim().toUpperCase().replace('-', '_'));
        } catch (IllegalArgumentException ex) {
            return GENERIC;
        }
    }
}
