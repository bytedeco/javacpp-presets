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
 * RoPE scaling strategy (mirrors LLaMA-Factory / HF {@code rope_scaling}).
 */
public enum RopeScalingType {
    NONE,
    LINEAR,
    DYNAMIC,
    YARN,
    LLAMA3;

    public static RopeScalingType parse(String raw) {
        if (raw == null || raw.isBlank()) {
            return NONE;
        }
        String s = raw.trim().toLowerCase(Locale.ROOT).replace('-', '_');
        return switch (s) {
            case "none", "null", "off" -> NONE;
            case "linear" -> LINEAR;
            case "dynamic", "dynamic_ntk" -> DYNAMIC;
            case "yarn" -> YARN;
            case "llama3", "llama_3" -> LLAMA3;
            default -> {
                try {
                    yield valueOf(s.toUpperCase(Locale.ROOT));
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(
                            "Unknown rope scaling '" + raw + "'; expected one of "
                                    + java.util.Arrays.toString(values()), e);
                }
            }
        };
    }

    public boolean enabled() {
        return this != NONE;
    }

    public String wireName() {
        return name().toLowerCase(Locale.ROOT);
    }
}
