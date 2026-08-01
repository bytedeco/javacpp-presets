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
 * Weight quantization backend (mirrors LLaMA-Factory {@code quantization_method}).
 *
 * <p>Load path is provided by {@code factory.extras.quant.QuantLoaderRegistry}
 * and/or {@code llm.bitsandbytes} / {@code llm.quantization}.
 */
public enum QuantizationMethod {
    NONE,
    BNB,
    GPTQ,
    AWQ,
    HQQ,
    AQLM,
    EETQ,
    FP8,
    /** LLM.int8() via bitsandbytes. */
    LLM_INT8;

    public static QuantizationMethod parse(String raw) {
        if (raw == null || raw.isBlank()) {
            return NONE;
        }
        String s = raw.trim().toLowerCase(Locale.ROOT).replace('-', '_').replace('.', '_');
        return switch (s) {
            case "none", "fp16", "bf16", "fp32" -> NONE;
            case "bnb", "bitsandbytes", "bits_and_bytes" -> BNB;
            case "gptq" -> GPTQ;
            case "awq" -> AWQ;
            case "hqq" -> HQQ;
            case "aqlm" -> AQLM;
            case "eetq" -> EETQ;
            case "fp8", "float8" -> FP8;
            case "llm_int8", "int8", "llm.int8" -> LLM_INT8;
            default -> {
                try {
                    yield valueOf(s.toUpperCase(Locale.ROOT));
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(
                            "Unknown quantization method '" + raw + "'; expected one of "
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
