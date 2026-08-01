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

import java.util.Locale;

/** GGUF {@code general.architecture} values commonly emitted by llama.cpp convert. */
public enum LlamaArchitecture {
    LLAMA,
    QWEN2,
    QWEN3,
    GEMMA,
    GEMMA2,
    GEMMA3,
    PHI3,
    PHI3V,
    MISTRAL,
    MIXTRAL,
    GPT2,
    GPTNEOX,
    FALCON,
    UNKNOWN;

    public static LlamaArchitecture fromMetadata(String arch) {
        if (arch == null || arch.isBlank()) return UNKNOWN;
        String a = arch.toLowerCase(Locale.ROOT).trim();
        return switch (a) {
            case "llama", "llama2", "llama3" -> LLAMA;
            case "qwen2", "qwen2vl" -> QWEN2;
            case "qwen3" -> QWEN3;
            case "gemma" -> GEMMA;
            case "gemma2" -> GEMMA2;
            case "gemma3" -> GEMMA3;
            case "phi3", "phi3.5" -> PHI3;
            case "phi3v" -> PHI3V;
            case "mistral" -> MISTRAL;
            case "mixtral" -> MIXTRAL;
            case "gpt2" -> GPT2;
            case "gptneox", "gpt-neox" -> GPTNEOX;
            case "falcon" -> FALCON;
            default -> {
                if (a.contains("llama")) yield LLAMA;
                if (a.contains("qwen3")) yield QWEN3;
                if (a.contains("qwen")) yield QWEN2;
                if (a.contains("gemma")) yield GEMMA;
                if (a.contains("mistral")) yield MISTRAL;
                if (a.contains("phi")) yield PHI3;
                yield UNKNOWN;
            }
        };
    }

    public String metadataPrefix() {
        return switch (this) {
            case LLAMA, MISTRAL, MIXTRAL -> "llama";
            case QWEN2, QWEN3 -> "qwen2";
            case GEMMA, GEMMA2, GEMMA3 -> "gemma";
            case PHI3, PHI3V -> "phi3";
            case GPT2 -> "gpt2";
            case GPTNEOX -> "gptneox";
            case FALCON -> "falcon";
            default -> "llama";
        };
    }
}
