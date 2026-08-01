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

/** Export target formats supported by Studio. */
public enum ExportFormat {
    SAFETENSORS_16BIT("16-bit safetensors"),
    SAFETENSORS_BF16("bf16 safetensors"),
    SAFETENSORS_FP32("fp32 safetensors"),
    LORA_ADAPTER("lora adapter"),
    MERGED_16BIT("merged 16-bit"),
    GGUF("gguf"),
    GGUF_Q4_K_M("gguf Q4_K_M"),
    GGUF_Q5_K_M("gguf Q5_K_M"),
    GGUF_Q8_0("gguf Q8_0");

    private final String label;
    ExportFormat(String label) { this.label = label; }
    public String label() { return label; }

    public boolean isGguf() {
        return name().startsWith("GGUF");
    }

    public static ExportFormat fromLabel(String raw) {
        if (raw == null || raw.isBlank()) throw new IllegalArgumentException("export format required");
        String t = raw.trim();
        for (ExportFormat f : values()) {
            if (f.name().equalsIgnoreCase(t) || f.label.equalsIgnoreCase(t)
                    || f.name().replace('_', '-').equalsIgnoreCase(t)) {
                return f;
            }
        }
        String lower = t.toLowerCase();
        if (lower.contains("gguf") && lower.contains("q4")) return GGUF_Q4_K_M;
        if (lower.contains("gguf") && lower.contains("q5")) return GGUF_Q5_K_M;
        if (lower.contains("gguf") && lower.contains("q8")) return GGUF_Q8_0;
        if (lower.contains("gguf")) return GGUF;
        if (lower.contains("lora") || lower.contains("adapter")) return LORA_ADAPTER;
        if (lower.contains("merge")) return MERGED_16BIT;
        if (lower.contains("bf16")) return SAFETENSORS_BF16;
        if (lower.contains("fp32") || lower.contains("32")) return SAFETENSORS_FP32;
        if (lower.contains("16") || lower.contains("safetensor")) return SAFETENSORS_16BIT;
        throw new IllegalArgumentException("Unknown export format: " + raw);
    }
}
