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

package org.bytedeco.pytorch.llm.llamacpp.quant;

import org.bytedeco.pytorch.data.gguf.GGUFConstants;

/** GGML type ids used by dequant / loader. */
public enum GgmlQuantType {
    F32(GGUFConstants.GGML_TYPE_F32, false),
    F16(GGUFConstants.GGML_TYPE_F16, false),
    BF16(GGUFConstants.GGML_TYPE_BF16, false),
    Q4_0(GGUFConstants.GGML_TYPE_Q4_0, true),
    Q4_1(GGUFConstants.GGML_TYPE_Q4_1, true),
    Q5_0(GGUFConstants.GGML_TYPE_Q5_0, true),
    Q5_1(GGUFConstants.GGML_TYPE_Q5_1, true),
    Q8_0(GGUFConstants.GGML_TYPE_Q8_0, true),
    Q8_1(GGUFConstants.GGML_TYPE_Q8_1, true),
    I8(GGUFConstants.GGML_TYPE_I8, false),
    I16(GGUFConstants.GGML_TYPE_I16, false),
    I32(GGUFConstants.GGML_TYPE_I32, false),
    I64(GGUFConstants.GGML_TYPE_I64, false),
    F64(GGUFConstants.GGML_TYPE_F64, false),
    UNKNOWN(-1, false);

    private final int id;
    private final boolean quantized;

    GgmlQuantType(int id, boolean quantized) {
        this.id = id;
        this.quantized = quantized;
    }

    public int id() { return id; }
    public boolean quantized() { return quantized; }

    public static GgmlQuantType fromId(int id) {
        for (GgmlQuantType t : values()) {
            if (t.id == id) return t;
        }
        return UNKNOWN;
    }
}
