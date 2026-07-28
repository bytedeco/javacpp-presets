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
package org.bytedeco.pytorch.distributed;

import static org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * Mixed-precision policy for {@link NativeFSDPTrainer} (param / reduce / buffer dtypes).
 *
 * <p>On Mac/CPU, prefer {@link #fp32()} — bf16/fp16 may be emulated or unsupported
 * depending on the libtorch build. Trainers record the config; actual cast is best-effort.
 */
public final class MixedPrecisionConfig {

    private final byte paramDtype;
    private final byte reduceDtype;
    private final byte bufferDtype;
    private final String label;

    private MixedPrecisionConfig(byte paramDtype, byte reduceDtype, byte bufferDtype, String label) {
        this.paramDtype = paramDtype;
        this.reduceDtype = reduceDtype;
        this.bufferDtype = bufferDtype;
        this.label = label;
    }

    public static MixedPrecisionConfig fp32() {
        return new MixedPrecisionConfig(ScalarType.Float.value, ScalarType.Float.value, ScalarType.Float.value, "fp32");
    }

    public static MixedPrecisionConfig fp16() {
        return new MixedPrecisionConfig(ScalarType.Half.value, ScalarType.Half.value, ScalarType.Half.value, "fp16");
    }

    public static MixedPrecisionConfig bf16() {
        return new MixedPrecisionConfig(ScalarType.BFloat16.value, ScalarType.BFloat16.value, ScalarType.BFloat16.value, "bf16");
    }

    public static MixedPrecisionConfig of(byte param, byte reduce, byte buffer) {
        return new MixedPrecisionConfig(param, reduce, buffer,
                "param=" + param + ",reduce=" + reduce + ",buffer=" + buffer);
    }

    public byte paramDtype() { return paramDtype; }
    public byte reduceDtype() { return reduceDtype; }
    public byte bufferDtype() { return bufferDtype; }
    public String label() { return label; }

    public boolean isFullPrecision() {
        return paramDtype == ScalarType.Float.value
                && reduceDtype == ScalarType.Float.value
                && bufferDtype == ScalarType.Float.value;
    }

    @Override
    public String toString() {
        return "MixedPrecisionConfig{" + label + '}';
    }
}
