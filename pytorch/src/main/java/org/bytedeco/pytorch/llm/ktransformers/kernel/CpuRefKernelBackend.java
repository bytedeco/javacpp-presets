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
package org.bytedeco.pytorch.llm.ktransformers.kernel;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;

/**
 * Default pure-torch / host reference backend for kt-kernel quant ops.
 *
 * <p>Always available in CI. Does <strong>not</strong> claim native AMX/AVX
 * linkage; {@link Capability#AMX_LIKE_INT8} is satisfied by the same group-wise
 * math used in production planning, executed via {@link DequantOps}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class CpuRefKernelBackend implements KtKernelBackend {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    public static final String NAME = "cpu-ref";

    @Override
    public String name() {
        return NAME;
    }

    @Override
    public boolean supports(Capability capability) {
        switch (capability) {
            case INT4_GROUPWISE:
            case INT8_GROUPWISE:
            case FP8_PER_CHANNEL:
            case AMX_LIKE_INT8:
            case AMX_LIKE_BF16:
            case GPTQ_GPU:
            case AVX2_FALLBACK:
                return true;
            default:
                return false;
        }
    }

    @Override
    public Tensor dequant(Tensor qweight, Tensor scale, Tensor zero, int bits, int groupSize) {
        if (groupSize <= 1 && bits == 8) {
            // treat as fp8-channel when groupSize==1 and scale is 1-D per out
            if (scale != null && scale.dim() == 1) {
                return DequantOps.dequantFp8PerChannel(qweight, scale);
            }
        }
        return DequantOps.dequantGroupwise(qweight, scale, zero, bits, groupSize);
    }

    @Override
    public Tensor quantMatmul(Tensor x, Tensor qweight, Tensor scale, Tensor zero,
                              Tensor bias, int bits, int groupSize) {
        Tensor w = dequant(qweight, scale, zero, bits, groupSize);
        try {
            return DequantOps.matmulDequant(x, w, bias);
        } finally {
            w.close();
        }
    }

    @Override
    public QuantizedWeight quantizeWeight(Tensor weight, int bits, int groupSize) {
        if (bits == 8 && groupSize <= 1) {
            return DequantOps.quantizeFp8PerChannel(weight);
        }
        return DequantOps.quantizeGroupwise(weight, bits, Math.max(1, groupSize));
    }

    @Override
    public void close() {
        // no native resources
    }
}
