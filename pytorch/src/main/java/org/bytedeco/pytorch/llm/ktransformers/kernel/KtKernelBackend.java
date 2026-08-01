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

import org.bytedeco.pytorch.Tensor;

/**
 * SPI for kt-kernel style quantized linear backends.
 *
 * <p>Upstream ships AMX/AVX/CUDA kernels; this interface keeps the same call
 * surface while defaulting to pure-torch reference implementations that are
 * numerically testable on any host.
 */
public interface KtKernelBackend extends AutoCloseable {

    enum Capability {
        INT4_GROUPWISE,
        INT8_GROUPWISE,
        FP8_PER_CHANNEL,
        AMX_LIKE_INT8,
        AMX_LIKE_BF16,
        GPTQ_GPU,
        AVX2_FALLBACK
    }

    String name();

    boolean supports(Capability capability);

    /**
     * Dequantize packed integer / fp8 weights to floating weights.
     *
     * @param qweight packed weights (layout backend-defined; ref uses int32/int8/float storage)
     * @param scale   group or channel scales
     * @param zero    optional zero-points (may be null for symmetric)
     * @param bits    4 or 8 (or 8 for fp8 path)
     * @param groupSize group size along input features
     */
    Tensor dequant(Tensor qweight, Tensor scale, Tensor zero, int bits, int groupSize);

    /**
     * Fused dequant + matmul: {@code y = x @ W_dequant^T} (+ optional bias).
     *
     * @param x         [N, K] activations
     * @param qweight   packed [N_out, K] or backend layout
     * @param scale     scales
     * @param zero      zero points or null
     * @param bias      optional [N_out]
     * @param bits      weight bits
     * @param groupSize group size
     */
    Tensor quantMatmul(Tensor x, Tensor qweight, Tensor scale, Tensor zero,
                       Tensor bias, int bits, int groupSize);

    /**
     * Quantize a floating weight matrix for storage / later matmul.
     *
     * @return bundle {qweight, scale, zero}
     */
    QuantizedWeight quantizeWeight(Tensor weight, int bits, int groupSize);

    /** Opaque quantized weight triple. */
    final class QuantizedWeight {
        public final Tensor qweight;
        public final Tensor scale;
        public final Tensor zero;
        public final int bits;
        public final int groupSize;

        public QuantizedWeight(Tensor qweight, Tensor scale, Tensor zero, int bits, int groupSize) {
            this.qweight = qweight;
            this.scale = scale;
            this.zero = zero;
            this.bits = bits;
            this.groupSize = groupSize;
        }
    }

    @Override
    default void close() {}
}
