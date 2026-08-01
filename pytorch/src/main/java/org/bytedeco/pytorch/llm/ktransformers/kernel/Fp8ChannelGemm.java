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
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.llm.ktransformers.util.KtPreconditions;

import static org.bytedeco.pytorch.global.torch.linear;

/**
 * FP8 per-channel GEMM reference (e4m3fn max ≈ 448 scale convention).
 *
 * <p>Aligns with upstream "Native BF16 and FP8 per channel Precision". Storage is
 * float-simulated FP8 range codes + per-out-channel scale; compute is dequant then
 * torch linear. Suitable for golden tests and hybrid weight demos.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Fp8ChannelGemm {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** e4m3fn finite max used by the reference quantizer. */
    public static final float E4M3_MAX = 448.0f;

    private Fp8ChannelGemm() {}

    public static KtKernelBackend.QuantizedWeight quantize(Tensor weight) {
        return DequantOps.quantizeFp8PerChannel(weight);
    }

    public static Tensor dequant(Tensor qweight, Tensor scale) {
        return DequantOps.dequantFp8PerChannel(qweight, scale);
    }

    /**
     * {@code y = x @ W_fp8_deq^T (+ bias)}.
     *
     * @param x       [N, K]
     * @param qweight FP8-range codes [N_out, K]
     * @param scale   per-out channel [N_out]
     * @param bias    optional
     */
    public static Tensor gemm(Tensor x, Tensor qweight, Tensor scale, Tensor bias) {
        KtPreconditions.checkNotNull(x, "x");
        KtPreconditions.checkNotNull(qweight, "qweight");
        KtPreconditions.checkNotNull(scale, "scale");
        Tensor w = dequant(qweight, scale);
        try {
            if (bias == null) {
                return linear(x, w);
            }
            return linear(x, w, new TensorOptional(bias));
        } finally {
            w.close();
        }
    }

    /** Round-trip max abs error helper for benchmarks. */
    public static double roundTripError(Tensor weight) {
        KtKernelBackend.QuantizedWeight q = quantize(weight);
        try {
            Tensor d = dequant(q.qweight, q.scale);
            try {
                return DequantOps.maxAbsError(weight, d);
            } finally {
                d.close();
            }
        } finally {
            q.qweight.close();
            q.scale.close();
            if (q.zero != null) q.zero.close();
        }
    }
}
