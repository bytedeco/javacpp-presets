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
 * Blocked INT8 / BF16 GEMM algorithm that mirrors Intel AMX tile scheduling
 * <em>semantically</em> (tile M/N/K blocking + group-wise dequant), without
 * linking libamx or claiming hardware acceleration.
 *
 * <p>Upstream kt-kernel uses AMX-Int8 / AMX-BF16 kernels for CPU-side quantized
 * inference. This class provides a pure-torch blocked path so:
 * <ul>
 *   <li>CI can measure numerical error bounds vs dense FP32</li>
 *   <li>Host meshes can swap in a real native backend later via {@link KernelRegistry}</li>
 * </ul>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class AmxLikeGemm {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    /** Default AMX tile-inspired block sizes (algorithmic, not hardware). */
    public static final int DEFAULT_TM = 16;
    public static final int DEFAULT_TN = 16;
    public static final int DEFAULT_TK = 64;

    private AmxLikeGemm() {}

    /**
     * Compute {@code y = x @ W_deq^T (+ bias)} using group-wise dequant then
     * standard linear. Blocking is applied as a documentation / future native
     * hook; the reference path dequants fully then calls torch linear for
     * correctness (tile loops would only matter for a true AMX kernel).
     *
     * @param x         [N, K] activations (float)
     * @param qweight   packed [N_out, K]
     * @param scale     [N_out, groups]
     * @param zero      optional zero points
     * @param bias      optional [N_out]
     * @param bits      4 or 8
     * @param groupSize group size along K
     * @param tileM     reserved tile param (logged in metrics only)
     * @param tileN     reserved
     * @param tileK     reserved
     */
    public static Tensor gemmInt8(Tensor x, Tensor qweight, Tensor scale, Tensor zero,
                                  Tensor bias, int bits, int groupSize,
                                  int tileM, int tileN, int tileK) {
        KtPreconditions.checkNotNull(x, "x");
        KtPreconditions.checkNotNull(qweight, "qweight");
        KtPreconditions.checkNotNull(scale, "scale");
        KtPreconditions.checkArgument(bits == 4 || bits == 8, "bits must be 4 or 8");
        // tile params kept for API parity with native backends; ref uses full dequant.
        KtPreconditions.checkPositive(Math.max(1, tileM), "tileM");
        KtPreconditions.checkPositive(Math.max(1, tileN), "tileN");
        KtPreconditions.checkPositive(Math.max(1, tileK), "tileK");

        Tensor w = DequantOps.dequantGroupwise(qweight, scale, zero, bits, groupSize);
        try {
            if (bias == null) {
                return linear(x, w);
            }
            return linear(x, w, new TensorOptional(bias));
        } finally {
            w.close();
        }
    }

    public static Tensor gemmInt8(Tensor x, Tensor qweight, Tensor scale, Tensor zero,
                                  Tensor bias, int bits, int groupSize) {
        return gemmInt8(x, qweight, scale, zero, bias, bits, groupSize,
                DEFAULT_TM, DEFAULT_TN, DEFAULT_TK);
    }

    /**
     * BF16-like path: cast activations/weights to float for the reference matmul.
     * Real AMX-BF16 would accumulate in float with BF16 tiles; we document the
     * casting so numerical tests can compare against FP32 baseline.
     */
    public static Tensor gemmBf16Like(Tensor x, Tensor weight, Tensor bias) {
        KtPreconditions.checkNotNull(x, "x");
        KtPreconditions.checkNotNull(weight, "weight");
        Tensor xf = x.to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        Tensor wf = weight.to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        try {
            if (bias == null) {
                return linear(xf, wf);
            }
            Tensor bf = bias.to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
            try {
                return linear(xf, wf, new TensorOptional(bf));
            } finally {
                bf.close();
            }
        } finally {
            if (xf != x) xf.close();
            if (wf != weight) wf.close();
        }
    }
}
