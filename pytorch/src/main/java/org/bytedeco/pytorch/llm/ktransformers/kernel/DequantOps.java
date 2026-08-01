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
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.llm.ktransformers.util.KtPreconditions;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.linear;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Group-wise / channel-wise dequant helpers for kt-kernel reference backends.
 *
 * <p>Host-side float paths mirror {@code llm.bitsandbytes.BitsAndBytes} for
 * JavaCPP stability. Math matches common GPTQ/BNB group-wise layouts:
 * {@code W ≈ scale * (q - zero)} with symmetric zero=0 by default.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DequantOps {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private DequantOps() {}

    public static int qmin(int bits) {
        if (bits == 4) return -8;
        if (bits == 8) return -128;
        throw new IllegalArgumentException("bits must be 4 or 8, got " + bits);
    }

    public static int qmax(int bits) {
        if (bits == 4) return 7;
        if (bits == 8) return 127;
        throw new IllegalArgumentException("bits must be 4 or 8, got " + bits);
    }

    /**
     * Quantize {@code weight} [out, in] group-wise along the last dim.
     * qweight stored as float codes for portable round-trip; scale [out*groups] flat
     * reshaped to [out, groups]; zero symmetric zeros.
     */
    public static KtKernelBackend.QuantizedWeight quantizeGroupwise(Tensor weight, int bits, int groupSize) {
        KtPreconditions.checkNotNull(weight, "weight");
        KtPreconditions.checkArgument(weight.dim() == 2, "weight must be rank-2 [out, in]");
        KtPreconditions.checkArgument(bits == 4 || bits == 8, "bits must be 4 or 8");
        KtPreconditions.checkPositive(groupSize, "groupSize");

        long outF = weight.size(0);
        long inF = weight.size(1);
        int out = (int) outF;
        int in = (int) inF;
        int groups = (in + groupSize - 1) / groupSize;
        float[] data = toFloatArray(weight);
        float[] codes = new float[out * in];
        float[] scales = new float[out * groups];
        float[] zeros = new float[out * groups];
        int qHi = qmax(bits);
        int qLo = qmin(bits);

        for (int o = 0; o < out; o++) {
            for (int g = 0; g < groups; g++) {
                int start = g * groupSize;
                int end = Math.min(in, start + groupSize);
                float amax = 0f;
                int base = o * in;
                for (int i = start; i < end; i++) {
                    float a = Math.abs(data[base + i]);
                    if (a > amax) amax = a;
                }
                if (amax < 1e-12f) amax = 1e-12f;
                float scale = amax / qHi;
                scales[o * groups + g] = scale;
                zeros[o * groups + g] = 0f;
                for (int i = start; i < end; i++) {
                    int q = Math.round(data[base + i] / scale);
                    if (q < qLo) q = qLo;
                    if (q > qHi) q = qHi;
                    codes[base + i] = q;
                }
            }
        }
        Tensor qweight = tensor(codes).reshape(outF, inF);
        Tensor scaleT = tensor(scales).reshape(outF, groups);
        Tensor zeroT = tensor(zeros).reshape(outF, groups);
        return new KtKernelBackend.QuantizedWeight(qweight, scaleT, zeroT, bits, groupSize);
    }

    /** Dequantize group-wise codes back to float [out, in]. */
    public static Tensor dequantGroupwise(Tensor qweight, Tensor scale, Tensor zero, int bits, int groupSize) {
        KtPreconditions.checkNotNull(qweight, "qweight");
        KtPreconditions.checkNotNull(scale, "scale");
        KtPreconditions.checkPositive(groupSize, "groupSize");

        long outF = qweight.size(0);
        long inF = qweight.size(1);
        int out = (int) outF;
        int in = (int) inF;
        int groups = (int) scale.size(scale.dim() - 1);
        if (scale.dim() == 1) {
            groups = (in + groupSize - 1) / groupSize;
        }
        float[] codes = toFloatArray(qweight);
        float[] scales = toFloatArray(scale);
        float[] zeros = zero != null ? toFloatArray(zero) : new float[out * groups];
        float[] outData = new float[out * in];

        for (int o = 0; o < out; o++) {
            for (int g = 0; g < groups; g++) {
                int start = g * groupSize;
                int end = Math.min(in, start + groupSize);
                int sg = o * groups + Math.min(g, groups - 1);
                float s = scales[Math.min(sg, scales.length - 1)];
                float z = zeros.length == 0 ? 0f : zeros[Math.min(sg, zeros.length - 1)];
                int base = o * in;
                for (int i = start; i < end; i++) {
                    outData[base + i] = (codes[base + i] - z) * s;
                }
            }
        }
        return tensor(outData).reshape(outF, inF);
    }

    /** Per-channel (out) FP8-range scaling using e4m3fn max 448. */
    public static KtKernelBackend.QuantizedWeight quantizeFp8PerChannel(Tensor weight) {
        KtPreconditions.checkArgument(weight.dim() == 2, "weight must be rank-2");
        long outF = weight.size(0);
        long inF = weight.size(1);
        int out = (int) outF;
        int in = (int) inF;
        float[] data = toFloatArray(weight);
        float[] qdata = new float[out * in];
        float[] scales = new float[out];
        float[] zeros = new float[out];
        final float maxVal = 448.0f;
        for (int o = 0; o < out; o++) {
            float amax = 0f;
            int base = o * in;
            for (int i = 0; i < in; i++) {
                float a = Math.abs(data[base + i]);
                if (a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            float scale = amax / maxVal;
            scales[o] = scale;
            for (int i = 0; i < in; i++) {
                float v = data[base + i] / scale;
                if (v > maxVal) v = maxVal;
                if (v < -maxVal) v = -maxVal;
                qdata[base + i] = v;
            }
        }
        return new KtKernelBackend.QuantizedWeight(
                tensor(qdata).reshape(outF, inF),
                tensor(scales),
                tensor(zeros),
                8,
                1);
    }

    public static Tensor dequantFp8PerChannel(Tensor qweight, Tensor scale) {
        long outF = qweight.size(0);
        long inF = qweight.size(1);
        int out = (int) outF;
        int in = (int) inF;
        float[] q = toFloatArray(qweight);
        float[] s = toFloatArray(scale);
        float[] outData = new float[out * in];
        for (int o = 0; o < out; o++) {
            float sc = s[Math.min(o, s.length - 1)];
            int base = o * in;
            for (int i = 0; i < in; i++) {
                outData[base + i] = q[base + i] * sc;
            }
        }
        return tensor(outData).reshape(outF, inF);
    }

    /** y = x @ W_deq^T (+ bias). */
    public static Tensor matmulDequant(Tensor x, Tensor wDeq, Tensor bias) {
        if (bias == null) {
            return linear(x, wDeq);
        }
        return linear(x, wDeq, new TensorOptional(bias));
    }

    /** Max abs error between two float tensors (host). */
    public static double maxAbsError(Tensor a, Tensor b) {
        float[] fa = toFloatArray(a);
        float[] fb = toFloatArray(b);
        int n = Math.min(fa.length, fb.length);
        double max = 0;
        for (int i = 0; i < n; i++) {
            double d = Math.abs(fa[i] - fb[i]);
            if (d > max) max = d;
        }
        return max;
    }

    public static float[] toFloatArray(Tensor t) {
        Tensor f = t.to(ScalarType.Float).contiguous().reshape(-1);
        long n = f.numel();
        float[] data = new float[(int) n];
        FloatIndexer idx = f.createIndexer();
        try {
            for (long i = 0; i < n; i++) {
                data[(int) i] = idx.get(i);
            }
        } finally {
            idx.release();
        }
        return data;
    }
}
