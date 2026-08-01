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
import org.bytedeco.pytorch.llm.ktransformers.util.KtPreconditions;
import org.bytedeco.pytorch.nn.Module;

/**
 * Quantized linear layer used by kt-kernel injection paths.
 *
 * <p>Holds packed weights ({@code qweight/scale/zero}) and dispatches matmul
 * through a {@link KtKernelBackend}. This is the drop-in replacement target for
 * dense {@code nn.Linear} layers during model injection.
 *
 * <p>Does not claim native AMX/AVX linkage; performance is that of the registered
 * backend (default {@link CpuRefKernelBackend}).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class QuantLinearOp extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final KtKernelBackend backend;
    private final int bits;
    private final int groupSize;
    private final long inFeatures;
    private final long outFeatures;
    private final boolean hasBias;

    /** Packed codes [out, in] (float storage for portable ref). */
    public Tensor qweight;
    /** Scales [out, groups] or [out] for per-channel. */
    public Tensor scale;
    /** Zero points matching scale layout (may be null for symmetric). */
    public Tensor zero;
    /** Optional bias [out]. */
    public Tensor bias;

    public QuantLinearOp(long inFeatures, long outFeatures, int bits, int groupSize,
                         KtKernelBackend backend, boolean bias) {
        super("QuantLinearOp");
        KtPreconditions.checkPositive((int) inFeatures, "inFeatures");
        KtPreconditions.checkPositive((int) outFeatures, "outFeatures");
        KtPreconditions.checkArgument(bits == 4 || bits == 8, "bits must be 4 or 8");
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.bits = bits;
        this.groupSize = Math.max(1, groupSize);
        this.backend = backend != null ? backend : KernelRegistry.defaultBackend();
        this.hasBias = bias;
        this.qweight = null;
        this.scale = null;
        this.zero = null;
        this.bias = null;
    }

    public QuantLinearOp(long inFeatures, long outFeatures, int bits, int groupSize) {
        this(inFeatures, outFeatures, bits, groupSize, null, false);
    }

    public int bits() { return bits; }
    public int groupSize() { return groupSize; }
    public long inFeatures() { return inFeatures; }
    public long outFeatures() { return outFeatures; }
    public KtKernelBackend backend() { return backend; }

    /**
     * Pack a floating weight matrix [out, in] into this layer's storage.
     * Closes any previously held packed tensors.
     */
    public void packFromFloat(Tensor weightFp, Tensor biasFp) {
        KtPreconditions.checkNotNull(weightFp, "weightFp");
        KtPreconditions.checkArgument(weightFp.dim() == 2, "weight must be [out, in]");
        KtPreconditions.checkArgument(weightFp.size(0) == outFeatures
                        && weightFp.size(1) == inFeatures,
                "weight shape mismatch");
        closePacked();
        KtKernelBackend.QuantizedWeight qw = backend.quantizeWeight(weightFp, bits, groupSize);
        this.qweight = qw.qweight;
        this.scale = qw.scale;
        this.zero = qw.zero;
        if (biasFp != null) {
            this.bias = biasFp.clone();
        } else if (hasBias) {
            this.bias = org.bytedeco.pytorch.global.torch.zeros(outFeatures);
        }
    }

    /** Assign already-packed tensors (takes ownership of the handles). */
    public void setPacked(Tensor qweight, Tensor scale, Tensor zero, Tensor bias) {
        closePacked();
        this.qweight = qweight;
        this.scale = scale;
        this.zero = zero;
        this.bias = bias;
    }

    @Override
    public Tensor forward(Tensor x) {
        KtPreconditions.checkState(qweight != null && scale != null, "QuantLinearOp not packed");
        // Support [..., in] by reshaping to 2-D then restoring.
        long[] shape = new long[(int) x.dim()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = x.size(i);
        }
        Tensor flat = x.dim() == 2 ? x : x.reshape(-1, inFeatures);
        Tensor y = backend.quantMatmul(flat, qweight, scale, zero, bias, bits, groupSize);
        if (x.dim() == 2) {
            return y;
        }
        shape[shape.length - 1] = outFeatures;
        return y.reshape(shape);
    }

    /** Materialize dequantized float weights [out, in] (caller owns). */
    public Tensor dequantWeight() {
        KtPreconditions.checkState(qweight != null, "not packed");
        return backend.dequant(qweight, scale, zero, bits, groupSize);
    }

    private void closePacked() {
        if (qweight != null) { qweight.close(); qweight = null; }
        if (scale != null) { scale.close(); scale = null; }
        if (zero != null) { zero.close(); zero = null; }
        if (bias != null) { bias.close(); bias = null; }
    }

    @Override
    public void close() {
        closePacked();
        super.close();
    }

    /**
     * Build a QuantLinearOp from an existing float Linear weight (no bias).
     */
    public static QuantLinearOp fromFloatWeight(Tensor weight, int bits, int groupSize,
                                                 KtKernelBackend backend) {
        long out = weight.size(0);
        long in = weight.size(1);
        QuantLinearOp op = new QuantLinearOp(in, out, bits, groupSize, backend, false);
        op.packFromFloat(weight, null);
        return op;
    }
}
