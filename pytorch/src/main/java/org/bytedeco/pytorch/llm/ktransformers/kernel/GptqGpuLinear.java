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
 * GPU-side GPTQ-style quantized linear (semantic reference).
 *
 * <p>Upstream KT supports GPTQ on GPU for MoE / dense layers. This module stores
 * group-wise INT4/INT8 packs and runs matmul via the active {@link KtKernelBackend}.
 * When tensors live on CUDA, torch linear after dequant still executes on that
 * device — no custom CUDA kernel is claimed.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GptqGpuLinear extends Module {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final QuantLinearOp inner;

    public GptqGpuLinear(long inFeatures, long outFeatures, int bits, int groupSize,
                         KtKernelBackend backend) {
        super("GptqGpuLinear");
        KtPreconditions.checkArgument(bits == 4 || bits == 8, "GPTQ bits must be 4 or 8");
        this.inner = register_module("inner",
                new QuantLinearOp(inFeatures, outFeatures, bits, groupSize, backend, false));
    }

    public GptqGpuLinear(long inFeatures, long outFeatures, int bits, int groupSize) {
        this(inFeatures, outFeatures, bits, groupSize, KernelRegistry.defaultBackend());
    }

    /** Pack floating weights into GPTQ group-wise layout. */
    public void pack(Tensor weight) {
        inner.packFromFloat(weight, null);
    }

    public QuantLinearOp inner() {
        return inner;
    }

    @Override
    public Tensor forward(Tensor x) {
        return inner.forward(x);
    }

    public static GptqGpuLinear fromFloat(Tensor weight, int bits, int groupSize) {
        long out = weight.size(0);
        long in = weight.size(1);
        GptqGpuLinear m = new GptqGpuLinear(in, out, bits, groupSize);
        m.pack(weight);
        return m;
    }
}
