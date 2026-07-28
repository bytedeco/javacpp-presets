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
package org.bytedeco.pytorch.llm.quantization;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.linear;

/**
 * Lightweight container for a quantized weight matrix plus optional scale /
 * zero-point, with a dequantized linear matmul helper.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class QuantizedLinear implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final Tensor weight;
    private final Tensor scale;
    private final Tensor zeroPoint;

    public QuantizedLinear(Tensor weight, Tensor scale, Tensor zeroPoint) {
        this.weight = Objects.requireNonNull(weight, "weight");
        this.scale = scale;
        this.zeroPoint = zeroPoint;
    }

    public Tensor getWeight() { return weight; }
    public Tensor getScale() { return scale; }
    public Tensor getZeroPoint() { return zeroPoint; }

    /** Dequantize weight to float32 using scale (and optional zero-point). */
    public Tensor dequantizeWeight() {
        Tensor w = weight.to(ScalarType.Float);
        if (scale != null) {
            w = w.mul(scale.to(ScalarType.Float));
        }
        if (zeroPoint != null) {
            w = w.sub(zeroPoint.to(ScalarType.Float));
        }
        return w;
    }

    /**
     * Apply a dequantized linear transform: {@code y = x @ W_deq^T + b}.
     * Bias may be {@code null}.
     */
    public Tensor forward(Tensor input, Tensor bias) {
        Tensor w = dequantizeWeight();
        try {
            if (bias == null) {
                return linear(input, w);
            }
            return linear(input, w, new TensorOptional(bias));
        } finally {
            w.close();
        }
    }

    public Tensor forward(Tensor input) {
        return forward(input, null);
    }

    @Override
    public void close() {
        weight.close();
        if (scale != null) {
            scale.close();
        }
        if (zeroPoint != null) {
            zeroPoint.close();
        }
    }
}
