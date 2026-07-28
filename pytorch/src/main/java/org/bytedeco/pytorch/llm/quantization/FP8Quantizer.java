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

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.linear;
import static org.bytedeco.pytorch.global.torch.max;
import static org.bytedeco.pytorch.global.torch.min;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Specialized FP8 quantizer (E4M3FN for inference, E5M2 for training).
 *
 * <pre>{@code
 * try (FP8Quantizer q = FP8Quantizer.e4m3fn()) {
 *     Tensor qw = q.quantize(weight);
 *     Tensor restored = q.dequantize(qw);
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class FP8Quantizer implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public enum FP8Type {
        /** 1 sign + 4 exp + 3 mantissa — preferred for inference. */
        E4M3FN(ScalarType.Float8_e4m3fn, 448.0f),
        /** 1 sign + 5 exp + 2 mantissa — preferred for training. */
        E5M2(ScalarType.Float8_e5m2, 57344.0f);

        private final ScalarType scalarType;
        private final float maxVal;

        FP8Type(ScalarType scalarType, float maxVal) {
            this.scalarType = scalarType;
            this.maxVal = maxVal;
        }

        public ScalarType toScalarType() { return scalarType; }
        public float getMaxVal() { return maxVal; }
    }

    private final FP8Type fp8Type;
    private final List<Float> scaleHistory = new ArrayList<>();
    private float observerMin = Float.MAX_VALUE;
    private float observerMax = -Float.MAX_VALUE;
    private boolean calibrated;

    public FP8Quantizer() {
        this(FP8Type.E4M3FN);
    }

    public FP8Quantizer(FP8Type fp8Type) {
        this.fp8Type = Objects.requireNonNull(fp8Type, "fp8Type");
    }

    public static FP8Quantizer e4m3fn() { return new FP8Quantizer(FP8Type.E4M3FN); }
    public static FP8Quantizer e5m2() { return new FP8Quantizer(FP8Type.E5M2); }

    public Tensor quantize(Tensor input) {
        return input.to(fp8Type.toScalarType());
    }

    public Tensor dequantize(Tensor input) {
        return input.to(ScalarType.Float);
    }

    /** Observe min/max for optional calibration bookkeeping. */
    public void observe(Tensor t) {
        float minV = min(t).item_float();
        float maxV = max(t).item_float();
        if (minV < observerMin) {
            observerMin = minV;
        }
        if (maxV > observerMax) {
            observerMax = maxV;
        }
        scaleHistory.add(1.0f);
        calibrated = true;
    }

    public float computeScale(float absMax) {
        return absMax > 0 ? fp8Type.getMaxVal() / absMax : 1.0f;
    }

    public FP8Linear quantizeLinear(Tensor weight) {
        return quantizeLinear(weight, null);
    }

    public FP8Linear quantizeLinear(Tensor weight, Tensor bias) {
        Tensor qw = quantize(weight);
        Tensor scale = tensor(1.0f);
        return new FP8Linear(qw, scale, bias, this);
    }

    public FP8Type getType() { return fp8Type; }
    public boolean isCalibrated() { return calibrated; }
    public float getObserverMin() { return observerMin; }
    public float getObserverMax() { return observerMax; }
    public List<Float> getScaleHistory() { return Collections.unmodifiableList(scaleHistory); }

    @Override
    public void close() {
        scaleHistory.clear();
    }

    /** Linear layer with FP8 weights; dequantizes to float for matmul. */
    public static final class FP8Linear implements AutoCloseable {
        private final Tensor weight;
        private final Tensor scale;
        private final Tensor bias;
        private final FP8Quantizer parent;

        FP8Linear(Tensor weight, Tensor scale, Tensor bias, FP8Quantizer parent) {
            this.weight = weight;
            this.scale = scale;
            this.bias = bias;
            this.parent = parent;
        }

        public Tensor forward(Tensor input) {
            Tensor w = parent.dequantize(weight);
            try {
                if (bias == null) {
                    return linear(input, w);
                }
                return linear(input, w, new TensorOptional(bias));
            } finally {
                w.close();
            }
        }

        public Tensor getWeight() { return weight; }
        public Tensor getScale() { return scale; }
        public Tensor getBias() { return bias; }

        @Override
        public void close() {
            weight.close();
            if (scale != null) {
                scale.close();
            }
        }
    }
}
