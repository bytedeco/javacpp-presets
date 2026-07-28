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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.ScalarOptional;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.abs;
import static org.bytedeco.pytorch.global.torch.clamp;
import static org.bytedeco.pytorch.global.torch.round;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * High-level tensor / model quantization helper (dynamic, static, weight-only,
 * FP8, INT). Named {@code TensorQuantizer} so it does not collide with the
 * generated ATen peer {@code org.bytedeco.pytorch.quantizer.Quantizer}, which
 * {@code relocate_packages.py} always routes into the {@code quantizer} package.
 *
 * <pre>{@code
 * try (TensorQuantizer q = TensorQuantizer.dynamicInt8()) {
 *     Tensor qWeight = q.quantize(weight);
 *     Tensor deq = q.dequantize(qWeight);
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class TensorQuantizer implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public enum Mode {
        DYNAMIC, STATIC, WEIGHT_ONLY, FP8, AWQ, GPTQ
    }

    public enum Scheme {
        PER_TENSOR, PER_CHANNEL, PER_GROUP
    }

    public enum QDType {
        FP32(ScalarType.Float, 32),
        FP16(ScalarType.Half, 16),
        BF16(ScalarType.BFloat16, 16),
        FP8_E4M3FN(ScalarType.Float8_e4m3fn, 8),
        FP8_E5M2(ScalarType.Float8_e5m2, 8),
        INT8(ScalarType.QInt8, 8),
        INT4(ScalarType.Char, 4),
        INT2(ScalarType.Char, 2);

        private final ScalarType scalarType;
        private final int bitWidth;

        QDType(ScalarType scalarType, int bitWidth) {
            this.scalarType = scalarType;
            this.bitWidth = bitWidth;
        }

        public ScalarType toScalarType() { return scalarType; }
        public int getBitWidth() { return bitWidth; }
        public float getCompressionRatio() { return 32.0f / bitWidth; }
    }

    private final Mode mode;
    private final Scheme scheme;
    private final QDType dtype;
    private final float scaleFactor;
    private final List<Tensor> calibrationData = new ArrayList<Tensor>();
    private Tensor scale;
    private boolean calibrated;

    public TensorQuantizer(Mode mode, Scheme scheme, QDType dtype) {
        this(mode, scheme, dtype, 0.5f);
    }

    public TensorQuantizer(Mode mode, Scheme scheme, QDType dtype, float scaleFactor) {
        this.mode = Objects.requireNonNull(mode);
        this.scheme = Objects.requireNonNull(scheme);
        this.dtype = Objects.requireNonNull(dtype);
        this.scaleFactor = scaleFactor;
    }

    public TensorQuantizer(Mode mode, QDType dtype) {
        this(mode, Scheme.PER_TENSOR, dtype);
    }

    public TensorQuantizer(QDType dtype) {
        this(Mode.DYNAMIC, Scheme.PER_TENSOR, dtype);
    }

    public static TensorQuantizer dynamic(QDType dtype) {
        return new TensorQuantizer(Mode.DYNAMIC, Scheme.PER_TENSOR, dtype);
    }

    public static TensorQuantizer dynamicFP8() { return dynamic(QDType.FP8_E4M3FN); }
    public static TensorQuantizer dynamicInt8() { return dynamic(QDType.INT8); }

    public static TensorQuantizer staticQ(QDType dtype) {
        return new TensorQuantizer(Mode.STATIC, Scheme.PER_TENSOR, dtype);
    }

    public static TensorQuantizer staticFP8() { return staticQ(QDType.FP8_E4M3FN); }
    public static TensorQuantizer staticInt8() { return staticQ(QDType.INT8); }

    public static TensorQuantizer weightOnly(QDType dtype) {
        return new TensorQuantizer(Mode.WEIGHT_ONLY, Scheme.PER_CHANNEL, dtype);
    }

    public static TensorQuantizer weightOnlyInt4() { return weightOnly(QDType.INT4); }

    public static TensorQuantizer fp8() { return fp8(QDType.FP8_E4M3FN); }

    public static TensorQuantizer fp8(QDType config) {
        return new TensorQuantizer(Mode.FP8, Scheme.PER_TENSOR, config);
    }

    public static TensorQuantizer awq() { return awq(QDType.INT4); }

    public static TensorQuantizer awq(QDType dtype) {
        return new TensorQuantizer(Mode.AWQ, Scheme.PER_CHANNEL, dtype);
    }

    public static TensorQuantizer gptq() { return gptq(QDType.INT4); }

    public static TensorQuantizer gptq(QDType dtype) {
        return new TensorQuantizer(Mode.GPTQ, Scheme.PER_CHANNEL, dtype);
    }

    public Tensor quantize(Tensor input) {
        if (!calibrated && mode == Mode.STATIC) {
            throw new IllegalStateException("Model must be calibrated before static quantization");
        }
        switch (dtype) {
            case FP8_E4M3FN:
                return quantizeFP8(input, 448.0f);
            case FP8_E5M2:
                return quantizeFP8(input, 57344.0f);
            case INT8:
                return quantizeInt8(input);
            case INT4:
            case INT2:
                return quantizeInt4(input);
            case BF16:
                return input.to(ScalarType.BFloat16);
            case FP16:
                return input.to(ScalarType.Half);
            default:
                return input;
        }
    }

    public Tensor dequantize(Tensor input) {
        switch (dtype) {
            case FP8_E4M3FN:
            case FP8_E5M2:
                return dequantizeFP8(input);
            case INT8:
            case INT4:
            case INT2:
                return dequantizeInt(input);
            default:
                return input;
        }
    }

    private Tensor quantizeFP8(Tensor input, float maxVal) {
        float absMax = abs(input).max().item_float();
        float qScale = absMax > 0 ? maxVal / absMax : 1.0f;
        Tensor scaled = input.mul(new Scalar(qScale));
        Tensor clamped = clamp(
                scaled,
                new ScalarOptional(new Scalar(-maxVal)),
                new ScalarOptional(new Scalar(maxVal)));
        return clamped.to(dtype.toScalarType());
    }

    private Tensor dequantizeFP8(Tensor input) {
        float absMax = abs(input).max().item_float();
        float maxVal = dtype == QDType.FP8_E4M3FN ? 448.0f : 57344.0f;
        float qScale = absMax > 0 ? absMax / maxVal : 1.0f;
        return input.to(ScalarType.Float).mul(new Scalar(qScale));
    }

    private Tensor quantizeInt8(Tensor input) {
        float qScale = computeScale(input);
        Tensor scaled = input.div(new Scalar(qScale));
        Tensor rounded = round(scaled);
        Tensor clamped = clamp(
                rounded,
                new ScalarOptional(new Scalar(-128)),
                new ScalarOptional(new Scalar(127)));
        return clamped.to(ScalarType.QInt8);
    }

    private Tensor quantizeInt4(Tensor input) {
        float qScale = computeScale(input);
        Tensor scaled = input.div(new Scalar(qScale));
        Tensor rounded = round(scaled);
        Tensor clamped = clamp(
                rounded,
                new ScalarOptional(new Scalar(-8)),
                new ScalarOptional(new Scalar(7)));
        return clamped.to(ScalarType.Char);
    }

    private Tensor dequantizeInt(Tensor input) {
        float qScale = computeScale(input);
        return input.to(ScalarType.Float).mul(new Scalar(qScale));
    }

    private float computeScale(Tensor input) {
        float absMax = abs(input).max().item_float();
        return absMax > 0 ? 127.0f / absMax : 1.0f;
    }

    public void addCalibrationData(Tensor data) {
        calibrationData.add(data.clone());
    }

    public void calibrate() {
        if (calibrationData.isEmpty()) {
            throw new IllegalStateException("No calibration data provided");
        }
        float totalAbsMax = 0.0f;
        for (Tensor t : calibrationData) {
            totalAbsMax += abs(t).max().item_float();
        }
        float avg = totalAbsMax / calibrationData.size();
        if (scale != null) {
            scale.close();
        }
        scale = tensor(avg);
        calibrated = true;
    }

    public void setScale(Tensor newScale) {
        if (scale != null) {
            scale.close();
        }
        scale = newScale.clone();
        calibrated = true;
    }

    public QuantizedModel quantizeModel(Module model) {
        Map<String, QuantizedLinear> modules = new LinkedHashMap<String, QuantizedLinear>();
        TensorVector params = model.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor param = params.get(i);
            Tensor q = quantize(param);
            Tensor s = scale != null ? scale.clone() : tensor(computeScale(param));
            modules.put("param_" + modules.size(), new QuantizedLinear(q, s, null));
        }
        return new QuantizedModel(model, modules, this);
    }

    public Mode getMode() { return mode; }
    public Scheme getScheme() { return scheme; }
    public QDType getDtype() { return dtype; }
    public float getScaleFactor() { return scaleFactor; }
    public boolean isCalibrated() { return calibrated; }
    public Tensor getScale() { return scale; }

    @Override
    public void close() {
        for (Tensor t : calibrationData) {
            t.close();
        }
        calibrationData.clear();
        if (scale != null) {
            scale.close();
            scale = null;
        }
    }
}
