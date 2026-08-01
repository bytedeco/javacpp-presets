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
package org.bytedeco.pytorch.llm.ktransformers.config;

/**
 * Quantization configuration for kt-kernel style weight formats.
 *
 * <p>Supports the mathematical layouts commonly used by upstream KT:
 * group-wise INT4/INT8, AMX-like INT8/BF16 paths (reference GEMM), FP8
 * per-channel, and hybrid schemes (e.g. IQ1_S/FP8 hybrid as experimental).
 */
public final class KtQuantConfig {

    public enum WeightBits {
        FP16,
        BF16,
        FP8,
        INT8,
        INT4,
        HYBRID_IQ1S_FP8,
        GPTQ_INT4,
        UNSlotH_1_58,
        UNSLOTH_2_51
    }

    public enum ActDType {
        FP16,
        BF16,
        FP32,
        FP8
    }

    private final WeightBits weightBits;
    private final ActDType actDType;
    private final int groupSize;
    private final boolean amxLike;
    private final boolean avx2Only;
    private final boolean fp8PerChannel;
    private final boolean gptqGpu;
    private final double quantErrorBound;

    private KtQuantConfig(Builder b) {
        this.weightBits = b.weightBits;
        this.actDType = b.actDType;
        if (b.groupSize <= 0) {
            throw new IllegalArgumentException("groupSize must be > 0");
        }
        this.groupSize = b.groupSize;
        this.amxLike = b.amxLike;
        this.avx2Only = b.avx2Only;
        this.fp8PerChannel = b.fp8PerChannel;
        this.gptqGpu = b.gptqGpu;
        this.quantErrorBound = b.quantErrorBound;
    }

    public WeightBits weightBits() { return weightBits; }
    public ActDType actDType() { return actDType; }
    public int groupSize() { return groupSize; }
    public boolean amxLike() { return amxLike; }
    public boolean avx2Only() { return avx2Only; }
//    public boolean fp8PerChannel() { return fp8PerChannel; }
    public boolean gptqGpu() { return gptqGpu; }
    public double quantErrorBound() { return quantErrorBound; }

    public boolean isIntegerWeights() {
        return weightBits == WeightBits.INT4
                || weightBits == WeightBits.INT8
                || weightBits == WeightBits.GPTQ_INT4;
    }

    public int effectiveBits() {
        switch (weightBits) {
            case INT4:
            case GPTQ_INT4:
                return 4;
            case INT8:
                return 8;
            case FP8:
                return 8;
            case HYBRID_IQ1S_FP8:
                return 2; // nominal mixed
            case UNSlotH_1_58:
            case UNSLOTH_2_51:
                return 2;
            case BF16:
            case FP16:
            default:
                return 16;
        }
    }

    public static Builder builder() { return new Builder(); }

    public static KtQuantConfig bf16() {
        return builder().weightBits(WeightBits.BF16).build();
    }

    public static KtQuantConfig int4(int groupSize) {
        return builder().weightBits(WeightBits.INT4).groupSize(groupSize).amxLike(true).build();
    }

    public static KtQuantConfig int8AmxLike() {
        return builder().weightBits(WeightBits.INT8).groupSize(128).amxLike(true).build();
    }

    public static KtQuantConfig fp8PerChannel() {
        return builder().weightBits(WeightBits.FP8).fp8PerChannel(true).groupSize(1).build();
    }

    public static final class Builder {
        private WeightBits weightBits = WeightBits.BF16;
        private ActDType actDType = ActDType.BF16;
        private int groupSize = 128;
        private boolean amxLike = false;
        private boolean avx2Only = false;
        private boolean fp8PerChannel = false;
        private boolean gptqGpu = false;
        private double quantErrorBound = 1e-2;

        public Builder weightBits(WeightBits v) { this.weightBits = v; return this; }
        public Builder actDType(ActDType v) { this.actDType = v; return this; }
        public Builder groupSize(int v) { this.groupSize = v; return this; }
        public Builder amxLike(boolean v) { this.amxLike = v; return this; }
        public Builder avx2Only(boolean v) { this.avx2Only = v; return this; }
        public Builder fp8PerChannel(boolean v) { this.fp8PerChannel = v; return this; }
        public Builder gptqGpu(boolean v) { this.gptqGpu = v; return this; }
        public Builder quantErrorBound(double v) { this.quantErrorBound = v; return this; }

        public KtQuantConfig build() { return new KtQuantConfig(this); }
    }
}
