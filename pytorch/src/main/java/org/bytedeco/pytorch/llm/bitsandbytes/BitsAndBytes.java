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
package org.bytedeco.pytorch.llm.bitsandbytes;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptional;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.llm.quantization.BitsAndBytesConfig;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.linear;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * BitsAndBytes-style 4/8-bit quantization helpers (pure Java / libtorch).
 *
 * <p>API surface mirrors the Python {@code bitsandbytes} package as used by
 * HuggingFace Transformers / PEFT / QLoRA:
 * <ul>
 *   <li>{@link #quantizeNf4} / {@link #dequantizeNf4} — NF4 blockwise (QLoRA)</li>
 *   <li>{@link #quantizeFp4} / {@link #dequantizeFp4} — FP4 E2M1 levels</li>
 *   <li>{@link #quantizeInt8} / {@link #dequantizeInt8} — blockwise INT8</li>
 *   <li>{@link Linear4bit} / {@link Linear8bitLt} — quantized linear wrappers</li>
 *   <li>{@link #quantizeModel} / {@link #prepareModelForKbitTraining} — HF-style helpers</li>
 *   <li>{@link #pack4bit} / {@link #unpack4bit} — nibble packing (2 codes / byte)</li>
 * </ul>
 *
 * <p><b>Note:</b> Not CUDA-kernel identical to the official bitsandbytes package.
 * Numerical path is correct for offline tests / CPU QLoRA pipelines and
 * API-compatible with {@link BitsAndBytesConfig}.
 *
 * <pre>{@code
 * BitsAndBytesConfig cfg = BitsAndBytesConfig.qloraDefaults();
 * QuantizedModel qm = BitsAndBytes.quantizeModel(linears, cfg);
 * BitsAndBytes.prepareModelForKbitTraining(params);
 * Linear4bit layer = BitsAndBytes.linear4bit(dense, cfg);
 * Tensor y = layer.forward(x);
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class BitsAndBytes {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "0.45.0-java";
    public static final int DEFAULT_BLOCKSIZE = 64;

    /**
     * Canonical NF4 quantization levels (QLoRA paper / bitsandbytes).
     * 16 values on [-1, 1] approximating a normal distribution quantile.
     */
    public static final float[] NF4_LEVELS = {
            -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
            -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
            0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
            0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
    };

    /**
     * FP4 E2M1 data-type levels used by bitsandbytes {@code fp4} quant type
     * (sign * 2^{e-1} * (1 + m/2) style discrete set on [-1, 1]).
     */
    public static final float[] FP4_LEVELS = {
            0.0f,
            0.0625f, 0.125f, 0.1875f, 0.25f, 0.375f, 0.5f, 0.75f,
            -0.0f,
            -0.0625f, -0.125f, -0.1875f, -0.25f, -0.375f, -0.5f, -0.75f
    };

    private BitsAndBytes() {}

    // ------------------------------------------------------------------ QuantState

    /**
     * Quantization state for a weight tensor — mirrors Python
     * {@code bitsandbytes.functional.QuantState}.
     */
    public static final class QuantState {
        /** Per-element codes (float-stored indices 0..15 or int8 -128..127). */
        public final Tensor qweight;
        /** Per-block absmax / scale. */
        public final Tensor absmax;
        public final int blocksize;
        public final String quantType;
        public final long[] originalShape;
        public final boolean doubleQuant;
        /**
         * Nested state for double quant of absmax (QLoRA). Non-null when
         * {@link #doubleQuant} is true and nested path was used.
         */
        public final QuantState nested;
        /** Optional code dtype hint: "uint8", "int8", "float". */
        public final String codeDtype;
        /** Optional packed nibble storage (2 x 4-bit codes per byte); may be null. */
        public final byte[] packedCodes;
        /** Second-level scale for double quant of absmax (when nested is null). */
        public final float nestedScale;

        public QuantState(Tensor qweight, Tensor absmax, int blocksize,
                          String quantType, long[] originalShape, boolean doubleQuant) {
            this(qweight, absmax, blocksize, quantType, originalShape, doubleQuant,
                    null, "float", null, 1f);
        }

        public QuantState(Tensor qweight, Tensor absmax, int blocksize,
                          String quantType, long[] originalShape, boolean doubleQuant,
                          QuantState nested, String codeDtype, byte[] packedCodes,
                          float nestedScale) {
            this.qweight = qweight;
            this.absmax = absmax;
            this.blocksize = blocksize;
            this.quantType = quantType;
            this.originalShape = originalShape;
            this.doubleQuant = doubleQuant;
            this.nested = nested;
            this.codeDtype = codeDtype == null ? "float" : codeDtype;
            this.packedCodes = packedCodes;
            this.nestedScale = nestedScale;
        }

        public long numel() {
            long n = 1;
            for (long s : originalShape) n *= s;
            return n;
        }

        public int numBlocks() {
            return (int) ((numel() + blocksize - 1) / blocksize);
        }

        /** Estimated on-disk / in-memory bytes if codes were packed. */
        public long memoryBytes() {
            return estimateMemoryBytes(numel(), quantType, blocksize, doubleQuant);
        }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("quant_type", quantType);
            m.put("blocksize", blocksize);
            m.put("double_quant", doubleQuant);
            m.put("shape", originalShape.clone());
            m.put("numel", numel());
            m.put("num_blocks", numBlocks());
            m.put("memory_bytes", memoryBytes());
            m.put("code_dtype", codeDtype);
            m.put("has_nested", nested != null);
            m.put("has_packed", packedCodes != null);
            m.put("nested_scale", nestedScale);
            return m;
        }

        @Override
        public String toString() {
            return "QuantState{type=" + quantType + ", shape=" + java.util.Arrays.toString(originalShape)
                    + ", blocksize=" + blocksize + ", doubleQuant=" + doubleQuant
                    + ", mem=" + memoryBytes() + "B}";
        }
    }

    // ------------------------------------------------------------------ INT8

    public static QuantState quantizeInt8(Tensor weight, int blocksize) {
        Objects.requireNonNull(weight, "weight");
        if (blocksize <= 0) blocksize = DEFAULT_BLOCKSIZE;
        Tensor w = weight.reshape(-1).to(ScalarType.Float).contiguous();
        long n = w.numel();
        int blocks = (int) ((n + blocksize - 1) / blocksize);
        float[] data = toFloatArray(w);
        float[] codes = new float[(int) n];
        float[] scales = new float[blocks];
        for (int b = 0; b < blocks; b++) {
            int start = b * blocksize;
            int end = Math.min((int) n, start + blocksize);
            float amax = 0f;
            for (int i = start; i < end; i++) {
                float a = Math.abs(data[i]);
                if (a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            scales[b] = amax / 127f;
            for (int i = start; i < end; i++) {
                int q = Math.round(data[i] / scales[b]);
                q = Math.max(-128, Math.min(127, q));
                codes[i] = q;
            }
        }
        return new QuantState(tensor(codes), tensor(scales), blocksize, "int8",
                shapeOf(weight), false, null, "int8", null, 1f);
    }

    public static Tensor dequantizeInt8(QuantState state) {
        float[] codes = toFloatArray(state.qweight);
        float[] scales = resolveScales(state);
        float[] out = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            int b = Math.min(scales.length - 1, i / state.blocksize);
            out[i] = codes[i] * scales[b];
        }
        return tensor(out).reshape(state.originalShape);
    }

    // ------------------------------------------------------------------ NF4

    public static QuantState quantizeNf4(Tensor weight, int blocksize) {
        return quantizeNf4(weight, blocksize, false);
    }

    public static QuantState quantizeNf4(Tensor weight, int blocksize, boolean doubleQuant) {
        Objects.requireNonNull(weight, "weight");
        if (blocksize <= 0) blocksize = DEFAULT_BLOCKSIZE;
        Tensor w = weight.reshape(-1).to(ScalarType.Float).contiguous();
        long n = w.numel();
        int blocks = (int) ((n + blocksize - 1) / blocksize);
        float[] data = toFloatArray(w);
        float[] codes = new float[(int) n];
        float[] scales = new float[blocks];
        for (int b = 0; b < blocks; b++) {
            int start = b * blocksize;
            int end = Math.min((int) n, start + blocksize);
            float amax = 0f;
            for (int i = start; i < end; i++) {
                float a = Math.abs(data[i]);
                if (a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            scales[b] = amax;
            for (int i = start; i < end; i++) {
                codes[i] = nearestLevelIndex(data[i] / amax, NF4_LEVELS);
            }
        }
        QuantState nested = null;
        float nestedScale = 1f;
        if (doubleQuant) {
            // Quantize absmax with int8 (nested) — store both nested state and restored scales
            float smax = 0f;
            for (float s : scales) if (Math.abs(s) > smax) smax = Math.abs(s);
            if (smax < 1e-12f) smax = 1e-12f;
            nestedScale = smax / 127f;
            float[] nestedCodes = new float[scales.length];
            for (int i = 0; i < scales.length; i++) {
                int q = Math.round(scales[i] / nestedScale);
                q = Math.max(-128, Math.min(127, q));
                nestedCodes[i] = q;
                scales[i] = q * nestedScale; // restore quantized scale
            }
            nested = new QuantState(tensor(nestedCodes), tensor(new float[]{nestedScale}),
                    scales.length, "int8", new long[]{scales.length}, false,
                    null, "int8", null, nestedScale);
        }
        byte[] packed = pack4bit(codes);
        return new QuantState(tensor(codes), tensor(scales), blocksize, "nf4",
                shapeOf(weight), doubleQuant, nested, "uint8", packed, nestedScale);
    }

    public static Tensor dequantizeNf4(QuantState state) {
        float[] codes = toFloatArray(state.qweight);
        float[] scales = resolveScales(state);
        float[] out = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            int b = Math.min(scales.length - 1, i / state.blocksize);
            int idx = Math.max(0, Math.min(15, Math.round(codes[i])));
            out[i] = NF4_LEVELS[idx] * scales[b];
        }
        return tensor(out).reshape(state.originalShape);
    }

    // ------------------------------------------------------------------ FP4

    public static QuantState quantizeFp4(Tensor weight, int blocksize) {
        return quantizeFp4(weight, blocksize, false);
    }

    public static QuantState quantizeFp4(Tensor weight, int blocksize, boolean doubleQuant) {
        Objects.requireNonNull(weight, "weight");
        if (blocksize <= 0) blocksize = DEFAULT_BLOCKSIZE;
        Tensor w = weight.reshape(-1).to(ScalarType.Float).contiguous();
        long n = w.numel();
        int blocks = (int) ((n + blocksize - 1) / blocksize);
        float[] data = toFloatArray(w);
        float[] codes = new float[(int) n];
        float[] scales = new float[blocks];
        for (int b = 0; b < blocks; b++) {
            int start = b * blocksize;
            int end = Math.min((int) n, start + blocksize);
            float amax = 0f;
            for (int i = start; i < end; i++) {
                float a = Math.abs(data[i]);
                if (a > amax) amax = a;
            }
            if (amax < 1e-12f) amax = 1e-12f;
            scales[b] = amax;
            for (int i = start; i < end; i++) {
                codes[i] = nearestLevelIndex(data[i] / amax, FP4_LEVELS);
            }
        }
        QuantState nested = null;
        float nestedScale = 1f;
        if (doubleQuant) {
            float smax = 0f;
            for (float s : scales) if (Math.abs(s) > smax) smax = Math.abs(s);
            if (smax < 1e-12f) smax = 1e-12f;
            nestedScale = smax / 127f;
            float[] nestedCodes = new float[scales.length];
            for (int i = 0; i < scales.length; i++) {
                int q = Math.round(scales[i] / nestedScale);
                q = Math.max(-128, Math.min(127, q));
                nestedCodes[i] = q;
                scales[i] = q * nestedScale;
            }
            nested = new QuantState(tensor(nestedCodes), tensor(new float[]{nestedScale}),
                    scales.length, "int8", new long[]{scales.length}, false,
                    null, "int8", null, nestedScale);
        }
        byte[] packed = pack4bit(codes);
        return new QuantState(tensor(codes), tensor(scales), blocksize, "fp4",
                shapeOf(weight), doubleQuant, nested, "uint8", packed, nestedScale);
    }

    public static Tensor dequantizeFp4(QuantState state) {
        float[] codes = toFloatArray(state.qweight);
        float[] scales = resolveScales(state);
        float[] out = new float[codes.length];
        for (int i = 0; i < codes.length; i++) {
            int b = Math.min(scales.length - 1, i / state.blocksize);
            int idx = Math.max(0, Math.min(15, Math.round(codes[i])));
            out[i] = FP4_LEVELS[idx] * scales[b];
        }
        return tensor(out).reshape(state.originalShape);
    }

    // ------------------------------------------------------------------ Generic

    public static Tensor dequantize(QuantState state) {
        return switch (state.quantType.toLowerCase(Locale.ROOT)) {
            case "nf4" -> dequantizeNf4(state);
            case "fp4" -> dequantizeFp4(state);
            case "int8" -> dequantizeInt8(state);
            default -> throw new IllegalArgumentException("Unknown quantType: " + state.quantType);
        };
    }

    public static QuantState quantize(Tensor weight, BitsAndBytesConfig cfg) {
        int bs = cfg == null ? DEFAULT_BLOCKSIZE : cfg.getBlocksize();
        return quantize(weight, cfg, bs);
    }

    public static QuantState quantize(Tensor weight, BitsAndBytesConfig cfg, int blocksize) {
        if (cfg != null && cfg.isLoadIn8Bit()) {
            return quantizeInt8(weight, blocksize);
        }
        String t = cfg == null ? "nf4" : cfg.getBnb4BitQuantType();
        boolean dq = cfg != null && cfg.isBnb4BitUseDoubleQuant();
        if ("fp4".equalsIgnoreCase(t)) {
            return quantizeFp4(weight, blocksize, dq);
        }
        return quantizeNf4(weight, blocksize, dq);
    }

    /**
     * Quantize then immediately dequantize — useful for measuring reconstruction error.
     */
    public static Tensor quantizeDequantize(Tensor weight, BitsAndBytesConfig cfg) {
        return dequantize(quantize(weight, cfg));
    }

    // ------------------------------------------------------------------ Pack / unpack 4-bit

    /** Pack float codes in [0,15] into nibbles (2 codes per byte). */
    public static byte[] pack4bit(float[] codes) {
        int n = codes.length;
        byte[] out = new byte[(n + 1) / 2];
        for (int i = 0; i < n; i += 2) {
            int lo = Math.max(0, Math.min(15, Math.round(codes[i])));
            int hi = (i + 1 < n) ? Math.max(0, Math.min(15, Math.round(codes[i + 1]))) : 0;
            out[i / 2] = (byte) ((hi << 4) | lo);
        }
        return out;
    }

    /** Unpack nibble-packed codes back to float indices. */
    public static float[] unpack4bit(byte[] packed, int numel) {
        float[] out = new float[numel];
        for (int i = 0; i < numel; i++) {
            int b = packed[i / 2] & 0xFF;
            out[i] = (i % 2 == 0) ? (b & 0x0F) : ((b >> 4) & 0x0F);
        }
        return out;
    }

    // ------------------------------------------------------------------ Memory estimates

    public static long estimateMemoryBytes(long numel, String quantType) {
        return estimateMemoryBytes(numel, quantType, DEFAULT_BLOCKSIZE, false);
    }

    public static long estimateMemoryBytes(long numel, String quantType, int blocksize, boolean doubleQuant) {
        if (blocksize <= 0) blocksize = DEFAULT_BLOCKSIZE;
        long blocks = (numel + blocksize - 1) / blocksize;
        return switch (quantType == null ? "fp32" : quantType.toLowerCase(Locale.ROOT)) {
            case "nf4", "fp4" -> {
                long codeBytes = (numel + 1) / 2; // packed nibbles
                long scaleBytes = doubleQuant
                        ? blocks /* int8 nested codes */ + 4 /* nested scale */
                        : blocks * 4L; // fp32 absmax
                yield codeBytes + scaleBytes;
            }
            case "int8" -> numel + blocks * 4L;
            case "fp16", "float16", "half" -> numel * 2L;
            case "bf16", "bfloat16" -> numel * 2L;
            default -> numel * 4L;
        };
    }

    public static double compressionRatio(long numel, String quantType, boolean doubleQuant) {
        long fp32 = numel * 4L;
        long q = estimateMemoryBytes(numel, quantType, DEFAULT_BLOCKSIZE, doubleQuant);
        return q == 0 ? 0.0 : (double) fp32 / (double) q;
    }

    // ------------------------------------------------------------------ Linear8bitLt

    /**
     * 8-bit linear layer (bitsandbytes {@code Linear8bitLt} API surface).
     * Forward dequantizes weight then runs standard matmul.
     */
    public static final class Linear8bitLt implements AutoCloseable {
        private final QuantState weightState;
        private final Tensor bias;
        private final long inFeatures;
        private final long outFeatures;
        private final boolean hasFp16Weights;
        private final double threshold;
        private Tensor cachedWeight; // optional dequant cache

        public Linear8bitLt(QuantState weightState, Tensor bias, long inFeatures, long outFeatures) {
            this(weightState, bias, inFeatures, outFeatures, false, 6.0);
        }

        public Linear8bitLt(QuantState weightState, Tensor bias, long inFeatures, long outFeatures,
                            boolean hasFp16Weights, double threshold) {
            this.weightState = Objects.requireNonNull(weightState, "weightState");
            this.bias = bias;
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            this.hasFp16Weights = hasFp16Weights;
            this.threshold = threshold;
        }

        public Tensor forward(Tensor input) {
            Tensor w = weight();
            if (bias == null) return linear(input, w);
            return linear(input, w, new TensorOptional(bias));
        }

        /** Dequantized weight (cached after first call). */
        public Tensor weight() {
            if (cachedWeight == null || !cachedWeight.defined()) {
                cachedWeight = dequantizeInt8(weightState);
            }
            return cachedWeight;
        }

        public QuantState weightState() { return weightState; }
        public Tensor bias() { return bias; }
        public long inFeatures() { return inFeatures; }
        public long outFeatures() { return outFeatures; }
        public boolean hasFp16Weights() { return hasFp16Weights; }
        public double threshold() { return threshold; }

        /** Replace underlying dense weight in-place with dequantized values (for Module graphs). */
        public void materializeInto(LinearImpl dense) {
            Tensor w = weight();
            try (org.bytedeco.pytorch.NoGradGuard guard = new org.bytedeco.pytorch.NoGradGuard()) {
                // Leaf params with requires_grad cannot be mutated in-place without no_grad.
                dense.weight().requires_grad_(false);
                dense.weight().copy_(w);
                if (bias != null && dense.bias() != null && dense.bias().defined()) {
                    dense.bias().requires_grad_(false);
                    dense.bias().copy_(bias);
                }
            }
        }

        @Override
        public void close() {
            if (cachedWeight != null) {
                try { cachedWeight.close(); } catch (Exception ignored) {}
                cachedWeight = null;
            }
        }
    }

    public static Linear8bitLt linear8bit(LinearImpl dense) {
        return linear8bit(dense, null);
    }

    public static Linear8bitLt linear8bit(LinearImpl dense, BitsAndBytesConfig cfg) {
        Objects.requireNonNull(dense, "dense");
        Tensor w = dense.weight();
        Tensor b = safeBias(dense);
        int bs = cfg == null ? DEFAULT_BLOCKSIZE : cfg.getBlocksize();
        QuantState qs = quantizeInt8(w, bs);
        boolean hasFp16 = cfg != null && cfg.isLlmInt8HasFp16Weight();
        double thr = cfg == null ? 6.0 : cfg.getLlmInt8Threshold();
        return new Linear8bitLt(qs, b, w.size(1), w.size(0), hasFp16, thr);
    }

    public static Linear8bitLt linear8bit(long inFeatures, long outFeatures) {
        return linear8bit(new LinearImpl(inFeatures, outFeatures));
    }

    // ------------------------------------------------------------------ Linear4bit

    /**
     * 4-bit linear layer (bitsandbytes {@code Linear4bit} / {@code Params4bit} API).
     */
    public static final class Linear4bit implements AutoCloseable {
        private final QuantState weightState;
        private final Tensor bias;
        private final long inFeatures;
        private final long outFeatures;
        private final String computeDtype;
        private final boolean quantStorage;
        private Tensor cachedWeight;

        public Linear4bit(QuantState weightState, Tensor bias,
                          long inFeatures, long outFeatures, String computeDtype) {
            this.weightState = Objects.requireNonNull(weightState, "weightState");
            this.bias = bias;
            this.inFeatures = inFeatures;
            this.outFeatures = outFeatures;
            this.computeDtype = computeDtype == null ? "float32" : computeDtype;
            this.quantStorage = weightState.packedCodes != null;
        }

        public Tensor forward(Tensor input) {
            Tensor w = weight();
            if (bias == null) return linear(input, w);
            return linear(input, w, new TensorOptional(bias));
        }

        public Tensor weight() {
            if (cachedWeight == null || !cachedWeight.defined()) {
                cachedWeight = dequantize(weightState);
            }
            return cachedWeight;
        }

        public QuantState weightState() { return weightState; }
        public Tensor bias() { return bias; }
        public String computeDtype() { return computeDtype; }
        public long inFeatures() { return inFeatures; }
        public long outFeatures() { return outFeatures; }
        public boolean quantStorage() { return quantStorage; }

        public void materializeInto(LinearImpl dense) {
            Tensor w = weight();
            try (org.bytedeco.pytorch.NoGradGuard guard = new org.bytedeco.pytorch.NoGradGuard()) {
                dense.weight().requires_grad_(false);
                dense.weight().copy_(w);
                if (bias != null && dense.bias() != null && dense.bias().defined()) {
                    dense.bias().requires_grad_(false);
                    dense.bias().copy_(bias);
                }
            }
        }

        public Map<String, Object> stats() {
            Map<String, Object> m = new LinkedHashMap<>(weightState.toMap());
            m.put("in_features", inFeatures);
            m.put("out_features", outFeatures);
            m.put("compute_dtype", computeDtype);
            m.put("quant_storage", quantStorage);
            return m;
        }

        @Override
        public void close() {
            if (cachedWeight != null) {
                try { cachedWeight.close(); } catch (Exception ignored) {}
                cachedWeight = null;
            }
        }
    }

    public static Linear4bit linear4bit(LinearImpl dense, BitsAndBytesConfig cfg) {
        Objects.requireNonNull(dense, "dense");
        BitsAndBytesConfig c = cfg == null
                ? BitsAndBytesConfig.builder().loadIn4Bit(true).build()
                : cfg;
        Tensor w = dense.weight();
        Tensor b = safeBias(dense);
        QuantState qs = quantize(w, c, c.getBlocksize());
        return new Linear4bit(qs, b, w.size(1), w.size(0), c.getBnb4BitComputeDtype());
    }

    public static Linear4bit linear4bit(long inFeatures, long outFeatures, BitsAndBytesConfig cfg) {
        return linear4bit(new LinearImpl(inFeatures, outFeatures), cfg);
    }

    public static Linear4bit linear4bit(long inFeatures, long outFeatures) {
        return linear4bit(inFeatures, outFeatures,
                BitsAndBytesConfig.builder().loadIn4Bit(true).build());
    }

    // ------------------------------------------------------------------ Model-level helpers

    /**
     * Result of quantizing a collection of named linears (HF-style
     * {@code replace_with_bnb_linear} bookkeeping without Module-tree rewrite).
     */
    public static final class QuantizedModel implements AutoCloseable {
        private final Map<String, Object> layers; // name -> Linear4bit | Linear8bitLt
        private final Map<String, QuantState> states;
        private final BitsAndBytesConfig config;
        private final long totalParams;
        private final long quantMemoryBytes;

        public QuantizedModel(Map<String, Object> layers, Map<String, QuantState> states,
                              BitsAndBytesConfig config, long totalParams, long quantMemoryBytes) {
            this.layers = Collections.unmodifiableMap(new LinkedHashMap<>(layers));
            this.states = Collections.unmodifiableMap(new LinkedHashMap<>(states));
            this.config = config;
            this.totalParams = totalParams;
            this.quantMemoryBytes = quantMemoryBytes;
        }

        public Map<String, Object> layers() { return layers; }
        public Map<String, QuantState> states() { return states; }
        public BitsAndBytesConfig config() { return config; }
        public long totalParams() { return totalParams; }
        public long quantMemoryBytes() { return quantMemoryBytes; }
        public int size() { return layers.size(); }

        public Object get(String name) { return layers.get(name); }
        public QuantState state(String name) { return states.get(name); }

        public Linear4bit as4bit(String name) {
            Object o = layers.get(name);
            return o instanceof Linear4bit l ? l : null;
        }

        public Linear8bitLt as8bit(String name) {
            Object o = layers.get(name);
            return o instanceof Linear8bitLt l ? l : null;
        }

        /**
         * Materialize all dequantized weights back into the provided dense linears
         * (freeze base). Keys must match.
         */
        public int materializeInto(Map<String, LinearImpl> dense) {
            int n = 0;
            for (Map.Entry<String, Object> e : layers.entrySet()) {
                LinearImpl d = dense.get(e.getKey());
                if (d == null) continue;
                Object layer = e.getValue();
                if (layer instanceof Linear4bit l4) {
                    l4.materializeInto(d);
                    n++;
                } else if (layer instanceof Linear8bitLt l8) {
                    l8.materializeInto(d);
                    n++;
                }
            }
            return n;
        }

        public Map<String, Object> stats() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("num_layers", layers.size());
            m.put("total_params", totalParams);
            m.put("quant_memory_bytes", quantMemoryBytes);
            m.put("fp32_memory_bytes", totalParams * 4L);
            m.put("compression_ratio", totalParams == 0 ? 0.0
                    : (double) (totalParams * 4L) / (double) Math.max(1, quantMemoryBytes));
            m.put("config", config == null ? null : config.toString());
            m.put("load_in_4bit", config != null && config.isLoadIn4Bit());
            m.put("load_in_8bit", config != null && config.isLoadIn8Bit());
            return m;
        }

        @Override
        public void close() {
            for (Object o : layers.values()) {
                if (o instanceof AutoCloseable c) {
                    try { c.close(); } catch (Exception ignored) {}
                }
            }
        }
    }

    /**
     * Quantize a map of named {@link LinearImpl}s according to {@code cfg}.
     * Skips modules listed in {@link BitsAndBytesConfig#shouldSkipModule(String)}.
     */
    public static QuantizedModel quantizeModel(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        Objects.requireNonNull(linears, "linears");
        BitsAndBytesConfig c = cfg == null ? BitsAndBytesConfig.qloraDefaults() : cfg;
        Map<String, Object> layers = new LinkedHashMap<>();
        Map<String, QuantState> states = new LinkedHashMap<>();
        long total = 0;
        long mem = 0;
        for (Map.Entry<String, LinearImpl> e : linears.entrySet()) {
            String name = e.getKey();
            LinearImpl lin = e.getValue();
            if (lin == null || lin.weight() == null || !lin.weight().defined()) continue;
            if (c.shouldSkipModule(name)) continue;
            total += lin.weight().numel();
            if (c.isLoadIn8Bit()) {
                Linear8bitLt l8 = linear8bit(lin, c);
                layers.put(name, l8);
                states.put(name, l8.weightState());
                mem += l8.weightState().memoryBytes();
            } else {
                Linear4bit l4 = linear4bit(lin, c);
                layers.put(name, l4);
                states.put(name, l4.weightState());
                mem += l4.weightState().memoryBytes();
            }
        }
        return new QuantizedModel(layers, states, c, total, mem);
    }

    /**
     * Quantize then materialize dequantized (frozen) weights back into the same
     * linears — the practical path for Java Module graphs that cannot freely
     * swap submodule types. Returns the {@link QuantizedModel} bookkeeping object.
     */
    public static QuantizedModel replaceLinearWithBnb(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        QuantizedModel qm = quantizeModel(linears, cfg);
        qm.materializeInto(linears);
        return qm;
    }

    /**
     * Freeze all parameters in {@code params} (set {@code requires_grad=False}).
     * Mirrors HF {@code prepare_model_for_kbit_training} base-weight freeze step.
     *
     * @return number of tensors frozen
     */
    public static int prepareModelForKbitTraining(TensorVector params) {
        if (params == null) return 0;
        int n = 0;
        for (long i = 0, m = params.size(); i < m; i++) {
            Tensor p = params.get((int) i);
            if (p != null && !p.isNull() && p.defined()) {
                p.requires_grad_(false);
                n++;
            }
        }
        return n;
    }

    /** Freeze every parameter of a Module (base model for QLoRA). */
    public static int prepareModelForKbitTraining(Module model) {
        if (model == null) return 0;
        try {
            return prepareModelForKbitTraining(model.parameters());
        } catch (Exception e) {
            return 0;
        }
    }

    /**
     * Freeze base linears then leave them ready for LoRA injection
     * (quant-dequant materialize + freeze).
     */
    public static QuantizedModel prepareForQLoRA(Map<String, LinearImpl> linears, BitsAndBytesConfig cfg) {
        BitsAndBytesConfig c = cfg == null ? BitsAndBytesConfig.qloraDefaults() : cfg;
        QuantizedModel qm = replaceLinearWithBnb(linears, c);
        for (LinearImpl lin : linears.values()) {
            if (lin == null) continue;
            try {
                lin.weight().requires_grad_(false);
                Tensor b = safeBias(lin);
                if (b != null) b.requires_grad_(false);
            } catch (Exception ignored) {}
        }
        return qm;
    }

    /**
     * Mean absolute reconstruction error after quantize→dequantize.
     */
    public static double reconstructionMae(Tensor weight, BitsAndBytesConfig cfg) {
        Tensor restored = quantizeDequantize(weight, cfg);
        try {
            float[] a = toFloatArray(weight.reshape(-1).to(ScalarType.Float));
            float[] b = toFloatArray(restored.reshape(-1).to(ScalarType.Float));
            double sum = 0;
            int n = Math.min(a.length, b.length);
            for (int i = 0; i < n; i++) sum += Math.abs(a[i] - b[i]);
            return n == 0 ? 0.0 : sum / n;
        } finally {
            try { restored.close(); } catch (Exception ignored) {}
        }
    }

    /**
     * Cosine similarity between original and quant→dequant weight (1 = perfect).
     */
    public static double reconstructionCosine(Tensor weight, BitsAndBytesConfig cfg) {
        Tensor restored = quantizeDequantize(weight, cfg);
        try {
            float[] a = toFloatArray(weight.reshape(-1).to(ScalarType.Float));
            float[] b = toFloatArray(restored.reshape(-1).to(ScalarType.Float));
            int n = Math.min(a.length, b.length);
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++) {
                dot += a[i] * b[i];
                na += a[i] * a[i];
                nb += b[i] * b[i];
            }
            if (na < 1e-24 || nb < 1e-24) return 0.0;
            return dot / (Math.sqrt(na) * Math.sqrt(nb));
        } finally {
            try { restored.close(); } catch (Exception ignored) {}
        }
    }

    // ------------------------------------------------------------------ Collect linears from common names

    /**
     * Build a name→LinearImpl map from an explicit list (caller enumerates).
     * Convenience for QLoRA target modules.
     */
    public static Map<String, LinearImpl> linearMap(String[] names, LinearImpl[] linears) {
        Map<String, LinearImpl> m = new LinkedHashMap<>();
        if (names == null || linears == null) return m;
        int n = Math.min(names.length, linears.length);
        for (int i = 0; i < n; i++) {
            if (names[i] != null && linears[i] != null) m.put(names[i], linears[i]);
        }
        return m;
    }

    public static List<String> defaultSkipModules() {
        return List.of("lm_head", "embed_tokens", "wte", "wpe", "embed_out");
    }

    // ------------------------------------------------------------------ Internals

    private static float[] resolveScales(QuantState state) {
        if (state.nested != null && state.doubleQuant) {
            // Reconstruct scales from nested int8 codes * nested scale
            float[] codes = toFloatArray(state.nested.qweight);
            float[] nestedScaleArr = toFloatArray(state.nested.absmax);
            float ns = nestedScaleArr.length > 0 ? nestedScaleArr[0] : state.nestedScale;
            float[] scales = new float[codes.length];
            for (int i = 0; i < codes.length; i++) scales[i] = codes[i] * ns;
            return scales;
        }
        return toFloatArray(state.absmax);
    }

    private static Tensor safeBias(LinearImpl dense) {
        try {
            Tensor bb = dense.bias();
            if (bb != null && !bb.isNull() && bb.defined()) return bb;
        } catch (Exception ignored) {
        }
        return null;
    }

    private static int nearestLevelIndex(float x, float[] levels) {
        int best = 0;
        float bestDist = Float.MAX_VALUE;
        for (int i = 0; i < levels.length; i++) {
            float d = Math.abs(x - levels[i]);
            if (d < bestDist) {
                bestDist = d;
                best = i;
            }
        }
        return best;
    }

    private static long[] shapeOf(Tensor weight) {
        long[] shape = new long[(int) weight.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = weight.size(i);
        return shape;
    }

    private static float[] toFloatArray(Tensor t) {
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
