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

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Top-level immutable configuration for KTransformers-Java.
 *
 * <p>Factory / host meshes may also pass a flat string map with {@code kt_*} keys
 * via {@link #fromMap(Map)}; unknown keys are ignored so LLaMA-Factory args stay valid.
 */
public final class KtConfig {

    private final KtModelFamily modelFamily;
    private final String modelNameOrPath;
    private final String dtype;
    private final KtInferenceConfig inference;
    private final KtSftConfig sft;
    private final KtQuantConfig quant;
    private final KtMoEConfig moe;
    private final KtCacheConfig cache;
    private final KtDevicePlacement placement;
    private final long hiddenSize;
    private final long intermediateSize;
    private final int vocabSize;
    private final int numLayers;
    private final boolean enableMonitor;
    private final Map<String, String> extra;

    private KtConfig(Builder b) {
        this.modelFamily = Objects.requireNonNull(b.modelFamily, "modelFamily");
        this.modelNameOrPath = b.modelNameOrPath;
        this.dtype = b.dtype != null ? b.dtype : "bf16";
        this.inference = b.inference != null ? b.inference : KtInferenceConfig.defaults();
        this.sft = b.sft != null ? b.sft : KtSftConfig.sftLoraDemo();
        this.quant = b.quant != null ? b.quant : KtQuantConfig.bf16();
        this.moe = b.moe != null ? b.moe : KtMoEConfig.mixtral();
        this.cache = b.cache != null ? b.cache : KtCacheConfig.mini();
        this.placement = b.placement != null ? b.placement : KtDevicePlacement.defaults();
        this.hiddenSize = b.hiddenSize;
        this.intermediateSize = b.intermediateSize;
        this.vocabSize = b.vocabSize;
        this.numLayers = b.numLayers;
        this.enableMonitor = b.enableMonitor;
        this.extra = Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
    }

    public KtModelFamily modelFamily() { return modelFamily; }
    public String modelNameOrPath() { return modelNameOrPath; }
    public String dtype() { return dtype; }
    public KtInferenceConfig inference() { return inference; }
    public KtSftConfig sft() { return sft; }
    public KtQuantConfig quant() { return quant; }
    public KtMoEConfig moe() { return moe; }
    public KtCacheConfig cache() { return cache; }
    public KtDevicePlacement placement() { return placement; }
    public long hiddenSize() { return hiddenSize; }
    public long intermediateSize() { return intermediateSize; }
    public int vocabSize() { return vocabSize; }
    public int numLayers() { return numLayers; }
    public boolean enableMonitor() { return enableMonitor; }
    public Map<String, String> extra() { return extra; }

    public static Builder builder() { return new Builder(); }

    /** Mini end-to-end config for CI / demos (no large weights). */
    public static KtConfig miniDemo() {
        return builder()
                .modelFamily(KtModelFamily.MIXTRAL)
                .modelNameOrPath("kt-mini-moe")
                .hiddenSize(64)
                .intermediateSize(128)
                .vocabSize(256)
                .numLayers(2)
                .moe(KtMoEConfig.builder().numExperts(4).topK(2).sharedExpert(true)
                        .schedule(KtMoEConfig.SchedulePolicy.BALANCED).gpuExpertSlots(2).build())
                .quant(KtQuantConfig.int8AmxLike())
                .cache(KtCacheConfig.mini())
                .inference(KtInferenceConfig.builder()
                        .maxBatch(2).maxSeqLen(128).maxNewTokens(16).concurrency(2)
                        .usePagedAttention(true).build())
                .sft(KtSftConfig.sftLoraDemo())
                .enableMonitor(true)
                .build();
    }

    /**
     * Parse optional {@code kt_*} keys from a host flat map (FactoryArgs style).
     * Missing keys fall back to {@link #miniDemo()} structural defaults with overrides.
     */
    public static KtConfig fromMap(Map<String, ?> raw) {
        Builder b = builder();
        if (raw == null || raw.isEmpty()) {
            return miniDemo();
        }
        Object family = first(raw, "kt_model_family", "model_family");
        if (family != null) {
            b.modelFamily(KtModelFamily.fromString(String.valueOf(family)));
        }
        Object path = first(raw, "kt_model_name_or_path", "model_name_or_path");
        if (path != null) {
            b.modelNameOrPath(String.valueOf(path));
        }
        Object dtype = first(raw, "kt_dtype", "dtype");
        if (dtype != null) b.dtype(String.valueOf(dtype));

        Object hs = first(raw, "kt_hidden_size", "hidden_size");
        if (hs != null) b.hiddenSize(Long.parseLong(String.valueOf(hs)));
        Object is = first(raw, "kt_intermediate_size", "intermediate_size");
        if (is != null) b.intermediateSize(Long.parseLong(String.valueOf(is)));
        Object vs = first(raw, "kt_vocab_size", "vocab_size");
        if (vs != null) b.vocabSize(Integer.parseInt(String.valueOf(vs)));
        Object nl = first(raw, "kt_num_layers", "num_layers", "num_hidden_layers");
        if (nl != null) b.numLayers(Integer.parseInt(String.valueOf(nl)));

        Object ne = first(raw, "kt_num_experts", "num_experts");
        Object tk = first(raw, "kt_top_k", "moe_top_k");
        KtMoEConfig.Builder moeB = KtMoEConfig.builder();
        if (ne != null) moeB.numExperts(Integer.parseInt(String.valueOf(ne)));
        if (tk != null) moeB.topK(Integer.parseInt(String.valueOf(tk)));
        Object sched = first(raw, "kt_expert_schedule", "expert_schedule");
        if (sched != null) {
            moeB.schedule(KtMoEConfig.SchedulePolicy.valueOf(
                    String.valueOf(sched).trim().toUpperCase()));
        }
        b.moe(moeB.build());

        Object bits = first(raw, "kt_weight_bits", "weight_bits");
        if (bits != null) {
            String s = String.valueOf(bits).toLowerCase();
            if (s.contains("4")) b.quant(KtQuantConfig.int4(128));
            else if (s.contains("8") && s.contains("fp")) b.quant(KtQuantConfig.fp8PerChannel());
            else if (s.contains("8")) b.quant(KtQuantConfig.int8AmxLike());
            else b.quant(KtQuantConfig.bf16());
        }

        Object out = first(raw, "kt_output_dir", "output_dir");
        KtSftConfig.Builder sftB = KtSftConfig.builder();
        if (out != null) sftB.outputDir(Path.of(String.valueOf(out)));
        Object stage = first(raw, "kt_stage", "stage");
        if (stage != null) {
            String st = String.valueOf(stage).trim().toUpperCase();
            try {
                sftB.stage(KtSftConfig.Stage.valueOf(st));
            } catch (IllegalArgumentException ignored) {
                // keep default SFT
            }
        }
        Object maxSteps = first(raw, "kt_max_steps", "max_steps");
        if (maxSteps != null) sftB.maxSteps(Integer.parseInt(String.valueOf(maxSteps)));
        Object board = first(raw, "kt_visual_board", "visual_board");
        if (board != null) sftB.visualBoard(Boolean.parseBoolean(String.valueOf(board)));
        b.sft(sftB.build());

        b.enableMonitor(true);
        for (Map.Entry<String, ?> e : raw.entrySet()) {
            if (e.getKey() != null && e.getKey().startsWith("kt_")) {
                b.extra(e.getKey(), e.getValue() == null ? "" : String.valueOf(e.getValue()));
            }
        }
        // Fill structural defaults if host only passed a few keys
        if (b.hiddenSize <= 0) b.hiddenSize(64);
        if (b.intermediateSize <= 0) b.intermediateSize(128);
        if (b.vocabSize <= 0) b.vocabSize(256);
        if (b.numLayers <= 0) b.numLayers(2);
        if (b.cache == null) b.cache(KtCacheConfig.mini());
        if (b.inference == null) b.inference(KtInferenceConfig.defaults());
        if (b.placement == null) b.placement(KtDevicePlacement.defaults());
        if (b.modelFamily == null) b.modelFamily(KtModelFamily.GENERIC);
        return b.build();
    }

    private static Object first(Map<String, ?> raw, String... keys) {
        for (String k : keys) {
            if (raw.containsKey(k) && raw.get(k) != null) {
                return raw.get(k);
            }
        }
        return null;
    }

    public static final class Builder {
        private KtModelFamily modelFamily = KtModelFamily.GENERIC;
        private String modelNameOrPath = "kt-model";
        private String dtype = "bf16";
        private KtInferenceConfig inference;
        private KtSftConfig sft;
        private KtQuantConfig quant;
        private KtMoEConfig moe;
        private KtCacheConfig cache;
        private KtDevicePlacement placement;
        private long hiddenSize = 64;
        private long intermediateSize = 128;
        private int vocabSize = 256;
        private int numLayers = 2;
        private boolean enableMonitor = true;
        private final Map<String, String> extra = new LinkedHashMap<>();

        public Builder modelFamily(KtModelFamily v) { this.modelFamily = v; return this; }
        public Builder modelNameOrPath(String v) { this.modelNameOrPath = v; return this; }
        public Builder dtype(String v) { this.dtype = v; return this; }
        public Builder inference(KtInferenceConfig v) { this.inference = v; return this; }
        public Builder sft(KtSftConfig v) { this.sft = v; return this; }
        public Builder quant(KtQuantConfig v) { this.quant = v; return this; }
        public Builder moe(KtMoEConfig v) { this.moe = v; return this; }
        public Builder cache(KtCacheConfig v) { this.cache = v; return this; }
        public Builder placement(KtDevicePlacement v) { this.placement = v; return this; }
        public Builder hiddenSize(long v) { this.hiddenSize = v; return this; }
        public Builder intermediateSize(long v) { this.intermediateSize = v; return this; }
        public Builder vocabSize(int v) { this.vocabSize = v; return this; }
        public Builder numLayers(int v) { this.numLayers = v; return this; }
        public Builder enableMonitor(boolean v) { this.enableMonitor = v; return this; }
        public Builder extra(String k, String v) { this.extra.put(k, v); return this; }

        public KtConfig build() { return new KtConfig(this); }
    }
}
