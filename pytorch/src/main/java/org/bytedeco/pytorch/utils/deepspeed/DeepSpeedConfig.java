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
package org.bytedeco.pytorch.utils.deepspeed;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * DeepSpeed configuration (Java port of common {@code ds_config.json} keys).
 *
 * <p>Covers ZeRO stages 0–3, CPU/NVMe offload flags, gradient accumulation /
 * clipping, fp16/bf16, communication buckets, activation checkpointing toggles.
 * Native CUDA kernel fusions are represented as configuration bookkeeping only.
 */
public final class DeepSpeedConfig {

    private final int zeroStage;
    private final boolean cpuOffload;
    private final boolean nvmeOffload;
    private final boolean offloadParam;
    private final boolean offloadOptimizer;
    private final int gradientAccumulationSteps;
    private final double gradientClip;
    private final boolean overlapComm;
    private final int reduceBucketSize;
    private final int allgatherBucketSize;
    private final String precision; // fp32 | fp16 | bf16
    private final boolean wallClockBreakdown;
    private final boolean activationCheckpointing;
    private final int trainMicroBatchSizePerGpu;
    private final int trainBatchSize;
    private final boolean contiguousGradients;
    private final boolean reduceScatter;
    private final String zeroAllowUntestedOptimizer;
    private final Map<String, Object> extra;

    private DeepSpeedConfig(Builder b) {
        this.zeroStage = b.zeroStage;
        this.cpuOffload = b.cpuOffload;
        this.nvmeOffload = b.nvmeOffload;
        this.offloadParam = b.offloadParam || b.cpuOffload || b.nvmeOffload;
        this.offloadOptimizer = b.offloadOptimizer || b.cpuOffload;
        this.gradientAccumulationSteps = Math.max(1, b.gradientAccumulationSteps);
        this.gradientClip = b.gradientClip;
        this.overlapComm = b.overlapComm;
        this.reduceBucketSize = b.reduceBucketSize;
        this.allgatherBucketSize = b.allgatherBucketSize;
        this.precision = b.precision == null ? "fp32" : b.precision;
        this.wallClockBreakdown = b.wallClockBreakdown;
        this.activationCheckpointing = b.activationCheckpointing;
        this.trainMicroBatchSizePerGpu = Math.max(1, b.trainMicroBatchSizePerGpu);
        this.trainBatchSize = b.trainBatchSize > 0 ? b.trainBatchSize
                : trainMicroBatchSizePerGpu * gradientAccumulationSteps;
        this.contiguousGradients = b.contiguousGradients;
        this.reduceScatter = b.reduceScatter;
        this.zeroAllowUntestedOptimizer = b.zeroAllowUntestedOptimizer;
        this.extra = Map.copyOf(b.extra);
        if (zeroStage < 0 || zeroStage > 3) {
            throw new IllegalArgumentException("zeroStage must be in [0,3], got " + zeroStage);
        }
    }

    public static Builder builder() { return new Builder(); }

    public static DeepSpeedConfig defaults() { return builder().build(); }

    /** Parse a flat / nested-ish config map (Python ds_config subset). */
    @SuppressWarnings("unchecked")
    public static DeepSpeedConfig fromMap(Map<String, Object> m) {
        Objects.requireNonNull(m, "config map");
        Builder b = builder();
        Object zero = m.get("zero_optimization");
        if (zero instanceof Map) {
            Map<String, Object> z = (Map<String, Object>) zero;
            if (z.get("stage") instanceof Number) b.zeroStage(((Number) z.get("stage")).intValue());
            if (z.containsKey("overlap_comm")) b.overlapComm(asBool(z.get("overlap_comm")));
            if (z.containsKey("contiguous_gradients")) b.contiguousGradients(asBool(z.get("contiguous_gradients")));
            if (z.containsKey("reduce_scatter")) b.reduceScatter(asBool(z.get("reduce_scatter")));
            if (z.get("reduce_bucket_size") instanceof Number)
                b.reduceBucketSize(((Number) z.get("reduce_bucket_size")).intValue());
            if (z.get("allgather_bucket_size") instanceof Number)
                b.allgatherBucketSize(((Number) z.get("allgather_bucket_size")).intValue());
            Object offOpt = z.get("offload_optimizer");
            if (offOpt instanceof Map) {
                Object device = ((Map<?, ?>) offOpt).get("device");
                if (device != null && !"none".equalsIgnoreCase(String.valueOf(device))) {
                    b.offloadOptimizer(true);
                    if ("cpu".equalsIgnoreCase(String.valueOf(device))) b.cpuOffload(true);
                    if ("nvme".equalsIgnoreCase(String.valueOf(device))) b.nvmeOffload(true);
                }
            }
            Object offParam = z.get("offload_param");
            if (offParam instanceof Map) {
                Object device = ((Map<?, ?>) offParam).get("device");
                if (device != null && !"none".equalsIgnoreCase(String.valueOf(device))) {
                    b.offloadParam(true);
                    if ("cpu".equalsIgnoreCase(String.valueOf(device))) b.cpuOffload(true);
                    if ("nvme".equalsIgnoreCase(String.valueOf(device))) b.nvmeOffload(true);
                }
            }
        }
        // flat keys
        putInt(m, "zero_optimization.stage", b::zeroStage);
        putInt(m, "gradient_accumulation_steps", b::gradientAccumulationSteps);
        if (m.get("gradient_clipping") instanceof Number)
            b.gradientClip(((Number) m.get("gradient_clipping")).doubleValue());
        if (m.get("gradient_clip") instanceof Number)
            b.gradientClip(((Number) m.get("gradient_clip")).doubleValue());
        putInt(m, "train_micro_batch_size_per_gpu", b::trainMicroBatchSizePerGpu);
        putInt(m, "train_batch_size", b::trainBatchSize);
        if (asBool(m.get("fp16.enabled")) || asBool(nested(m, "fp16", "enabled"))) b.precision("fp16");
        if (asBool(m.get("bf16.enabled")) || asBool(nested(m, "bf16", "enabled"))) b.precision("bf16");
        if (m.get("wall_clock_breakdown") != null) b.wallClockBreakdown(asBool(m.get("wall_clock_breakdown")));
        Object act = m.get("activation_checkpointing");
        if (act instanceof Map && asBool(((Map<?, ?>) act).get("partition_activations"))) {
            b.activationCheckpointing(true);
        }
        if (asBool(m.get("activation_checkpointing"))) b.activationCheckpointing(true);
        for (Map.Entry<String, Object> e : m.entrySet()) {
            if (!isKnownKey(e.getKey())) b.extra(e.getKey(), e.getValue());
        }
        return b.build();
    }

    private static boolean isKnownKey(String k) {
        return k.startsWith("zero_optimization")
                || k.startsWith("fp16") || k.startsWith("bf16")
                || k.equals("gradient_accumulation_steps")
                || k.equals("gradient_clipping") || k.equals("gradient_clip")
                || k.equals("train_micro_batch_size_per_gpu")
                || k.equals("train_batch_size")
                || k.equals("wall_clock_breakdown")
                || k.equals("activation_checkpointing");
    }

    private static Object nested(Map<String, Object> m, String a, String b) {
        Object o = m.get(a);
        if (o instanceof Map) return ((Map<?, ?>) o).get(b);
        return null;
    }

    private static void putInt(Map<String, Object> m, String k, java.util.function.IntConsumer c) {
        Object v = m.get(k);
        if (v instanceof Number) c.accept(((Number) v).intValue());
    }

    private static boolean asBool(Object o) {
        if (o == null) return false;
        if (o instanceof Boolean) return (Boolean) o;
        if (o instanceof Number) return ((Number) o).intValue() != 0;
        return Boolean.parseBoolean(String.valueOf(o));
    }

    public int zeroStage() { return zeroStage; }
    public boolean cpuOffload() { return cpuOffload; }
    public boolean nvmeOffload() { return nvmeOffload; }
    public boolean offloadParam() { return offloadParam; }
    public boolean offloadOptimizer() { return offloadOptimizer; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public double gradientClip() { return gradientClip; }
    public boolean overlapComm() { return overlapComm; }
    public int reduceBucketSize() { return reduceBucketSize; }
    public int allgatherBucketSize() { return allgatherBucketSize; }
    public String precision() { return precision; }
    public boolean wallClockBreakdown() { return wallClockBreakdown; }
    public boolean activationCheckpointing() { return activationCheckpointing; }
    public int trainMicroBatchSizePerGpu() { return trainMicroBatchSizePerGpu; }
    public int trainBatchSize() { return trainBatchSize; }
    public boolean contiguousGradients() { return contiguousGradients; }
    public boolean reduceScatter() { return reduceScatter; }
    public String zeroAllowUntestedOptimizer() { return zeroAllowUntestedOptimizer; }
    public Map<String, Object> extra() { return extra; }

    public boolean fp16() { return "fp16".equalsIgnoreCase(precision); }
    public boolean bf16() { return "bf16".equalsIgnoreCase(precision); }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        Map<String, Object> zero = new LinkedHashMap<>();
        zero.put("stage", zeroStage);
        zero.put("overlap_comm", overlapComm);
        zero.put("contiguous_gradients", contiguousGradients);
        zero.put("reduce_scatter", reduceScatter);
        zero.put("reduce_bucket_size", reduceBucketSize);
        zero.put("allgather_bucket_size", allgatherBucketSize);
        Map<String, Object> offOpt = new LinkedHashMap<>();
        offOpt.put("device", offloadOptimizer ? (nvmeOffload ? "nvme" : "cpu") : "none");
        zero.put("offload_optimizer", offOpt);
        Map<String, Object> offPar = new LinkedHashMap<>();
        offPar.put("device", offloadParam ? (nvmeOffload ? "nvme" : "cpu") : "none");
        zero.put("offload_param", offPar);
        m.put("zero_optimization", zero);
        m.put("gradient_accumulation_steps", gradientAccumulationSteps);
        m.put("gradient_clipping", gradientClip);
        m.put("train_micro_batch_size_per_gpu", trainMicroBatchSizePerGpu);
        m.put("train_batch_size", trainBatchSize);
        Map<String, Object> fp16 = new LinkedHashMap<>();
        fp16.put("enabled", fp16());
        m.put("fp16", fp16);
        Map<String, Object> bf16 = new LinkedHashMap<>();
        bf16.put("enabled", bf16());
        m.put("bf16", bf16);
        m.put("wall_clock_breakdown", wallClockBreakdown);
        Map<String, Object> act = new LinkedHashMap<>();
        act.put("partition_activations", activationCheckpointing);
        m.put("activation_checkpointing", act);
        // flat aliases for older Java MVP consumers
        m.put("zero_optimization.stage", zeroStage);
        m.put("zero_optimization.offload_optimizer.device", offloadOptimizer ? (cpuOffload ? "cpu" : "nvme") : "none");
        m.put("zero_optimization.offload_param.device", offloadParam ? (nvmeOffload ? "nvme" : "cpu") : "none");
        m.put("bf16.enabled", bf16());
        m.put("fp16.enabled", fp16());
        m.putAll(extra);
        return m;
    }

    @Override
    public String toString() {
        return "DeepSpeedConfig{zeroStage=" + zeroStage
                + ", precision=" + precision
                + ", gas=" + gradientAccumulationSteps
                + ", clip=" + gradientClip
                + ", offloadOpt=" + offloadOptimizer
                + ", offloadParam=" + offloadParam + '}';
    }

    public static final class Builder {
        private int zeroStage = 2;
        private boolean cpuOffload;
        private boolean nvmeOffload;
        private boolean offloadParam;
        private boolean offloadOptimizer;
        private int gradientAccumulationSteps = 1;
        private double gradientClip = 1.0;
        private boolean overlapComm = true;
        private int reduceBucketSize = 500_000_000;
        private int allgatherBucketSize = 500_000_000;
        private String precision = "fp32";
        private boolean wallClockBreakdown;
        private boolean activationCheckpointing;
        private int trainMicroBatchSizePerGpu = 1;
        private int trainBatchSize;
        private boolean contiguousGradients = true;
        private boolean reduceScatter = true;
        private String zeroAllowUntestedOptimizer = "true";
        private final Map<String, Object> extra = new LinkedHashMap<>();

        public Builder zeroStage(int zeroStage) { this.zeroStage = zeroStage; return this; }
        public Builder cpuOffload(boolean cpuOffload) { this.cpuOffload = cpuOffload; return this; }
        public Builder nvmeOffload(boolean nvmeOffload) { this.nvmeOffload = nvmeOffload; return this; }
        public Builder offloadParam(boolean offloadParam) { this.offloadParam = offloadParam; return this; }
        public Builder offloadOptimizer(boolean offloadOptimizer) { this.offloadOptimizer = offloadOptimizer; return this; }
        public Builder gradientAccumulationSteps(int s) { this.gradientAccumulationSteps = s; return this; }
        public Builder gradientClip(double gradientClip) { this.gradientClip = gradientClip; return this; }
        public Builder overlapComm(boolean overlapComm) { this.overlapComm = overlapComm; return this; }
        public Builder reduceBucketSize(int reduceBucketSize) { this.reduceBucketSize = reduceBucketSize; return this; }
        public Builder allgatherBucketSize(int v) { this.allgatherBucketSize = v; return this; }
        public Builder precision(String precision) { this.precision = precision; return this; }
        public Builder wallClockBreakdown(boolean v) { this.wallClockBreakdown = v; return this; }
        public Builder activationCheckpointing(boolean v) { this.activationCheckpointing = v; return this; }
        public Builder trainMicroBatchSizePerGpu(int v) { this.trainMicroBatchSizePerGpu = v; return this; }
        public Builder trainBatchSize(int v) { this.trainBatchSize = v; return this; }
        public Builder contiguousGradients(boolean v) { this.contiguousGradients = v; return this; }
        public Builder reduceScatter(boolean v) { this.reduceScatter = v; return this; }
        public Builder zeroAllowUntestedOptimizer(String v) { this.zeroAllowUntestedOptimizer = v; return this; }
        public Builder extra(String k, Object v) { this.extra.put(k, v); return this; }
        public DeepSpeedConfig build() { return new DeepSpeedConfig(this); }
    }
}
