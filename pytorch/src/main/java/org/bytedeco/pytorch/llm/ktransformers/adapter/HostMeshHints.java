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
package org.bytedeco.pytorch.llm.ktransformers.adapter;

import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtDevicePlacement;
import org.bytedeco.pytorch.llm.ktransformers.config.KtInferenceConfig;
import org.bytedeco.pytorch.llm.ktransformers.config.KtMoEConfig;
import org.bytedeco.pytorch.llm.ktransformers.util.DeviceBudget;
import org.bytedeco.pytorch.llm.ktransformers.util.KtPreconditions;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Explicit, documented host-mesh hints for industry-style platform integration.
 *
 * <p>Not bound to any single vendor mesh. Carries recommended degrees of
 * parallelism and offload flags that a host (ByteDance / Alibaba / Tencent style
 * training grid, or local accelerate / DeepSpeed) can consume together with
 * {@link KtConfig}.
 *
 * <pre>{@code
 * HostMeshHints h = HostMeshHints.suggest(KtConfig.miniDemo(), 8);
 * Map&lt;String, Object&gt; ds = h.deepSpeedZeROHints();
 * Map&lt;String, Object&gt; acc = h.accelerateHints();
 * }</pre>
 *
 * <p>Recommended combinations (documented constants + code):
 * <ul>
 *   <li>Single node, many experts → high {@code expertParallel}, {@code cpuOffloadExperts=true}</li>
 *   <li>Multi-GPU dense → {@code tensorParallel} ≥ 1, low EP</li>
 *   <li>Long context 24GB class → {@code diskPrefix=true}, modest TP, CPU expert offload</li>
 * </ul>
 */
public final class HostMeshHints {

    /** Suggested data-parallel degree (process replicas). */
    private final int dataParallel;
    /** Suggested expert-parallel degree (shard experts across ranks). */
    private final int expertParallel;
    /** Suggested tensor-parallel degree (column/row shard linears). */
    private final int tensorParallel;
    /** Pipeline-parallel degree (optional; 1 = disabled). */
    private final int pipelineParallel;
    /** Keep non-selected / frozen experts on CPU. */
    private final boolean cpuOffloadExperts;
    /** Enable three-tier disk prefix cache. */
    private final boolean diskPrefix;
    /** Prefer NUMA-aware expert placement. */
    private final boolean numaAware;
    /** Soft GPU memory fraction for planners. */
    private final double gpuMemFraction;
    /** Free-form notes for operators. */
    private final String notes;
    private final Map<String, String> extra;

    private HostMeshHints(Builder b) {
        this.dataParallel = Math.max(1, b.dataParallel);
        this.expertParallel = Math.max(1, b.expertParallel);
        this.tensorParallel = Math.max(1, b.tensorParallel);
        this.pipelineParallel = Math.max(1, b.pipelineParallel);
        this.cpuOffloadExperts = b.cpuOffloadExperts;
        this.diskPrefix = b.diskPrefix;
        this.numaAware = b.numaAware;
        this.gpuMemFraction = b.gpuMemFraction <= 0 || b.gpuMemFraction > 1
                ? 0.90 : b.gpuMemFraction;
        this.notes = b.notes != null ? b.notes : "";
        this.extra = Collections.unmodifiableMap(new LinkedHashMap<>(b.extra));
    }

    public int dataParallel() { return dataParallel; }
    public int expertParallel() { return expertParallel; }
    public int tensorParallel() { return tensorParallel; }
    public int pipelineParallel() { return pipelineParallel; }
    public boolean cpuOffloadExperts() { return cpuOffloadExperts; }
    public boolean diskPrefix() { return diskPrefix; }
    public boolean numaAware() { return numaAware; }
    public double gpuMemFraction() { return gpuMemFraction; }
    public String notes() { return notes; }
    public Map<String, String> extra() { return extra; }

    public static Builder builder() { return new Builder(); }

    /** Conservative single-process defaults (CI / laptop). */
    public static HostMeshHints singleProcess() {
        return builder()
                .dataParallel(1).expertParallel(1).tensorParallel(1).pipelineParallel(1)
                .cpuOffloadExperts(true).diskPrefix(true).numaAware(false)
                .gpuMemFraction(0.90)
                .notes("single-process mini / CI")
                .build();
    }

    /**
     * Suggest mesh degrees from model shape + available GPU count.
     *
     * @param config   KT config (MoE / placement / cache)
     * @param numGpus  visible accelerator count (1 if CPU-only)
     */
    public static HostMeshHints suggest(KtConfig config, int numGpus) {
        Objects.requireNonNull(config, "config");
        int gpus = Math.max(1, numGpus);
        KtMoEConfig moe = config.moe();
        int experts = Math.max(1, moe.numExperts());
        // EP: shard experts; cap by gpus and expert count
        int ep = Math.min(gpus, experts);
        // TP: only if wide hidden and multi-GPU leftover
        int tp = 1;
        if (gpus >= 4 && config.hiddenSize() >= 2048) {
            tp = Math.min(2, gpus / Math.max(1, ep));
            tp = Math.max(1, tp);
        }
        // DP: remaining ranks
        int used = Math.max(1, ep * tp);
        int dp = Math.max(1, gpus / used);
        boolean offload = config.placement().offloadFrozenToCpu()
                || moe.schedule() != KtMoEConfig.SchedulePolicy.GPU_FIRST;
        boolean disk = config.cache().prefixEnable() && config.cache().diskEnabled();
        String note = "suggest gpus=" + gpus + " experts=" + experts
                + " schedule=" + moe.schedule();
        return builder()
                .dataParallel(dp)
                .expertParallel(ep)
                .tensorParallel(tp)
                .pipelineParallel(1)
                .cpuOffloadExperts(offload)
                .diskPrefix(disk || config.cache().prefixEnable())
                .numaAware(moe.numaAware())
                .gpuMemFraction(config.placement().gpuMemFraction())
                .notes(note)
                .extra("model", config.modelNameOrPath())
                .extra("family", config.modelFamily() != null
                        ? config.modelFamily().name() : "GENERIC")
                .build();
    }

    public static HostMeshHints suggest(KtConfig config) {
        return suggest(config, 1);
    }

    /** Parse optional flat map ({@code kt_dp}, {@code data_parallel}, …). */
    public static HostMeshHints fromMap(Map<String, ?> raw) {
        if (raw == null || raw.isEmpty()) return singleProcess();
        Builder b = builder();
        Object dp = first(raw, "kt_dp", "data_parallel", "dp");
        Object ep = first(raw, "kt_ep", "expert_parallel", "ep");
        Object tp = first(raw, "kt_tp", "tensor_parallel", "tp");
        Object pp = first(raw, "kt_pp", "pipeline_parallel", "pp");
        if (dp != null) b.dataParallel(asInt(dp, 1));
        if (ep != null) b.expertParallel(asInt(ep, 1));
        if (tp != null) b.tensorParallel(asInt(tp, 1));
        if (pp != null) b.pipelineParallel(asInt(pp, 1));
        Object off = first(raw, "kt_cpu_offload_experts", "cpu_offload_experts");
        if (off != null) b.cpuOffloadExperts(asBool(off, true));
        Object disk = first(raw, "kt_disk_prefix", "disk_prefix");
        if (disk != null) b.diskPrefix(asBool(disk, true));
        Object numa = first(raw, "kt_numa_aware", "numa_aware");
        if (numa != null) b.numaAware(asBool(numa, false));
        Object frac = first(raw, "kt_gpu_mem_fraction", "gpu_mem_fraction");
        if (frac != null) b.gpuMemFraction(asDouble(frac, 0.90));
        return b.notes("fromMap").build();
    }

    /**
     * Apply hints onto a placement builder (offload + mem fraction).
     */
    public KtDevicePlacement toPlacement(KtDevicePlacement base) {
        KtDevicePlacement src = base != null ? base : KtDevicePlacement.defaults();
        return KtDevicePlacement.builder()
                .defaultCompute(src.defaultCompute())
                .layerDeviceMap(src.layerDeviceMap())
                .expertDeviceMap(src.expertDeviceMap())
                .pinMemory(src.pinMemory() || numaAware)
                .offloadFrozenToCpu(cpuOffloadExperts || src.offloadFrozenToCpu())
                .gpuMemFraction(gpuMemFraction)
                .cpuExpertBytesBudget(src.cpuExpertBytesBudget())
                .diskBytesBudget(diskPrefix
                        ? Math.max(src.diskBytesBudget(), 1L << 30)
                        : src.diskBytesBudget())
                .build();
    }

    /**
     * Hints consumable by accelerate-style PartialState / device mesh setup.
     * Keys are stable strings; values are JSON-friendly primitives.
     */
    public Map<String, Object> accelerateHints() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("num_processes", dataParallel * expertParallel * tensorParallel * pipelineParallel);
        m.put("data_parallel_size", dataParallel);
        m.put("expert_parallel_size", expertParallel);
        m.put("tensor_parallel_size", tensorParallel);
        m.put("pipeline_parallel_size", pipelineParallel);
        m.put("cpu_offload", cpuOffloadExperts);
        m.put("mixed_precision", "bf16");
        m.put("dynamo_backend", "no");
        m.put("gpu_mem_fraction", gpuMemFraction);
        return m;
    }

    /**
     * DeepSpeed ZeRO combination table (recommended, not auto-applied).
     *
     * <ul>
     *   <li>DP only, small model → ZeRO-1</li>
     *   <li>DP + CPU offload experts → ZeRO-2 + offload_optimizer optional</li>
     *   <li>Large MoE + multi-node → ZeRO-3 + expert parallel</li>
     * </ul>
     */
    public Map<String, Object> deepSpeedZeROHints() {
        Map<String, Object> m = new LinkedHashMap<>();
        int world = dataParallel * expertParallel * tensorParallel;
        int stage;
        if (world <= 2 && !cpuOffloadExperts) {
            stage = 1;
        } else if (expertParallel > 1 || cpuOffloadExperts) {
            stage = world >= 8 ? 3 : 2;
        } else {
            stage = 2;
        }
        m.put("zero_optimization.stage", stage);
        m.put("zero_optimization.overlap_comm", true);
        m.put("zero_optimization.contiguous_gradients", true);
        m.put("zero_optimization.offload_param.device",
                cpuOffloadExperts ? "cpu" : "none");
        m.put("zero_optimization.offload_optimizer.device",
                cpuOffloadExperts && stage >= 2 ? "cpu" : "none");
        m.put("data_parallel_size", dataParallel);
        m.put("expert_parallel_size", expertParallel);
        m.put("tensor_parallel_size", tensorParallel);
        m.put("pipeline_parallel_size", pipelineParallel);
        m.put("activation_checkpointing", stage >= 2);
        m.put("notes", notes);
        return m;
    }

    /** Flat {@code kt_*} map mergeable into FactoryArgs. */
    public Map<String, Object> toKtKeys() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("kt_dp", dataParallel);
        m.put("kt_ep", expertParallel);
        m.put("kt_tp", tensorParallel);
        m.put("kt_pp", pipelineParallel);
        m.put("kt_cpu_offload_experts", cpuOffloadExperts);
        m.put("kt_disk_prefix", diskPrefix);
        m.put("kt_numa_aware", numaAware);
        m.put("kt_gpu_mem_fraction", gpuMemFraction);
        return m;
    }

    /**
     * World-size sanity: product of parallel degrees should match host world
     * when the host is mesh-aware; returns true if {@code hostWorld} is 1
     * (single process) or divisible appropriately.
     */
    public boolean compatibleWithWorldSize(int hostWorld) {
        KtPreconditions.checkArgument(hostWorld >= 1, "hostWorld >= 1");
        if (hostWorld == 1) return dataParallel == 1 && expertParallel == 1
                && tensorParallel == 1 && pipelineParallel == 1
                || true; // single process can still *plan* multi-dim for export
        int product = dataParallel * expertParallel * tensorParallel * pipelineParallel;
        return hostWorld % product == 0 || product % hostWorld == 0;
    }

    /** Estimate whether budget allows keeping {@code gpuExperts} on GPU. */
    public boolean fitsExpertBudget(DeviceBudget budget, int gpuExperts, long bytesPerExpert) {
        Objects.requireNonNull(budget, "budget");
        long need = Math.max(0, gpuExperts) * Math.max(0L, bytesPerExpert);
        return budget.allowsGpuExpertPromote(need);
    }

    /** Inference concurrency hint derived from DP and config. */
    public int suggestedConcurrency(KtInferenceConfig inf) {
        int base = inf != null ? Math.max(1, inf.concurrency()) : 1;
        return Math.max(base, dataParallel);
    }

    @Override
    public String toString() {
        return "HostMeshHints{dp=" + dataParallel
                + ", ep=" + expertParallel
                + ", tp=" + tensorParallel
                + ", pp=" + pipelineParallel
                + ", cpuOffloadExperts=" + cpuOffloadExperts
                + ", diskPrefix=" + diskPrefix
                + ", numa=" + numaAware
                + ", gpuFrac=" + gpuMemFraction
                + ", notes=" + notes + "}";
    }

    public static final class Builder {
        private int dataParallel = 1;
        private int expertParallel = 1;
        private int tensorParallel = 1;
        private int pipelineParallel = 1;
        private boolean cpuOffloadExperts = true;
        private boolean diskPrefix = true;
        private boolean numaAware = false;
        private double gpuMemFraction = 0.90;
        private String notes = "";
        private final Map<String, String> extra = new LinkedHashMap<>();

        public Builder dataParallel(int v) { this.dataParallel = v; return this; }
        public Builder expertParallel(int v) { this.expertParallel = v; return this; }
        public Builder tensorParallel(int v) { this.tensorParallel = v; return this; }
        public Builder pipelineParallel(int v) { this.pipelineParallel = v; return this; }
        public Builder cpuOffloadExperts(boolean v) { this.cpuOffloadExperts = v; return this; }
        public Builder diskPrefix(boolean v) { this.diskPrefix = v; return this; }
        public Builder numaAware(boolean v) { this.numaAware = v; return this; }
        public Builder gpuMemFraction(double v) { this.gpuMemFraction = v; return this; }
        public Builder notes(String v) { this.notes = v; return this; }
        public Builder extra(String k, String v) {
            if (k != null) this.extra.put(k, v != null ? v : "");
            return this;
        }

        public HostMeshHints build() { return new HostMeshHints(this); }
    }

    private static Object first(Map<String, ?> raw, String... keys) {
        for (String k : keys) {
            if (raw.containsKey(k) && raw.get(k) != null) return raw.get(k);
        }
        return null;
    }

    private static int asInt(Object v, int dft) {
        if (v instanceof Number) return ((Number) v).intValue();
        try { return Integer.parseInt(String.valueOf(v).trim()); }
        catch (Exception e) { return dft; }
    }

    private static double asDouble(Object v, double dft) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        try { return Double.parseDouble(String.valueOf(v).trim()); }
        catch (Exception e) { return dft; }
    }

    private static boolean asBool(Object v, boolean dft) {
        if (v instanceof Boolean) return (Boolean) v;
        String s = String.valueOf(v).trim().toLowerCase();
        if ("1".equals(s) || "true".equals(s) || "yes".equals(s)) return true;
        if ("0".equals(s) || "false".equals(s) || "no".equals(s)) return false;
        return dft;
    }
}
