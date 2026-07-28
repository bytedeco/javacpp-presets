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
package org.bytedeco.pytorch.llm.accelerate;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.distributed.FSDPTrainer;
import org.bytedeco.pytorch.distributed.NativeDDPTrainer;
import org.bytedeco.pytorch.distributed.NativeFSDPTrainer;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.quantizer.AutocastContext;
import org.bytedeco.pytorch.llm.accelerate.plugins.DeepSpeedPlugin;
import org.bytedeco.pytorch.llm.accelerate.plugins.FullyShardedDataParallelPlugin;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;
import org.bytedeco.pytorch.llm.accelerate.utils.Operations;
import org.bytedeco.pytorch.llm.deepspeed.DeepSpeedEngine;
import org.bytedeco.pytorch.llm.deepspeed.runtime.CheckpointEngine;
import org.bytedeco.pytorch.llm.deepspeed.runtime.GradientClipper;
import org.bytedeco.pytorch.llm.deepspeed.zero.PartitionedParameter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.bytedeco.pytorch.global.torch.DeviceType;
import static org.bytedeco.pytorch.global.torch.cuda_is_available;
import static org.bytedeco.pytorch.global.torch.hasMPS;

/**
 * HuggingFace {@code accelerate.Accelerator} — full Java API surface.
 *
 * <p>Device placement, prepare(), gradient accumulation, mixed-precision flags,
 * gather/reduce helpers, DeepSpeed / FSDP plugins, multi-process launch.
 * Collectives use {@link ProcessGroupWrapper} when supplied.
 *
 * <pre>{@code
 * Accelerator acc = Accelerator.builder().mixedPrecision("fp32").build();
 * acc.prepare(model, optimizer);
 * try (Accelerator.GradientAccumulation ga = acc.accumulate()) {
 *     acc.backward(loss);
 * }
 * acc.step();
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Accelerator implements AutoCloseable {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";

    private final Device device;
    private final String mixedPrecision;
    private final int gradientAccumulationSteps;
    private final ProcessGroupWrapper processGroup;
    private final boolean cpuOffload;
    private final boolean evenBatches;
    private final DeepSpeedPlugin deepSpeedPlugin;
    private final FullyShardedDataParallelPlugin fsdpPlugin;
    private final Map<String, Object> state = new HashMap<>();

    private Module model;
    private Optimizer optimizer;
    private DeepSpeedEngine deepSpeedEngine;
    private FSDPTrainer fsdpTrainer;
    private NativeFSDPTrainer nativeFsdpTrainer;
    private NativeDDPTrainer nativeDdpTrainer;
    private long stepCount;
    private long microStep;
    private boolean prepared;
    private double lastGradNorm;
    private boolean syncGradients = true;

    private Accelerator(Builder b) {
        this.mixedPrecision = b.mixedPrecision == null ? "no" : b.mixedPrecision;
        this.gradientAccumulationSteps = Math.max(1, b.gradientAccumulationSteps);
        this.processGroup = b.processGroup;
        this.cpuOffload = b.cpuOffload;
        this.evenBatches = b.evenBatches;
        this.deepSpeedPlugin = b.deepSpeedPlugin;
        this.fsdpPlugin = b.fsdpPlugin;
        this.device = b.device != null ? b.device
                : (processGroup != null ? processGroup.getDevice() : autoDevice(b.cpu));
    }

    private static Device autoDevice(boolean forceCpu) {
        if (forceCpu) return new Device(DeviceType.CPU);
        try {
            if (cuda_is_available()) return new Device(DeviceType.CUDA);
        } catch (Throwable ignored) {}
        try {
            if (hasMPS()) return new Device(DeviceType.MPS);
        } catch (Throwable ignored) {}
        return new Device(DeviceType.CPU);
    }

    public static Builder builder() { return new Builder(); }
    public static Accelerator create() { return builder().build(); }
    public static String version() { return VERSION; }

    public static MultiProcessLauncher.LaunchResult launch(Class<?> mainClass, int numProcesses, String... args)
            throws IOException, InterruptedException {
        return MultiProcessLauncher.launch(numProcesses, mainClass, args);
    }

    public Device device() { return device; }
    public String mixedPrecision() { return mixedPrecision; }
    public int gradientAccumulationSteps() { return gradientAccumulationSteps; }
    public int numProcesses() {
        return processGroup == null ? 1 : processGroup.getWorldSize();
    }
    public int processIndex() {
        return processGroup == null ? 0 : processGroup.getRank();
    }
    public boolean isMainProcess() { return processIndex() == 0; }
    public boolean isLocalMainProcess() { return isMainProcess(); }
    public long stepCount() { return stepCount; }
    public Module model() { return model; }
    public Optimizer optimizer() { return optimizer; }
    public boolean isPrepared() { return prepared; }
    public boolean cpuOffload() { return cpuOffload; }
    public boolean evenBatches() { return evenBatches; }
    public ProcessGroupWrapper processGroup() { return processGroup; }
    public DeepSpeedPlugin deepSpeedPlugin() { return deepSpeedPlugin; }
    public FullyShardedDataParallelPlugin fsdpPlugin() { return fsdpPlugin; }
    public DeepSpeedEngine deepSpeedEngine() { return deepSpeedEngine; }
    public FSDPTrainer fsdpTrainer() { return fsdpTrainer; }
    public NativeFSDPTrainer nativeFsdpTrainer() { return nativeFsdpTrainer; }
    public NativeDDPTrainer nativeDdpTrainer() { return nativeDdpTrainer; }
    public double lastGradNorm() { return lastGradNorm; }

    public PartialState statePartial() {
        return processGroup != null ? PartialState.of(processGroup)
                : PartialState.builder().device(device).mixedPrecision(mixedPrecision).build();
    }

    public void prepare(Module model, Optimizer optimizer) {
        this.model = Objects.requireNonNull(model, "model");
        this.optimizer = optimizer;
        if (deepSpeedPlugin != null) {
            this.deepSpeedEngine = deepSpeedPlugin.initialize(model, optimizer, processGroup);
            this.model = deepSpeedEngine.module();
        } else if (fsdpPlugin != null && processGroup != null && processGroup.getWorldSize() > 1) {
            // Try native FSDP first; fall back to legacy
            if (fsdpPlugin.useNative()) {
                this.nativeFsdpTrainer = fsdpPlugin.wrapNative(model, processGroup);
                this.model = nativeFsdpTrainer.getModule();
            } else {
                this.fsdpTrainer = fsdpPlugin.wrapLegacy(model, processGroup);
                this.model = fsdpTrainer.getModule();
            }
        } else if (processGroup != null && processGroup.getWorldSize() > 1) {
            // No plugin but distributed: native DDP unless overridden
            this.nativeDdpTrainer = new NativeDDPTrainer(model, processGroup);
            this.model = nativeDdpTrainer.getModule();
        } else {
            this.model.to(device, false);
        }
        this.prepared = true;
        state.put("prepared_at", System.currentTimeMillis());
        state.put("device", String.valueOf(device));
        state.put("mixed_precision", mixedPrecision);
        state.put("num_processes", numProcesses());
    }

    public void prepare(Module model) {
        prepare(model, null);
    }

    public <T> DataLoaderShard<T> prepareDataLoader(List<T> data) {
        return new DataLoaderShard<>(data, processIndex(), numProcesses(), evenBatches);
    }

    public Tensor toDevice(Tensor t) {
        return t.to(device, t.scalar_type());
    }

    public GradientAccumulation accumulate() {
        return new GradientAccumulation(this);
    }

    public void backward(Tensor loss) {
        Objects.requireNonNull(loss, "loss");
        if (deepSpeedEngine != null) {
            deepSpeedEngine.backward(loss);
            microStep = deepSpeedEngine.microStep();
            return;
        }
        Tensor scaled = loss;
        if (gradientAccumulationSteps > 1) {
            scaled = loss.div(new Scalar(gradientAccumulationSteps));
        }
        if (isAutocastEnabled()) {
            try (AutocastContext ignored = openAutocast()) {
                // loss already computed; enter for bookkeeping parity with torch.autocast
            }
        }
        scaled.backward();
        microStep++;
        if (syncGradients && microStep % gradientAccumulationSteps == 0) {
            syncGradients();
        }
    }

    public void step(Optimizer opt) {
        if (deepSpeedEngine != null) {
            deepSpeedEngine.step();
            stepCount = deepSpeedEngine.globalStep();
            return;
        }
        Optimizer o = opt != null ? opt : optimizer;
        if (o == null) throw new IllegalStateException("No optimizer prepared");
        if (microStep % gradientAccumulationSteps != 0) return;
        o.step();
        o.zero_grad();
        stepCount++;
    }

    public void step() { step(optimizer); }

    public void zeroGrad() {
        if (deepSpeedEngine != null) deepSpeedEngine.zeroGrad();
        else if (nativeFsdpTrainer != null) nativeFsdpTrainer.zeroGrad();
        else if (nativeDdpTrainer != null) nativeDdpTrainer.zeroGrad();
        else if (optimizer != null) optimizer.zero_grad();
    }

    public double clipGradNorm(double maxNorm) {
        if (model == null) return 0;
        List<PartitionedParameter> parts = new ArrayList<>();
        TensorVector params = model.parameters();
        int world = numProcesses();
        int rank = processIndex();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            int owner = (int) (i % Math.max(1, world));
            parts.add(new PartitionedParameter((int) i, owner, owner == rank || world <= 1,
                    p.numel(), p, world, 0));
        }
        lastGradNorm = GradientClipper.clipGradNorm(parts, maxNorm, processGroup);
        return lastGradNorm;
    }

    public void syncGradients() {
        if (processGroup == null || processGroup.getWorldSize() <= 1 || model == null) return;
        List<Tensor> gradients = new ArrayList<>();
        TensorVector params = model.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p == null || p.isNull()) continue;
            try {
                Tensor g = p.grad();
                if (g != null && !g.isNull() && g.defined()) gradients.add(g);
            } catch (Exception ignored) {}
        }
        if (!gradients.isEmpty()) {
            processGroup.allreduce(gradients);
            Scalar world = new Scalar(processGroup.getWorldSize());
            for (Tensor g : gradients) g.div_(world);
        }
    }

    public Module unwrapModel(Module m) {
        Module target = m == null ? model : m;
        if (deepSpeedEngine != null) return deepSpeedEngine.module();
        if (fsdpTrainer != null) return fsdpTrainer.getModule();
        return target;
    }

    public void waitForEveryone() {
        if (processGroup != null && processGroup.getWorldSize() > 1) {
            try { processGroup.barrier(); } catch (Exception ignored) {}
        }
    }

    public void print(String msg) {
        if (isMainProcess()) System.out.println(msg);
    }

    public void freeMemory() {
        try { System.gc(); } catch (Throwable ignored) {}
        state.put("free_memory_at", System.currentTimeMillis());
    }

    public void saveState(String key, Object value) { state.put(key, value); }
    public Object loadState(String key) { return state.get(key); }
    public Map<String, Object> state() { return Map.copyOf(state); }

    public void saveState(Path dir) throws IOException {
        Files.createDirectories(dir);
        if (model != null) {
            CheckpointEngine.save(dir, model, null, stepCount, Map.copyOf(state), processGroup);
        }
        state.put("saved_to", dir.toString());
    }

    public void loadState(Path dir) throws IOException, ClassNotFoundException {
        if (model != null) {
            Map<String, Object> meta = CheckpointEngine.load(dir, model, processGroup);
            state.putAll(meta);
            Object gs = meta.get("global_step");
            if (gs instanceof Number) stepCount = ((Number) gs).longValue();
        }
    }

    public void saveModel(Path dir) throws IOException {
        saveState(dir);
    }

    public <T> List<T> gatherObject(T obj) {
        return Operations.gatherObject(obj, processGroup);
    }

    public <T> List<T> gatherForMetrics(T obj) {
        return gatherObject(obj);
    }

    public Tensor reduce(Tensor t, String reduction) {
        if ("mean".equalsIgnoreCase(reduction) || "avg".equalsIgnoreCase(reduction)) {
            return Operations.reduceMean(t, processGroup);
        }
        return Operations.reduceSum(t, processGroup);
    }

    public Tensor padAcrossProcesses(Tensor t) {
        return t;
    }

    public boolean isGradientAccumulationBoundary() {
        if (deepSpeedEngine != null) return deepSpeedEngine.isGradientAccumulationBoundary();
        return microStep > 0 && microStep % gradientAccumulationSteps == 0;
    }

    public Tensor trainingStep(Tensor input, Tensor target,
                               BiFunction<Module, Tensor, Tensor> forward,
                               Function<Tensor[], Tensor> lossFn) {
        if (model == null) throw new IllegalStateException("call prepare() first");
        Tensor x = toDevice(input);
        Tensor y = toDevice(target);
        Tensor out;
        if (isAutocastEnabled()) {
            try (AutocastContext ignored = openAutocast()) {
                out = forward.apply(model, x);
            }
        } else {
            out = forward.apply(model, x);
        }
        Tensor loss = lossFn.apply(new Tensor[]{out, y});
        backward(loss);
        step();
        return loss;
    }

    private boolean isAutocastEnabled() {
        return "fp16".equalsIgnoreCase(mixedPrecision) || "bf16".equalsIgnoreCase(mixedPrecision);
    }

    private AutocastContext openAutocast() {
        DeviceType dt = device.type();
        if ("bf16".equalsIgnoreCase(mixedPrecision)) {
            return new AutocastContext(dt, org.bytedeco.pytorch.global.torch.ScalarType.BFloat16, true, true);
        }
        if ("fp16".equalsIgnoreCase(mixedPrecision)) {
            return new AutocastContext(dt, org.bytedeco.pytorch.global.torch.ScalarType.Half, true, true);
        }
        return new AutocastContext(dt, false);
    }

    @Override
    public void close() {
        if (deepSpeedEngine != null) {
            try { deepSpeedEngine.close(); } catch (Exception ignored) {}
        }
        if (fsdpTrainer != null) {
            try { fsdpTrainer.close(); } catch (Exception ignored) {}
        }
        model = null;
        optimizer = null;
        prepared = false;
    }

    public static final class GradientAccumulation implements AutoCloseable {
        private final Accelerator acc;
        private final boolean prevSync;

        GradientAccumulation(Accelerator acc) {
            this.acc = acc;
            this.prevSync = acc.syncGradients;
            acc.syncGradients = true;
        }

        @Override
        public void close() {
            acc.syncGradients = prevSync;
        }
    }

    public static final class Builder {
        private Device device;
        private boolean cpu;
        private String mixedPrecision = "no";
        private int gradientAccumulationSteps = 1;
        private ProcessGroupWrapper processGroup;
        private boolean cpuOffload;
        private boolean evenBatches = true;
        private DeepSpeedPlugin deepSpeedPlugin;
        private FullyShardedDataParallelPlugin fsdpPlugin;

        public Builder device(Device device) { this.device = device; return this; }
        public Builder cpu(boolean cpu) { this.cpu = cpu; return this; }
        public Builder mixedPrecision(String mixedPrecision) {
            this.mixedPrecision = mixedPrecision; return this;
        }
        public Builder gradientAccumulationSteps(int steps) {
            this.gradientAccumulationSteps = steps; return this;
        }
        public Builder processGroup(ProcessGroupWrapper processGroup) {
            this.processGroup = processGroup; return this;
        }
        public Builder cpuOffload(boolean cpuOffload) {
            this.cpuOffload = cpuOffload; return this;
        }
        public Builder evenBatches(boolean evenBatches) {
            this.evenBatches = evenBatches; return this;
        }
        public Builder deepSpeedPlugin(DeepSpeedPlugin plugin) {
            this.deepSpeedPlugin = plugin; return this;
        }
        public Builder fsdpPlugin(FullyShardedDataParallelPlugin plugin) {
            this.fsdpPlugin = plugin; return this;
        }
        public Accelerator build() { return new Accelerator(this); }
    }
}
