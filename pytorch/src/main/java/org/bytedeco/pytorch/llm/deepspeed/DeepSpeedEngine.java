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
package org.bytedeco.pytorch.llm.deepspeed;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.llm.deepspeed.runtime.CheckpointEngine;
import org.bytedeco.pytorch.llm.deepspeed.runtime.GradientClipper;
import org.bytedeco.pytorch.llm.deepspeed.zero.PartitionedParameter;
import org.bytedeco.pytorch.llm.deepspeed.zero.ZeroOptimizer;

import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * DeepSpeed training engine (Java port of {@code DeepSpeedEngine}).
 *
 * <p>Implements ZeRO stage bookkeeping, gradient accumulation, clipping,
 * checkpointing, and process-group collectives via {@link ZeroOptimizer}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DeepSpeedEngine implements AutoCloseable {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final Module module;
    private final DeepSpeedConfig config;
    private final ProcessGroupWrapper processGroup;
    private final ZeroOptimizer zero;
    private long microStep;
    private long globalStep;
    private boolean trainMode = true;
    private boolean offload;
    private double lastGradNorm;
    private int trainBatchSize;
    private final Map<String, Object> clientState = new LinkedHashMap<>();
    private final Map<String, Long> timers = new LinkedHashMap<>();

    public DeepSpeedEngine(Module module, Optimizer optimizer, DeepSpeedConfig config,
                           ProcessGroupWrapper processGroup) {
        this.module = Objects.requireNonNull(module, "module");
        this.config = config == null ? DeepSpeedConfig.defaults() : config;
        this.processGroup = processGroup;
        this.zero = new ZeroOptimizer(module, optimizer, this.config, processGroup);
        this.trainBatchSize = this.config.trainBatchSize();
        this.offload = this.config.offloadOptimizer() || this.config.cpuOffload();
    }

    public Module module() { return module; }
    public Optimizer optimizer() { return zero.optimizer(); }
    public DeepSpeedConfig config() { return config; }
    public ProcessGroupWrapper processGroup() { return processGroup; }
    public ZeroOptimizer zero() { return zero; }
    public long globalStep() { return globalStep; }
    public long microStep() { return microStep; }
    public int zeroStage() { return config.zeroStage(); }
    public List<PartitionedParameter> partitions() { return zero.partitions(); }
    public boolean isTrainMode() { return trainMode; }
    public boolean isGathered() { return zero.isParamsGathered(); }
    public boolean isOffload() { return offload; }
    public double getGlobalGradNorm() { return lastGradNorm; }
    public int getTrainBatchSize() { return trainBatchSize; }

    public void setTrainBatchSize(int batchSize) {
        this.trainBatchSize = Math.max(1, batchSize);
    }

    public int worldSize() { return zero.worldSize(); }
    public int rank() { return zero.rank(); }

    public boolean isGradientAccumulationBoundary() {
        return microStep > 0 && microStep % config.gradientAccumulationSteps() == 0;
    }

    public Tensor forward(Tensor input) {
        if (config.wallClockBreakdown()) timers.put("forward_start", System.nanoTime());
        if (config.zeroStage() >= 3) {
            zero.gatherParametersForForward();
        }
        Tensor out = module.forward(input);
        if (config.wallClockBreakdown()) {
            long t0 = timers.getOrDefault("forward_start", System.nanoTime());
            timers.put("forward_ns", System.nanoTime() - t0);
        }
        return out;
    }

    public void backward(Tensor loss) {
        Objects.requireNonNull(loss, "loss");
        if (config.wallClockBreakdown()) timers.put("backward_start", System.nanoTime());
        Tensor scaled = loss;
        if (config.gradientAccumulationSteps() > 1) {
            scaled = loss.div(new Scalar(config.gradientAccumulationSteps()));
        }
        scaled.backward();
        microStep++;
        if (isGradientAccumulationBoundary()) {
            zero.synchronizeGradients();
            if (config.gradientClip() > 0) {
                lastGradNorm = GradientClipper.clipGradNorm(
                        zero.partitions(), config.gradientClip(), processGroup);
            } else {
                lastGradNorm = GradientClipper.computeGradNorm(zero.partitions());
            }
        }
        if (config.wallClockBreakdown()) {
            long t0 = timers.getOrDefault("backward_start", System.nanoTime());
            timers.put("backward_ns", System.nanoTime() - t0);
        }
    }

    public void step() {
        if (!isGradientAccumulationBoundary()) return;
        if (config.wallClockBreakdown()) timers.put("step_start", System.nanoTime());
        if (config.cpuOffload() || config.offloadOptimizer()) {
            offload = true;
        }
        zero.step();
        globalStep++;
        if (config.wallClockBreakdown()) {
            long t0 = timers.getOrDefault("step_start", System.nanoTime());
            timers.put("step_ns", System.nanoTime() - t0);
        }
    }

    public void zeroGrad() {
        zero.zeroGrad();
    }

    public void train() {
        trainMode = true;
        module.train(true);
    }

    public void eval() {
        trainMode = false;
        module.eval();
    }

    public void saveCheckpoint(Path dir) throws IOException {
        saveCheckpoint(dir, null);
    }

    public void saveCheckpoint(Path dir, Map<String, Object> extraClientState) throws IOException {
        Map<String, Object> cs = new LinkedHashMap<>(clientState);
        if (extraClientState != null) cs.putAll(extraClientState);
        cs.put("micro_step", microStep);
        CheckpointEngine.save(dir, module, config, globalStep, cs, processGroup);
    }

    public Map<String, Object> loadCheckpoint(Path dir) throws IOException, ClassNotFoundException {
        Map<String, Object> meta = CheckpointEngine.load(dir, module, processGroup);
        Object gs = meta.get("global_step");
        if (gs instanceof Number) globalStep = ((Number) gs).longValue();
        Object ms = meta.get("micro_step");
        if (ms instanceof Number) microStep = ((Number) ms).longValue();
        clientState.putAll(meta);
        zero.rebuildPartitions();
        return meta;
    }

    public void saveClientState(String key, Object value) {
        clientState.put(key, value);
    }

    public Object loadClientState(String key) {
        return clientState.get(key);
    }

    public Map<String, Object> memoryStats() {
        Map<String, Object> m = new LinkedHashMap<>();
        long totalParams = 0;
        long localParams = 0;
        long optimLocal = 0;
        long gradLocal = 0;
        long paramLocal = 0;
        for (PartitionedParameter p : zero.partitions()) {
            totalParams += p.numel;
            if (p.local) localParams += p.numel;
            optimLocal += p.optimStateBytesLocal;
            gradLocal += p.gradBytesLocal(worldSize(), config.zeroStage());
            paramLocal += p.paramBytesLocal(worldSize(), config.zeroStage());
        }
        m.put("total_param_numel", totalParams);
        m.put("local_param_numel", localParams);
        m.put("zero_stage", config.zeroStage());
        m.put("world_size", worldSize());
        m.put("cpu_offload", config.cpuOffload());
        m.put("nvme_offload", config.nvmeOffload());
        m.put("offload_optimizer", config.offloadOptimizer());
        m.put("offload_param", config.offloadParam());
        m.put("estimated_param_bytes_local", paramLocal);
        m.put("estimated_param_bytes_total", totalParams * 4);
        m.put("optim_state_bytes_local_est", optimLocal);
        m.put("grad_bytes_local_est", gradLocal);
        m.put("param_bytes_local_est", paramLocal);
        m.put("activation_checkpointing", config.activationCheckpointing());
        m.put("precision", config.precision());
        m.put("global_step", globalStep);
        m.put("last_grad_norm", lastGradNorm);
        if (config.wallClockBreakdown()) {
            m.putAll(timers);
        }
        return m;
    }

    public Map<String, Long> timers() {
        return Map.copyOf(timers);
    }

    @Override
    public void close() {
        // partitions held by zero; nothing else
    }

    @Override
    public String toString() {
        return "DeepSpeedEngine{zeroStage=" + zeroStage()
                + ", rank=" + rank() + "/" + worldSize()
                + ", globalStep=" + globalStep
                + ", train=" + trainMode + '}';
    }
}
