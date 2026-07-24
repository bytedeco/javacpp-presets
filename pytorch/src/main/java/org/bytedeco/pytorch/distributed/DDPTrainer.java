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
package org.bytedeco.pytorch.distributed;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.optim.Optimizer;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Data-parallel trainer: broadcast parameters at init, allreduce+average
 * gradients after each backward, then {@code optimizer.step()}.
 *
 * <pre>{@code
 * try (DDPTrainer trainer = DDPTrainer.create(model, pg)) {
 *     Tensor loss = trainer.step(input, target, optimizer);
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DDPTrainer implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "1.0";

    private final Module model;
    private final ProcessGroupWrapper processGroup;
    private final ModuleForward forward;
    private final Map<String, Object> extraState = new HashMap<>();
    private long numForwardCalls;
    private long numBackwardCalls;

    public DDPTrainer(Module model, ProcessGroupWrapper processGroup) {
        this.model = Objects.requireNonNull(model, "model");
        this.processGroup = Objects.requireNonNull(processGroup, "processGroup");
        this.forward = ModuleForward.of(model);
        initialize();
    }

    public static DDPTrainer create(Module model, ProcessGroupWrapper pg) {
        return builder().module(model).processGroup(pg).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private void initialize() {
        Device device = processGroup.getDevice();
        model.to(device, true);
        if (processGroup.getWorldSize() > 1) {
            broadcastInitialParameters();
        }
        System.out.printf(
                "[DDPTrainer] Initialized on rank %d with device=%s, worldSize=%d%n",
                processGroup.getRank(), device, processGroup.getWorldSize());
    }

    private void broadcastInitialParameters() {
        for (Tensor p : collectParameters()) {
            processGroup.broadcast(p, 0);
        }
    }

    private List<Tensor> collectParameters() {
        List<Tensor> params = new ArrayList<>();
        TensorVector paramVec = model.parameters();
        for (long i = 0, n = paramVec.size(); i < n; i++) {
            Tensor p = paramVec.get(i);
            if (p != null && !p.isNull()) {
                params.add(p);
            }
        }
        return params;
    }

    public Tensor forward(Tensor input) {
        numForwardCalls++;
        return forward.apply(model, input);
    }

    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        Tensor output = forward(input);
        Tensor loss = cross_entropy(output, target);
        optimizer.zero_grad();
        loss.backward();
        numBackwardCalls++;
        reduceGradients();
        optimizer.step();
        return loss;
    }

    public Tensor trainingStep(Tensor input, Tensor target, Optimizer optimizer) {
        return step(input, target, optimizer);
    }

    public void synchronize() {
        reduceGradients();
    }

    private void reduceGradients() {
        if (processGroup.getWorldSize() <= 1) {
            return;
        }
        List<Tensor> gradients = new ArrayList<>();
        TensorVector paramVec = model.parameters();
        for (long i = 0, n = paramVec.size(); i < n; i++) {
            Tensor p = paramVec.get(i);
            if (p == null || p.isNull()) {
                continue;
            }
            try {
                Tensor grad = p.grad();
                if (grad != null && !grad.isNull() && grad.defined()) {
                    gradients.add(grad);
                }
            } catch (Exception ignored) {
                // skip param without grad
            }
        }
        if (gradients.isEmpty()) {
            return;
        }
        processGroup.allreduce(gradients, ReduceOp.RedOpType.SUM);
        Scalar world = new Scalar(processGroup.getWorldSize());
        for (Tensor g : gradients) {
            g.div_(world);
        }
    }

    public Module getModule() { return model; }
    public Module getLocalModule() { return model; }
    public Module getModuleForTraining() { return model; }

    public List<Tensor> parameters() {
        return collectParameters();
    }

    public void setParameters(List<Tensor> params) {
        TensorVector paramVec = model.parameters();
        int i = 0;
        for (long j = 0, n = paramVec.size(); j < n && i < params.size(); j++) {
            Tensor p = paramVec.get(j);
            if (p != null && !p.isNull()) {
                p.set_(params.get(i));
                i++;
            }
        }
    }

    public Map<String, Tensor> namedBuffers() {
        Map<String, Tensor> buffers = new HashMap<>();
        TensorVector bufVec = model.buffers();
        for (long i = 0, n = bufVec.size(); i < n; i++) {
            buffers.put("buffer_" + i, bufVec.get(i));
        }
        return buffers;
    }

    public Map<String, Object> getTempStateDict() { return extraState; }
    public void loadTempStateDict(Map<String, Object> state) { extraState.putAll(state); }

    public void train() { model.train(true); }
    public void eval() { model.eval(); }
    public boolean isTraining() { return model.is_training(); }

    public long getNumForwardCalls() { return numForwardCalls; }
    public long getNumBackwardCalls() { return numBackwardCalls; }
    public void resetStats() { numForwardCalls = 0; numBackwardCalls = 0; }

    public ProcessGroupWrapper getProcessGroup() { return processGroup; }
    public int getRank() { return processGroup.getRank(); }
    public int getWorldSize() { return processGroup.getWorldSize(); }
    public boolean isMainProcess() { return processGroup.isMainProcess(); }
    public Device getDevice() { return processGroup.getDevice(); }

    @Override
    public void close() {
        model.close();
    }

    @Override
    public String toString() {
        return "DDPTrainer{rank=" + processGroup.getRank()
                + ", worldSize=" + processGroup.getWorldSize()
                + ", device=" + processGroup.getDevice()
                + ", forwardCalls=" + numForwardCalls + '}';
    }

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper processGroup;

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        /** Accepted for API parity; currently a no-op. */
        public Builder broadcastBuffers(boolean b) { return this; }
        /** Accepted for API parity; currently a no-op. */
        public Builder gradientAsBucketView(boolean b) { return this; }
        /** Accepted for API parity; currently a no-op. */
        public Builder bucketCapKb(int kb) { return this; }

        public DDPTrainer build() {
            Objects.requireNonNull(module, "module is required");
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new DDPTrainer(module, processGroup);
        }
    }
}
