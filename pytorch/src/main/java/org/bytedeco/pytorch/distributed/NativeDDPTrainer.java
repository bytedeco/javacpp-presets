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
import org.bytedeco.pytorch.BoolVector;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.SizeTStringMap;
import org.bytedeco.pytorch.SizeTVector;
import org.bytedeco.pytorch.SizeTVectorVector;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Industrial data-parallel trainer on top of c10d.
 *
 * <p>Prefers native {@link Reducer} (bucketed async allreduce, Python DDP path).
 * If Reducer construction fails under JavaCPP (intrusive_ptr / bucket types),
 * falls back to post-backward coalesced allreduce and logs
 * {@code mode=FALLBACK} — never silently fakes industrial DDP.
 *
 * <pre>{@code
 * try (NativeDDPTrainer ddp = NativeDDPTrainer.create(model, pg)) {
 *     Tensor loss = ddp.step(input, target, optimizer);
 *     System.out.println(ddp.commMode()); // REDUCER or FALLBACK
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class NativeDDPTrainer implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public enum CommMode { REDUCER, FALLBACK }

    public static final String VERSION = "2.0";

    private final Module model;
    private final ProcessGroupWrapper processGroup;
    private final ModuleForward forward;
    private final boolean broadcastBuffers;
    private final boolean findUnusedParameters;
    private final boolean gradientAsBucketView;
    private final long bucketCapBytes;
    private final long firstBucketCapBytes;
    private final boolean staticGraph;

    private Reducer reducer;
    private CommMode commMode = CommMode.FALLBACK;
    private boolean requireBackwardSync = true;
    private long numForwardCalls;
    private long numBackwardCalls;
    private long numSyncCalls;
    private String reducerInitError;

    public NativeDDPTrainer(Module model, ProcessGroupWrapper processGroup) {
        this(model, processGroup, true, false, false, 25L * 1024L * 1024L,
                1024L * 1024L, false, true);
    }

    public NativeDDPTrainer(
            Module model,
            ProcessGroupWrapper processGroup,
            boolean broadcastBuffers,
            boolean findUnusedParameters,
            boolean gradientAsBucketView,
            long bucketCapBytes,
            long firstBucketCapBytes,
            boolean staticGraph,
            boolean tryReducer) {
        this.model = Objects.requireNonNull(model, "model");
        this.processGroup = Objects.requireNonNull(processGroup, "processGroup");
        this.broadcastBuffers = broadcastBuffers;
        this.findUnusedParameters = findUnusedParameters;
        this.gradientAsBucketView = gradientAsBucketView;
        this.bucketCapBytes = Math.max(1024L, bucketCapBytes);
        this.firstBucketCapBytes = Math.max(1024L, firstBucketCapBytes);
        this.staticGraph = staticGraph;
        this.forward = ModuleForward.of(model);

        Device device = processGroup.getDevice();
        model.to(device, true);
        if (processGroup.getWorldSize() > 1) {
            broadcastInitialParameters();
            if (broadcastBuffers) {
                broadcastInitialBuffers();
            }
        }
        if (tryReducer && processGroup.getWorldSize() > 1) {
            tryInitReducer();
        } else {
            commMode = CommMode.FALLBACK;
            if (processGroup.getWorldSize() <= 1) {
                reducerInitError = "worldSize<=1 — no collective needed";
            }
        }
        System.out.printf(
                "[NativeDDPTrainer] rank=%d worldSize=%d device=%s mode=%s bucketCapBytes=%d%s%n",
                processGroup.getRank(), processGroup.getWorldSize(), device,
                commMode, this.bucketCapBytes,
                reducerInitError != null ? (" initNote=" + reducerInitError) : "");
    }

    public static NativeDDPTrainer create(Module model, ProcessGroupWrapper pg) {
        return builder().module(model).processGroup(pg).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private void broadcastInitialParameters() {
        for (Tensor p : collectParameters()) {
            processGroup.broadcast(p, 0);
        }
    }

    private void broadcastInitialBuffers() {
        TensorVector bufs = model.buffers();
        for (long i = 0, n = bufs.size(); i < n; i++) {
            Tensor b = bufs.get(i);
            if (b != null && !b.isNull() && b.defined()) {
                processGroup.broadcast(b, 0);
            }
        }
    }

    private void tryInitReducer() {
        try {
            TensorVector params = collectParamVector();
            long n = params.size();
            if (n == 0) {
                reducerInitError = "no parameters";
                commMode = CommMode.FALLBACK;
                return;
            }
            // Single-bucket assignment (index order) — stable under JavaCPP;
            // multi-bucket by byte cap can be refined once Reducer smoke is green.
            SizeTVector bucket = new SizeTVector();
            for (long i = 0; i < n; i++) {
                bucket.push_back(i);
            }
            SizeTVectorVector buckets = new SizeTVectorVector(bucket);

            BoolVector expectSparse = new BoolVector();
            expectSparse.resize(n);
            for (long i = 0; i < n; i++) {
                expectSparse.put(i, false);
            }

            SizeTStringMap names = new SizeTStringMap();
            for (long i = 0; i < n; i++) {
                names.put(i, "p" + i);
            }

            LongVector capList = new LongVector();
            capList.push_back(bucketCapBytes);

            ProcessGroup pg = processGroup.getProcessGroup();
            reducer = new Reducer(
                    params,
                    buckets,
                    pg,
                    expectSparse,
                    bucketCapBytes,
                    findUnusedParameters,
                    gradientAsBucketView,
                    names,
                    firstBucketCapBytes,
                    /*skip_all_reduce_unused_params*/ false,
                    /*use_python_reducer*/ false,
                    capList);
            if (staticGraph) {
                try {
                    reducer.set_static_graph();
                } catch (Throwable ignored) {
                }
            }
            commMode = CommMode.REDUCER;
            reducerInitError = null;
        } catch (Throwable t) {
            reducer = null;
            commMode = CommMode.FALLBACK;
            reducerInitError = t.getClass().getSimpleName() + ": " + t.getMessage();
            System.err.println("[NativeDDPTrainer] Reducer init failed — mode=FALLBACK: " + reducerInitError);
        }
    }

    private TensorVector collectParamVector() {
        TensorVector out = new TensorVector();
        TensorVector params = model.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p != null && !p.isNull()) {
                out.push_back(p);
            }
        }
        return out;
    }

    private List<Tensor> collectParameters() {
        List<Tensor> list = new ArrayList<>();
        TensorVector params = model.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor p = params.get(i);
            if (p != null && !p.isNull()) {
                list.add(p);
            }
        }
        return list;
    }

    public Tensor forward(Tensor input) {
        numForwardCalls++;
        if (commMode == CommMode.REDUCER && reducer != null) {
            try {
                reducer.prepare_for_forward();
            } catch (Throwable ignored) {
            }
        }
        return forward.apply(model, input);
    }

    /**
     * Standard CE training step: forward → loss → zero_grad → backward (+ sync) → step.
     */
    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        Tensor output = forward(input);
        Tensor loss = DistributedLoss.crossEntropy(output, target);
        optimizer.zero_grad();
        backward(loss, output);
        optimizer.step();
        return loss;
    }

    public Tensor trainingStep(Tensor input, Tensor target, Optimizer optimizer) {
        return step(input, target, optimizer);
    }

    /**
     * Backward with optional DDP gradient sync. When {@link #noSync()} is active,
     * skips collective (gradient accumulation micro-step).
     */
    public void backward(Tensor loss) {
        backward(loss, null);
    }

    public void backward(Tensor loss, Tensor outputForReducer) {
        Objects.requireNonNull(loss, "loss");
        if (commMode == CommMode.REDUCER && reducer != null && requireBackwardSync) {
            try {
                TensorVector outputs = new TensorVector();
                if (outputForReducer != null && !outputForReducer.isNull()) {
                    outputs.push_back(outputForReducer);
                } else {
                    outputs.push_back(loss);
                }
                reducer.prepare_for_backward(outputs);
            } catch (Throwable t) {
                System.err.println("[NativeDDPTrainer] prepare_for_backward failed, using FALLBACK sync: "
                        + t.getMessage());
                loss.backward();
                numBackwardCalls++;
                fallbackReduceGradients();
                return;
            }
            loss.backward();
            numBackwardCalls++;
            // Reducer hooks fire during backward; no explicit allreduce needed.
            numSyncCalls++;
            return;
        }

        loss.backward();
        numBackwardCalls++;
        if (requireBackwardSync) {
            fallbackReduceGradients();
        }
    }

    /** Post-backward gradient allreduce + average (FALLBACK / single-process no-op). */
    public void synchronize() {
        fallbackReduceGradients();
    }

    private void fallbackReduceGradients() {
        if (processGroup.getWorldSize() <= 1) {
            return;
        }
        List<Tensor> gradients = new ArrayList<>();
        TensorVector paramVec = model.parameters();
        for (long i = 0, n = paramVec.size(); i < n; i++) {
            Tensor p = paramVec.get(i);
            if (p == null || p.isNull()) continue;
            try {
                Tensor grad = p.grad();
                if (grad != null && !grad.isNull() && grad.defined()) {
                    gradients.add(grad);
                }
            } catch (Exception ignored) {
            }
        }
        if (gradients.isEmpty()) {
            return;
        }
        try {
            processGroup.allreduceCoalesced(gradients, ReduceOp.RedOpType.SUM);
        } catch (Throwable t) {
            processGroup.allreduce(gradients, ReduceOp.RedOpType.SUM);
        }
        Scalar world = new Scalar(processGroup.getWorldSize());
        for (Tensor g : gradients) {
            g.div_(world);
        }
        numSyncCalls++;
    }

    /**
     * Disable gradient sync for the next backward (grad accumulation).
     * Call {@link #enableSync()} before the boundary step, or use try-with-resources
     * via {@link #noSync()}.
     */
    public void disableSync() { requireBackwardSync = false; }
    public void enableSync() { requireBackwardSync = true; }
    public boolean isSyncEnabled() { return requireBackwardSync; }

    /** Zero parameter gradients (public for Accelerator / grad-accum). */
    public void zeroGrad() {
        TensorVector paramVec = model.parameters();
        for (long i = 0, n = paramVec.size(); i < n; i++) {
            Tensor p = paramVec.get(i);
            if (p == null || p.isNull()) continue;
            try {
                Tensor g = p.grad();
                if (g != null && !g.isNull() && g.defined()) g.zero_();
            } catch (Exception ignored) {}
        }
    }

    /** RAII helper: {@code try (var ns = ddp.noSync()) { ddp.backward(loss); }}. */
    public NoSync noSync() {
        return new NoSync(this);
    }

    public static final class NoSync implements AutoCloseable {
        private final NativeDDPTrainer trainer;
        private final boolean prev;

        NoSync(NativeDDPTrainer trainer) {
            this.trainer = trainer;
            this.prev = trainer.requireBackwardSync;
            trainer.requireBackwardSync = false;
        }

        @Override
        public void close() {
            trainer.requireBackwardSync = prev;
        }
    }

    public Module getModule() { return model; }
    public Module getLocalModule() { return model; }
    public List<Tensor> parameters() { return collectParameters(); }
    public void train() { model.train(true); }
    public void eval() { model.eval(); }
    public boolean isTraining() { return model.is_training(); }

    public CommMode commMode() { return commMode; }
    public String getReducerInitError() { return reducerInitError; }
    public Reducer getReducer() { return reducer; }
    public ProcessGroupWrapper getProcessGroup() { return processGroup; }
    public int getRank() { return processGroup.getRank(); }
    public int getWorldSize() { return processGroup.getWorldSize(); }
    public boolean isMainProcess() { return processGroup.isMainProcess(); }
    public Device getDevice() { return processGroup.getDevice(); }
    public long getNumForwardCalls() { return numForwardCalls; }
    public long getNumBackwardCalls() { return numBackwardCalls; }
    public long getNumSyncCalls() { return numSyncCalls; }
    public void resetStats() { numForwardCalls = 0; numBackwardCalls = 0; numSyncCalls = 0; }

    @Override
    public void close() {
        if (reducer != null) {
            try {
                reducer.remove_autograd_hooks();
            } catch (Throwable ignored) {
            }
            reducer = null;
        }
    }

    @Override
    public String toString() {
        return "NativeDDPTrainer{rank=" + processGroup.getRank()
                + ", worldSize=" + processGroup.getWorldSize()
                + ", mode=" + commMode
                + ", forwards=" + numForwardCalls
                + ", syncs=" + numSyncCalls + '}';
    }

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper processGroup;
        private boolean broadcastBuffers = true;
        private boolean findUnusedParameters = false;
        private boolean gradientAsBucketView = false;
        private long bucketCapBytes = 25L * 1024L * 1024L;
        private long firstBucketCapBytes = 1024L * 1024L;
        private boolean staticGraph = false;
        private boolean tryReducer = true;

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder broadcastBuffers(boolean b) { this.broadcastBuffers = b; return this; }
        public Builder findUnusedParameters(boolean b) { this.findUnusedParameters = b; return this; }
        public Builder gradientAsBucketView(boolean b) { this.gradientAsBucketView = b; return this; }
        public Builder bucketCapKb(int kb) {
            this.bucketCapBytes = Math.max(1, kb) * 1024L;
            return this;
        }
        public Builder bucketCapBytes(long bytes) { this.bucketCapBytes = bytes; return this; }
        public Builder firstBucketCapBytes(long bytes) { this.firstBucketCapBytes = bytes; return this; }
        public Builder staticGraph(boolean b) { this.staticGraph = b; return this; }
        /** When false, skip Reducer and always use FALLBACK allreduce. */
        public Builder tryReducer(boolean b) { this.tryReducer = b; return this; }

        public NativeDDPTrainer build() {
            Objects.requireNonNull(module, "module is required");
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new NativeDDPTrainer(
                    module, processGroup, broadcastBuffers, findUnusedParameters,
                    gradientAsBucketView, bucketCapBytes, firstBucketCapBytes,
                    staticGraph, tryReducer);
        }
    }
}
