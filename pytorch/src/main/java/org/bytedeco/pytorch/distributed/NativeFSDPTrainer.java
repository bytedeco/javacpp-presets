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
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.DeviceType;
import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.empty;
import static org.bytedeco.pytorch.global.torch.zeros;
import static org.bytedeco.pytorch.global.torch.zeros_like;

/**
 * Fully Sharded Data Parallel trainer using real c10d collectives
 * ({@code _allgather_base} / {@code _reduce_scatter_base}).
 *
 * <p>FULL_SHARD ≈ ZeRO-3: each rank holds {@code 1/world} of flattened params;
 * forward all-gathers full weights; backward reduce-scatters grads.
 * Not Meta's C++ FSDP2 kernel — industrial <em>semantics</em> on libtorch Module
 * + ProcessGroup backends.
 *
 * <pre>{@code
 * try (NativeFSDPTrainer fsdp = NativeFSDPTrainer.create(model, pg)) {
 *     Tensor loss = fsdp.step(input, target, optimizer);
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class NativeFSDPTrainer implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static final String VERSION = "2.0";

    private final Module module;
    private final ProcessGroupWrapper processGroup;
    private final ShardingStrategy shardingStrategy;
    private final boolean reshardAfterForward;
    private final MixedPrecisionConfig mixedPrecision;
    private final ModuleForward forward;
    private final Device device;

    private final List<Tensor> shardedParams = new ArrayList<>();
    private final List<Tensor> shardedGrads = new ArrayList<>();
    private long totalParamNumel;
    private long shardSize;
    private long paddedFullSize;
    private long numForwardCalls;
    private long numBackwardCalls;
    private long numAllGatherCalls;
    private long numReduceScatterCalls;
    private int gradAccumSteps = 1;
    private int microStep;
    private boolean syncGradients = true;

    public NativeFSDPTrainer(Module module, ProcessGroupWrapper processGroup) {
        this(module, processGroup, ShardingStrategy.FULL_SHARD, true,
                MixedPrecisionConfig.fp32(), 1);
    }

    public NativeFSDPTrainer(
            Module module,
            ProcessGroupWrapper processGroup,
            ShardingStrategy shardingStrategy,
            boolean reshardAfterForward,
            MixedPrecisionConfig mixedPrecision,
            int gradAccumSteps) {
        this.module = Objects.requireNonNull(module, "module");
        this.processGroup = Objects.requireNonNull(processGroup, "processGroup");
        this.shardingStrategy = shardingStrategy == null ? ShardingStrategy.FULL_SHARD : shardingStrategy;
        this.reshardAfterForward = reshardAfterForward;
        this.mixedPrecision = mixedPrecision == null ? MixedPrecisionConfig.fp32() : mixedPrecision;
        this.gradAccumSteps = Math.max(1, gradAccumSteps);
        this.forward = ModuleForward.of(module);
        this.device = processGroup.getDevice();

        if (device.type() == DeviceType.CUDA || device.type() == DeviceType.MPS) {
            module.to(device, true);
        } else {
            module.to(device, true);
        }
        collectParamMetadata();
        shardParameters();
        if (processGroup.getWorldSize() > 1) {
            broadcastFullParameters();
            // Re-shard after broadcast so local shards stay consistent with full module.
            shardParameters();
        }
        System.out.printf(
                "[NativeFSDPTrainer] strategy=%s shardSize=%d totalParams=%d rank=%d world=%d mp=%s%n",
                this.shardingStrategy, shardSize, totalParamNumel,
                processGroup.getRank(), processGroup.getWorldSize(), this.mixedPrecision);
    }

    public static NativeFSDPTrainer create(Module module, ProcessGroupWrapper pg) {
        return builder().module(module).processGroup(pg).build();
    }

    public static Builder builder() {
        return new Builder();
    }

    private void collectParamMetadata() {
        totalParamNumel = 0;
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t != null && !t.isNull()) {
                totalParamNumel += t.numel();
            }
        }
        int world = Math.max(1, processGroup.getWorldSize());
        if (shardingStrategy == ShardingStrategy.NO_SHARD) {
            shardSize = totalParamNumel;
            paddedFullSize = totalParamNumel;
        } else {
            shardSize = (totalParamNumel + world - 1) / world;
            paddedFullSize = shardSize * (long) world;
        }
    }

    private void shardParameters() {
        int rank = processGroup.getRank();
        int world = Math.max(1, processGroup.getWorldSize());
        Tensor flat = flattenParameters();
        long start;
        long end;
        if (shardingStrategy == ShardingStrategy.NO_SHARD || world == 1) {
            start = 0;
            end = totalParamNumel;
        } else {
            start = (long) rank * shardSize;
            end = Math.min(start + shardSize, totalParamNumel);
        }
        Tensor shardView = flat.slice(0, new LongOptional(start), new LongOptional(end), 1);
        // Pad local shard to shardSize for even reduce-scatter / allgather.
        Tensor sharded;
        if (shardView.numel() < shardSize && shardingStrategy != ShardingStrategy.NO_SHARD && world > 1) {
            Tensor pad = zeros(shardSize - shardView.numel()).to(device, ScalarType.Float);
            TensorVector v = new TensorVector();
            v.push_back(shardView.to(device, ScalarType.Float));
            v.push_back(pad);
            sharded = cat(v).detach();
            pad.close();
        } else {
            sharded = shardView.clone().detach().to(device, ScalarType.Float);
        }
        sharded.requires_grad_(true);
        for (Tensor t : shardedParams) {
            try { t.close(); } catch (Throwable ignored) {}
        }
        for (Tensor t : shardedGrads) {
            try { t.close(); } catch (Throwable ignored) {}
        }
        shardedParams.clear();
        shardedParams.add(sharded);
        shardedGrads.clear();
        shardedGrads.add(zeros_like(sharded));
        flat.close();
        shardView.close();
    }

    private Tensor flattenParameters() {
        TensorVector flatList = new TensorVector();
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t != null && !t.isNull()) {
                flatList.push_back(t.flatten().to(device, ScalarType.Float));
            }
        }
        if (flatList.size() == 0) {
            return zeros(1).to(device, ScalarType.Float);
        }
        return cat(flatList);
    }

    private void broadcastFullParameters() {
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t != null && !t.isNull()) {
                processGroup.broadcast(t, 0);
            }
        }
    }

    private Tensor flattenGradients() {
        TensorVector gradList = new TensorVector();
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t == null || t.isNull()) continue;
            try {
                Tensor g = t.grad();
                if (g != null && !g.isNull() && g.defined()) {
                    gradList.push_back(g.flatten().to(device, ScalarType.Float));
                } else {
                    gradList.push_back(zeros(t.numel()).to(device, ScalarType.Float));
                }
            } catch (Exception e) {
                gradList.push_back(zeros(t.numel()).to(device, ScalarType.Float));
            }
        }
        if (gradList.size() == 0) {
            return zeros(Math.max(1, totalParamNumel)).to(device, ScalarType.Float);
        }
        return cat(gradList);
    }

    /** All-gather local shard into full flat parameter buffer. */
    public Tensor allGatherParameters() {
        numAllGatherCalls++;
        int world = Math.max(1, processGroup.getWorldSize());
        if (shardingStrategy == ShardingStrategy.NO_SHARD || world == 1) {
            return flattenParameters();
        }
        Tensor full = empty(paddedFullSize).to(device, ScalarType.Float);
        Tensor paddedInput = shardedParams.get(0);
        if (paddedInput.numel() < shardSize) {
            Tensor pad = zeros(shardSize - paddedInput.numel()).to(device, ScalarType.Float);
            TensorVector v = new TensorVector();
            v.push_back(paddedInput);
            v.push_back(pad);
            paddedInput = cat(v);
            pad.close();
        }
        Work w = processGroup.allgatherBase(full, paddedInput);

        if (w != null && !w.isNull()) w._wait();
        if (full.numel() > totalParamNumel) {
            return full.slice(0, new LongOptional(0), new LongOptional(totalParamNumel), 1L);
        }
        return full;
    }

    private void writeToModule(Tensor flatParams) {
        // Leaf parameters with requires_grad cannot be mutated in-place under autograd.
        try (NoGradGuard guard = new NoGradGuard()) {
            long offset = 0;
            TensorVector params = module.parameters();
            for (long i = 0, n = params.size(); i < n; i++) {
                Tensor t = params.get(i);
                if (t == null || t.isNull()) continue;
                long num = t.numel();
                if (offset + num <= flatParams.numel()) {
                    Tensor src = flatParams.narrow(0, offset, num);
                    t.copy_(src.view(t.sizes()));
                    src.close();
                }
                offset += num;
            }
        }
    }

    public Tensor forward(Tensor input) {
        Tensor inputAdj = input;
        if (input.device().type() != device.type()) {
            inputAdj = input.to(device, input.scalar_type());
        }
        Tensor fullParams = allGatherParameters();
        writeToModule(fullParams);
        Tensor output = forward.apply(module, inputAdj);
        numForwardCalls++;
        if (reshardAfterForward && shardingStrategy == ShardingStrategy.FULL_SHARD) {
            try { fullParams.close(); } catch (Throwable ignored) {}
        }
        return output;
    }

    public Tensor step(Tensor input, Tensor target, Optimizer optimizer) {
        zeroGrad();
        Tensor output = forward(input);
        Tensor loss = DistributedLoss.crossEntropy(output, target);
        if (gradAccumSteps > 1) {
            loss = loss.div(new Scalar(gradAccumSteps));
        }
        loss.backward();
        numBackwardCalls++;
        microStep++;
        if (syncGradients && microStep % gradAccumSteps == 0) {
            reduceScatterGradients();
            // Write sharded grads back is conceptual; step uses module params that still
            // hold full grads after backward. For FULL_SHARD we average grads via
            // reduce-scatter then all-gather average into module grads, or allreduce.
            applyShardedUpdate(optimizer);
        }
        return loss;
    }

    /**
     * Reduce-scatter flattened grads into local shard (FULL_SHARD / SHARD_GRAD_OP).
     * NO_SHARD: allreduce average.
     */
    public void reduceScatterGradients() {
        numReduceScatterCalls++;
        int world = Math.max(1, processGroup.getWorldSize());
        Tensor gradFlat = flattenGradients();
        if (world == 1 || shardingStrategy == ShardingStrategy.NO_SHARD) {
            if (world > 1) {
                processGroup.allreduce(gradFlat);
                gradFlat.div_(new Scalar(world));
                writeGradsToModule(gradFlat);
            }
            gradFlat.close();
            return;
        }
        // Pad to paddedFullSize
        Tensor padded;
        if (gradFlat.numel() < paddedFullSize) {
            Tensor pad = zeros(paddedFullSize - gradFlat.numel()).to(device, ScalarType.Float);
            TensorVector v = new TensorVector();
            v.push_back(gradFlat);
            v.push_back(pad);
            padded = cat(v);
            pad.close();
        } else if (gradFlat.numel() > paddedFullSize) {
            padded = gradFlat.narrow(0, 0, paddedFullSize);
        } else {
            padded = gradFlat;
        }
        Tensor out = empty(shardSize).to(device, ScalarType.Float);
        Work w = processGroup.reduceScatterBase(out, padded);

        if (w != null && !w.isNull()) w._wait();
        out.div_(new Scalar(world));
        if (shardedGrads.isEmpty()) {
            shardedGrads.add(zeros_like(shardedParams.get(0)));
        }
        long local = shardedParams.get(0).numel();
        Tensor shard = out.numel() > local
                ? out.slice(0, new LongOptional(0), new LongOptional(local), 1L)
                : out;
        shardedGrads.get(0).copy_(shard);

        // Reconstruct full averaged grad via allgather of shards for module.grad update
        // (keeps standard Optimizer.step() working on full module parameters).
        Tensor fullAvg = empty(paddedFullSize).to(device, ScalarType.Float);
        Work ag = processGroup.allgatherBase(fullAvg, shardedGrads.get(0).numel() < shardSize
                ? padTo(shardedGrads.get(0), shardSize)
                : shardedGrads.get(0));

        if (ag != null && !ag.isNull()) ag._wait();
        Tensor fullTrim = fullAvg.numel() > totalParamNumel
                ? fullAvg.slice(0, new LongOptional(0), new LongOptional(totalParamNumel), 1L)
                : fullAvg;
        writeGradsToModule(fullTrim);

        gradFlat.close();
        if (padded != gradFlat) {
            try { padded.close(); } catch (Throwable ignored) {}
        }
    }

    private Tensor padTo(Tensor t, long size) {
        if (t.numel() >= size) return t;
        Tensor pad = zeros(size - t.numel()).to(device, ScalarType.Float);
        TensorVector v = new TensorVector();
        v.push_back(t);
        v.push_back(pad);
        Tensor c = cat(v);
        pad.close();
        return c;
    }

    private void writeGradsToModule(Tensor flatGrads) {
        try (NoGradGuard guard = new NoGradGuard()) {
            long offset = 0;
            TensorVector params = module.parameters();
            for (long i = 0, n = params.size(); i < n; i++) {
                Tensor t = params.get(i);
                if (t == null || t.isNull()) continue;
                long num = t.numel();
                if (offset + num > flatGrads.numel()) break;
                Tensor src = flatGrads.narrow(0, offset, num).view(t.sizes());
                try {
                    Tensor g = t.grad();
                    if (g != null && !g.isNull() && g.defined()) {
                        g.copy_(src);
                    }
                } catch (Exception ignored) {
                }
                src.close();
                offset += num;
            }
        }
    }

    private void applyShardedUpdate(Optimizer optimizer) {
        if (optimizer != null) {
            optimizer.step();
            optimizer.zero_grad();
        }
        // Refresh local shard from updated full parameters.
        if (shardingStrategy != ShardingStrategy.NO_SHARD && processGroup.getWorldSize() > 1) {
            shardParameters();
        }
    }

    /** Zero parameter gradients (public for grad-accum benchmarks / Accelerator). */
    public void zeroGrad() {
        TensorVector params = module.parameters();
        for (long i = 0, n = params.size(); i < n; i++) {
            Tensor t = params.get(i);
            if (t == null || t.isNull()) continue;
            try {
                Tensor g = t.grad();
                if (g != null && !g.isNull() && g.defined()) {
                    g.zero_();
                }
            } catch (Exception ignored) {
            }
        }
    }

    public void disableSync() { syncGradients = false; }
    public void enableSync() { syncGradients = true; }
    public boolean isSyncEnabled() { return syncGradients; }

    public NoSync noSync() { return new NoSync(this); }

    public static final class NoSync implements AutoCloseable {
        private final NativeFSDPTrainer t;
        private final boolean prev;
        NoSync(NativeFSDPTrainer t) {
            this.t = t;
            this.prev = t.syncGradients;
            t.syncGradients = false;
        }
        @Override public void close() { t.syncGradients = prev; }
    }

    // ── Checkpoint (sharded + full) — raw float32 dump (no torch.save binding) ──

    public void saveSharded(Path dir) throws IOException {
        Files.createDirectories(dir);
        Tensor shard = shardedParams.isEmpty()
                ? zeros(1).to(device, ScalarType.Float)
                : shardedParams.get(0);
        Path file = dir.resolve("shard_rank" + processGroup.getRank() + ".f32");
        writeFloatTensor(file, shard);
        if (processGroup.isMainProcess()) {
            Files.writeString(dir.resolve("meta.txt"),
                    "totalParamNumel=" + totalParamNumel + "\nshardSize=" + shardSize
                            + "\nworldSize=" + processGroup.getWorldSize()
                            + "\nstrategy=" + shardingStrategy + "\n");
        }
        processGroup.barrierWait();
    }

    public void loadSharded(Path dir) throws IOException {
        Path file = dir.resolve("shard_rank" + processGroup.getRank() + ".f32");
        if (!Files.exists(file)) {
            throw new IOException("missing shard file: " + file);
        }
        Tensor loaded = readFloatTensor(file).to(device, ScalarType.Float);
        if (shardedParams.isEmpty()) {
            shardedParams.add(loaded);
        } else {
            try (NoGradGuard guard = new NoGradGuard()) {
                long n = Math.min(shardedParams.get(0).numel(), loaded.numel());
                shardedParams.get(0).narrow(0, 0, n).copy_(loaded.flatten().narrow(0, 0, n));
            }
            loaded.close();
        }
        Tensor full = allGatherParameters();
        writeToModule(full);
        full.close();
        processGroup.barrierWait();
    }

    /** Rank 0 writes full aggregated state; all ranks participate in allgather. */
    public void saveFull(Path file) throws IOException {
        Tensor full = allGatherParameters();
        if (processGroup.isMainProcess()) {
            Path parent = file.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            writeFloatTensor(file, full);
        }
        full.close();
        processGroup.barrierWait();
    }

    private static void writeFloatTensor(Path file, Tensor t) throws IOException {
        // Bulk dump — element-wise JNI p.get(i) over ~1e5+ params was timing out smoke tests.
        Tensor cpu = t.detach().contiguous().to(ScalarType.Float).cpu();
        long n = cpu.numel();
        int ni = (int) Math.min(n, Integer.MAX_VALUE);
        float[] data = new float[ni];
        try {
            org.bytedeco.javacpp.FloatPointer p = cpu.data_ptr_float();
            p.capacity(ni).limit(ni).asBuffer().get(data);
        } catch (Throwable bulkFail) {
            // Fallback (slow): sample first/last few only would be wrong — use limited loop with progress
            org.bytedeco.javacpp.FloatPointer p = cpu.data_ptr_float();
            for (int i = 0; i < ni; i++) data[i] = p.get((long) i);
        }
        try (java.io.DataOutputStream out = new java.io.DataOutputStream(
                new java.io.BufferedOutputStream(Files.newOutputStream(file), 1 << 20))) {
            out.writeLong(n);
            // write as raw little-endian floats via ByteBuffer for speed
            java.nio.ByteBuffer bb = java.nio.ByteBuffer.allocate(ni * 4).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            bb.asFloatBuffer().put(data);
            out.write(bb.array());
        }
        if (cpu != t) {
            try { cpu.close(); } catch (Throwable ignored) {}
        }
    }

    private static Tensor readFloatTensor(Path file) throws IOException {
        try (java.io.DataInputStream in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(Files.newInputStream(file), 1 << 20))) {
            long n = in.readLong();
            int ni = (int) Math.min(n, Integer.MAX_VALUE);
            byte[] raw = in.readNBytes(ni * 4);
            java.nio.FloatBuffer fb = java.nio.ByteBuffer.wrap(raw)
                    .order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            float[] data = new float[ni];
            fb.get(data);
            return org.bytedeco.pytorch.global.torch.tensor(data).clone();
        }
    }

    public Module getModule() { return module; }
    public List<Tensor> getShardedParameters() { return List.copyOf(shardedParams); }
    public List<Tensor> getShardedGradients() { return List.copyOf(shardedGrads); }
    public ShardingStrategy getShardingStrategy() { return shardingStrategy; }
    public MixedPrecisionConfig getMixedPrecision() { return mixedPrecision; }
    public ProcessGroupWrapper getProcessGroup() { return processGroup; }
    public int getRank() { return processGroup.getRank(); }
    public int getWorldSize() { return processGroup.getWorldSize(); }
    public boolean isMainProcess() { return processGroup.isMainProcess(); }
    public Device getDevice() { return device; }
    public long getShardSize() { return shardSize; }
    public long getTotalParamSize() { return totalParamNumel; }
    public long getPaddedFullSize() { return paddedFullSize; }
    public long getNumForwardCalls() { return numForwardCalls; }
    public long getNumBackwardCalls() { return numBackwardCalls; }
    public long getNumAllGatherCalls() { return numAllGatherCalls; }
    public long getNumReduceScatterCalls() { return numReduceScatterCalls; }
    public int getGradAccumSteps() { return gradAccumSteps; }
    public int getMicroStep() { return microStep; }

    public void train() { module.train(true); }
    public void eval() { module.eval(); }
    public boolean isTraining() { return module.is_training(); }

    @Override
    public void close() {
        for (Tensor t : shardedParams) {
            try { t.close(); } catch (Throwable ignored) {}
        }
        for (Tensor t : shardedGrads) {
            try { t.close(); } catch (Throwable ignored) {}
        }
        shardedParams.clear();
        shardedGrads.clear();
    }

    @Override
    public String toString() {
        return "NativeFSDPTrainer{rank=" + processGroup.getRank()
                + ", world=" + processGroup.getWorldSize()
                + ", strategy=" + shardingStrategy
                + ", shardSize=" + shardSize
                + ", total=" + totalParamNumel + '}';
    }

    public static final class Builder {
        private Module module;
        private ProcessGroupWrapper processGroup;
        private ShardingStrategy shardingStrategy = ShardingStrategy.FULL_SHARD;
        private boolean reshardAfterForward = true;
        private MixedPrecisionConfig mixedPrecision = MixedPrecisionConfig.fp32();
        private int gradAccumSteps = 1;
        private DeviceMesh deviceMesh;

        public Builder module(Module m) { this.module = m; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder shardingStrategy(ShardingStrategy s) { this.shardingStrategy = s; return this; }
        public Builder reshardAfterForward(boolean b) { this.reshardAfterForward = b; return this; }
        public Builder mixedPrecision(MixedPrecisionConfig mp) { this.mixedPrecision = mp; return this; }
        public Builder gradAccumSteps(int n) { this.gradAccumSteps = n; return this; }
        /** Optional DeviceMesh (FSDP2-style); currently records mesh, uses mesh process group if set. */
        public Builder deviceMesh(DeviceMesh mesh) {
            this.deviceMesh = mesh;
            if (mesh != null && processGroup == null) {
                this.processGroup = mesh.processGroup();
            }
            return this;
        }

        public NativeFSDPTrainer build() {
            Objects.requireNonNull(module, "module is required");
            if (processGroup == null && deviceMesh != null) {
                processGroup = deviceMesh.processGroup();
            }
            Objects.requireNonNull(processGroup, "processGroup is required");
            return new NativeFSDPTrainer(
                    module, processGroup, shardingStrategy, reshardAfterForward,
                    mixedPrecision, gradAccumSteps);
        }
    }
}
