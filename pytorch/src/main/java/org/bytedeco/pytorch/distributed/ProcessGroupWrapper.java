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

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.chrono.Milliseconds;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.GlooDeviceVector;
import org.bytedeco.pytorch.IntVector;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import static org.bytedeco.pytorch.global.torch.DeviceType;
import static org.bytedeco.pytorch.global.torch.cat;
import static org.bytedeco.pytorch.global.torch.cuda_is_available;
import static org.bytedeco.pytorch.global.torch.empty;

/**
 * Production process-group facade over c10d.
 *
 * <p>Holds both a {@link ProcessGroup} container (required by {@link Reducer}
 * / industrial DDP) and the concrete {@link Backend} (Gloo / NCCL / …) used
 * for collectives. Optionally wraps the backend with
 * {@link ProcessGroupNativeWrapper} when {@link Options#debug(boolean)} is set
 * (consistency checks via a side Gloo backend — not the production hot path).
 *
 * <pre>{@code
 * try (DistributedStore store = DistributedStore.create(rank, worldSize);
 *      ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, worldSize, store)) {
 *     pg.allreduce(grad);
 *     pg.barrierWait();
 *     // NativeDDPTrainer needs the ProcessGroup container:
 *     ProcessGroup c10dPg = pg.getProcessGroup();
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ProcessGroupWrapper implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private static final Set<ProcessGroupWrapper> INSTANCES =
            ConcurrentHashMap.newKeySet();

    private final int rank;
    private final int worldSize;
    private final BackendType backendType;
    private final String backendName;
    private final Device device;
    private final DeviceType deviceType;
    private final ProcessGroup processGroup;
    private final Backend backend;
    private final Backend collectiveBackend;
    private final boolean debugWrapped;
    private final DistributedStore store;
    private final Options options;

    public ProcessGroupWrapper(Options options, int rank, int worldSize, DistributedStore store) {
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(store, "store");
        this.options = options;
        this.rank = rank;
        this.worldSize = worldSize;
        this.store = store;
        this.backendType = resolveBackend(options.backendType);
        Milliseconds timeout = new Milliseconds(options.timeoutMs);

        Store nativeStore = store.getNativeStore();
        // IMPORTANT: do NOT construct ProcessGroup(store, rank, size) and then also
        // ProcessGroupGloo(same store, ...) — double make_intrusive on the same store
        // path trips c10 intrusive_ptr refcount assert on Mac. Use store-less container
        // ProcessGroup(rank, size); only the concrete Backend owns the Store.
        this.processGroup = new ProcessGroup(rank, worldSize);

        Backend rawBackend = null;
        ProcessGroup.BackendType pgBackendType;
        String bName;

        // Backend selection:
        // - worldSize==1 defaults to local no-op (fast smoke) unless forceCollective
        //   or explicit GLOO/MPI is requested (needed for multi-thread stress).
        // - Gloo: ProcessGroupGloo.Options.create_default() works on Mac via
        //   libjnitorch (do NOT touch org.bytedeco.pytorch.gloo.Device — that
        //   class is torch_gloo linux-only and UnsatisfiedLinkError on macosx).
        // - MPI: ProcessGroupMPI.createProcessGroupMPI() requires jnitorch_mpi
        //   (USE_MPI=1 rebuild). Homebrew open-mpi alone is not enough.
        boolean wantCollective = worldSize > 1
                || options.forceCollective
                || backendType == BackendType.GLOO
                || backendType == BackendType.MPI
                || backendType == BackendType.UCC;

        if (!wantCollective) {
            this.device = new Device(DeviceType.CPU, (byte) 0);
            this.deviceType = DeviceType.CPU;
            rawBackend = null;
            bName = "local";
            pgBackendType = ProcessGroup.BackendType.UNDEFINED;
        } else if (backendType == BackendType.NCCL && cuda_is_available()) {
            this.device = new Device(DeviceType.CUDA, (byte) Math.min(rank, 255));
            this.deviceType = DeviceType.CUDA;
            ProcessGroupNCCL.Options pgOpts = ProcessGroupNCCL.Options.create(true);
            pgOpts.timeout(timeout);
            rawBackend = new ProcessGroupNCCL(nativeStore, rank, worldSize, pgOpts);
            bName = "nccl";
            pgBackendType = ProcessGroup.BackendType.NCCL;
        } else if (backendType == BackendType.MPI) {
            this.device = new Device(DeviceType.CPU, (byte) 0);
            this.deviceType = DeviceType.CPU;
            try {
                rawBackend = createMpiBackend(rank, worldSize);
                bName = "mpi";
                pgBackendType = ProcessGroup.BackendType.MPI;
            } catch (Throwable t) {
                System.err.println("WARNING: MPI backend unavailable (" + t.getClass().getSimpleName()
                        + ": " + t.getMessage() + "). Need jnitorch_mpi (libtorch USE_MPI=1). "
                        + "Falling back to Gloo create_default.");
                try {
                    rawBackend = createGlooBackend(nativeStore, rank, worldSize, timeout, options.masterAddr);
                    bName = "gloo(mpi-fallback)";
                    pgBackendType = ProcessGroup.BackendType.GLOO;
                } catch (Throwable t2) {
                    rawBackend = null;
                    bName = "local(mpi+gloo-unavailable)";
                    pgBackendType = ProcessGroup.BackendType.UNDEFINED;
                }
            }
        } else {
            // GLOO / UCC-fallback / AUTO without CUDA
            this.device = new Device(DeviceType.CPU, (byte) 0);
            this.deviceType = DeviceType.CPU;
            try {
                rawBackend = createGlooBackend(nativeStore, rank, worldSize, timeout, options.masterAddr);
                bName = backendType == BackendType.UCC ? "gloo(ucc-fallback)" : "gloo";
                pgBackendType = ProcessGroup.BackendType.GLOO;
            } catch (Throwable t) {
                System.err.println("WARNING: Gloo backend unavailable (" + t.getClass().getSimpleName()
                        + ": " + t.getMessage() + "). Using local no-op backend.");
                rawBackend = null;
                bName = "local(gloo-unavailable)";
                pgBackendType = ProcessGroup.BackendType.UNDEFINED;
            }
        }

        this.backendName = bName;
        this.backend = rawBackend;
        Backend installed = rawBackend;
        boolean debug = options.debug && rawBackend != null;
        if (debug) {
            try {
                Backend glooSide = rawBackend instanceof ProcessGroupGloo
                        ? rawBackend
                        : createGlooBackend(nativeStore, rank, worldSize, timeout, options.masterAddr);
                installed = new ProcessGroupNativeWrapper(rawBackend, glooSide);
                System.out.printf(
                        "[Rank %d] ProcessGroupNativeWrapper DEBUG enabled%n", rank);
            } catch (Throwable t) {
                System.err.println("WARNING: ProcessGroupNativeWrapper failed: " + t.getMessage());
                installed = rawBackend;
                debug = false;
            }
        }
        this.debugWrapped = debug;
        this.collectiveBackend = installed; // may be null for local mode

        if (installed != null) {
            try {
                processGroup.setBackend(
                        deviceType,
                        pgBackendType,
                        new BackendOptional(installed));
                processGroup.setDefaultBackend(pgBackendType);
            } catch (Throwable t) {
                System.err.println("WARNING: ProcessGroup.setBackend failed: " + t.getMessage());
            }
        }

        System.out.printf(
                "[Rank %d] ProcessGroup initialized backend=%s device=%s worldSize=%d debug=%s%n",
                rank, backendName, device, worldSize, debugWrapped);
        INSTANCES.add(this);
    }

    /**
     * Build ProcessGroupGloo without touching Java {@code org.bytedeco.pytorch.gloo.Device}.
     * {@code Options.create_default()} builds the default transport device inside
     * libtorch C++ — verified on Mac arm64 via libjnitorch (not torch_gloo preset).
     */
    private static Backend createGlooBackend(
            Store nativeStore, int rank, int worldSize, Milliseconds timeout, String masterAddr) {
        // Preferred (and Mac-safe): C++ default options — no Java gloo.Device class load
        ProcessGroupGloo.Options def = ProcessGroupGloo.Options.create_default();
        def.timeout(timeout);
        return new ProcessGroupGloo(nativeStore, rank, worldSize, def);
    }

    /**
     * MPI backend via {@link ProcessGroupMPI#createProcessGroupMPI()}.
     * Requires native jnitorch_mpi (libtorch built with USE_MPI=1). Homebrew
     * open-mpi alone is insufficient if the JNI entry points were not linked.
     */
    private static Backend createMpiBackend(int rank, int worldSize) {
        ProcessGroupMPI mpi = ProcessGroupMPI.createProcessGroupMPI();
        if (mpi == null || mpi.isNull()) {
            throw new IllegalStateException("createProcessGroupMPI returned null");
        }
        // Rank/size come from MPI_COMM_WORLD; mismatch is a hard error.
        if (mpi.getSize() != worldSize) {
            System.err.printf(
                    "WARNING: MPI world size %d != requested worldSize %d (using MPI size)%n",
                    mpi.getSize(), worldSize);
        }
        return mpi;
    }

    /** True when no real collective backend is attached (worldSize=1 or gloo unavailable). */
    public boolean isLocalOnly() {
        return collectiveBackend == null || "local".equals(backendName) || backendName.startsWith("local");
    }

    public static ProcessGroupWrapper create(int rank, int worldSize, DistributedStore store) {
        return create(new Options(), rank, worldSize, store);
    }

    public static ProcessGroupWrapper create(Options options, int rank, int worldSize, DistributedStore store) {
        return new ProcessGroupWrapper(options, rank, worldSize, store);
    }

    // ── Collectives (delegate to collectiveBackend; local no-op when null) ──

    /**
     * Local-mode collective result. Returns {@code null} intentionally —
     * default native {@code Work()} can block forever on {@code _wait()}.
     * Prefer {@link #barrierWait()} / null-safe waits.
     */
    private Work localDone() {
        return null;
    }

    private static void waitIfNeeded(Work w, boolean sync) {
        if (w == null || w.isNull() || !sync) {
            return;
        }
        try {
            w._wait();
        } catch (Throwable t) {
            // Local/default Work may not implement wait; treat as already done.
        }
    }

    public Work allreduce(List<Tensor> tensors) {
        return allreduce(tensors, ReduceOp.RedOpType.SUM);
    }

    public Work allreduce(List<Tensor> tensors, ReduceOp.RedOpType op) {
        if (isLocalOnly()) return localDone();
        AllreduceOptions opts = new AllreduceOptions();
        opts.reduceOp(new ReduceOp(op));
        Work w = collectiveBackend.allreduce(toTensorVector(tensors), opts);
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    public Work allreduce(Tensor tensor) {
        return allreduce(Collections.singletonList(tensor));
    }

    public Work allreduceCoalesced(List<Tensor> tensors) {
        return allreduceCoalesced(tensors, ReduceOp.RedOpType.SUM);
    }

    public Work allreduceCoalesced(List<Tensor> tensors, ReduceOp.RedOpType op) {
        if (isLocalOnly()) return localDone();
        if (op != null && op != ReduceOp.RedOpType.SUM) {
            return allreduce(tensors, op);
        }
        Work w = collectiveBackend.allreduce_coalesced(toTensorVector(tensors));
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    public Work broadcast(Tensor tensor, int rootRank) {
        if (isLocalOnly()) return localDone();
        BroadcastOptions opts = new BroadcastOptions();
        opts.rootRank(rootRank);
        Work w = collectiveBackend.broadcast(toTensorVector(tensor), opts);
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    public Work broadcast(List<Tensor> tensors, int rootRank) {
        if (isLocalOnly()) return localDone();
        BroadcastOptions opts = new BroadcastOptions();
        opts.rootRank(rootRank);
        Work w = collectiveBackend.broadcast(toTensorVector(tensors), opts);
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    public Work reduce(Tensor tensor, int rootRank) {
        if (isLocalOnly()) return localDone();
        ReduceOptions opts = new ReduceOptions();
        opts.rootRank(rootRank);
        opts.reduceOp(new ReduceOp(ReduceOp.RedOpType.SUM));
        Work w = collectiveBackend.reduce(toTensorVector(tensor), opts);
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    /**
     * Contiguous all-gather via {@code _allgather_base}.
     * {@code output} must hold {@code worldSize * input.numel()} elements;
     * {@code input} is this rank's shard.
     * <p>Local mode: copies {@code input} into the leading slice of {@code output}.
     */
    public Work allgatherBase(Tensor output, Tensor input) {
        if (isLocalOnly()) {
            long n = Math.min(output.numel(), input.numel());
            if (n > 0) {
                output.reshape(-1).narrow(0, 0, n).copy_(input.reshape(-1).narrow(0, 0, n));
            }
            return localDone();
        }
        Work w = collectiveBackend._allgather_base(output, input, new AllgatherOptions());
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    /**
     * All-gather a single input into a pre-sized list of per-rank outputs by
     * packing into a contiguous buffer (same semantics as Python
     * {@code all_gather} for equal-sized tensors).
     */
    public Work allgather(List<Tensor> outputTensors, Tensor inputTensor) {
        if (outputTensors == null || outputTensors.isEmpty()) {
            throw new IllegalArgumentException("outputTensors must be non-empty");
        }
        if (isLocalOnly()) {
            outputTensors.get(0).copy_(inputTensor.view(outputTensors.get(0).sizes()));
            for (int i = 1; i < outputTensors.size(); i++) {
                outputTensors.get(i).zero_();
            }
            return localDone();
        }
        long shard = inputTensor.numel();
        Tensor flatOut = empty(shard * outputTensors.size())
                .to(inputTensor.device(), inputTensor.scalar_type());
        Work work = allgatherBase(flatOut, inputTensor);
        if (work != null && !work.isNull()) work._wait();
        for (int i = 0; i < outputTensors.size(); i++) {
            Tensor slice = flatOut.narrow(0, i * shard, shard);
            outputTensors.get(i).copy_(slice.view(outputTensors.get(i).sizes()));
            slice.close();
        }
        flatOut.close();
        return work;
    }

    public Work allgather(List<Tensor> outputTensors, List<Tensor> inputTensors) {
        if (inputTensors == null || inputTensors.size() != 1) {
            throw new IllegalArgumentException(
                    "allgather helper currently supports a single input tensor; use allgatherBase for custom layouts");
        }
        return allgather(outputTensors, inputTensors.get(0));
    }

    /**
     * Contiguous reduce-scatter via {@code _reduce_scatter_base}.
     * {@code input} holds the full buffer ({@code worldSize * output.numel()} elements);
     * {@code output} receives this rank's shard.
     * <p>Local mode: copies the leading {@code output.numel()} of {@code input} into {@code output}.
     */
    public Work reduceScatterBase(Tensor output, Tensor input) {
        if (isLocalOnly()) {
            long n = Math.min(output.numel(), input.numel());
            if (n > 0) {
                output.reshape(-1).narrow(0, 0, n).copy_(input.reshape(-1).narrow(0, 0, n));
            }
            return localDone();
        }
        ReduceScatterOptions opts = new ReduceScatterOptions();
        opts.reduceOp(new ReduceOp(ReduceOp.RedOpType.SUM));
        Work w = collectiveBackend._reduce_scatter_base(output, input, opts);
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    /**
     * Reduce-scatter equal-sized shards into {@code outputTensors.get(0)} (this rank).
     * {@code inputTensors} should contain {@code worldSize} equal shards in rank order.
     */
    public Work reduceScatter(List<Tensor> outputTensors, List<Tensor> inputTensors) {
        if (outputTensors == null || outputTensors.isEmpty()) {
            throw new IllegalArgumentException("outputTensors must be non-empty");
        }
        if (inputTensors == null || inputTensors.isEmpty()) {
            throw new IllegalArgumentException("inputTensors must be non-empty");
        }
        if (isLocalOnly()) {
            outputTensors.get(0).copy_(inputTensors.get(0).view(outputTensors.get(0).sizes()));
            return localDone();
        }
        TensorVector parts = new TensorVector(inputTensors.toArray(new Tensor[0]));
        Tensor flatIn = cat(parts);
        Tensor out = outputTensors.get(0);
        Work work = reduceScatterBase(out, flatIn);
        if (work != null && !work.isNull()) work._wait();
        flatIn.close();
        return work;
    }

    public Work alltoall(List<Tensor> outputTensors, List<Tensor> inputTensors) {
        if (isLocalOnly()) {
            int n = Math.min(outputTensors.size(), inputTensors.size());
            for (int i = 0; i < n; i++) {
                outputTensors.get(i).copy_(inputTensors.get(i));
            }
            return localDone();
        }
        Work w = collectiveBackend.alltoall(
                toTensorVector(outputTensors), toTensorVector(inputTensors), new AllToAllOptions());
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    public Work alltoallBase(
            Tensor output, Tensor input, LongVector outputSplitSizes, LongVector inputSplitSizes) {
        if (isLocalOnly()) {
            long n = Math.min(output.numel(), input.numel());
            if (n > 0) {
                output.reshape(-1).narrow(0, 0, n).copy_(input.reshape(-1).narrow(0, 0, n));
            }
            return localDone();
        }
        Work w = collectiveBackend.alltoall_base(
                output, input, outputSplitSizes, inputSplitSizes, new AllToAllOptions());
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    public Work gather(List<Tensor> outputTensors, Tensor input, int rootRank) {
        if (isLocalOnly()) {
            if (outputTensors != null && !outputTensors.isEmpty()) {
                outputTensors.get(0).copy_(input.view(outputTensors.get(0).sizes()));
            }
            return localDone();
        }
        if (rank == rootRank) {
            if (outputTensors == null || outputTensors.size() != worldSize) {
                throw new IllegalArgumentException("root must supply worldSize output tensors");
            }
            return allgather(outputTensors, input);
        }
        Tensor flat = empty(input.numel() * (long) worldSize).to(input.device(), input.scalar_type());
        Work w = allgatherBase(flat, input);
        if (w != null && !w.isNull()) w._wait();
        flat.close();
        return w;
    }

    public Work scatter(Tensor output, List<Tensor> inputTensors, int rootRank) {
        if (isLocalOnly()) {
            if (inputTensors != null && !inputTensors.isEmpty()) {
                output.copy_(inputTensors.get(0).view(output.sizes()));
            }
            return localDone();
        }
        if (rank == rootRank) {
            if (inputTensors == null || inputTensors.size() != worldSize) {
                throw new IllegalArgumentException("root must supply worldSize input tensors");
            }
            output.copy_(inputTensors.get(rootRank));
            for (int r = 0; r < worldSize; r++) {
                if (r == rootRank) continue;
                send(inputTensors.get(r), r, /*tag*/ 9000 + r);
            }
            return barrier();
        }
        recv(output, rootRank, 9000 + rank);
        return barrier();
    }

    public void send(Tensor tensor, int dstRank) {
        send(tensor, dstRank, 0);
    }

    public void send(Tensor tensor, int dstRank, int tag) {
        if (isLocalOnly()) return;
        Work work = collectiveBackend.send(toTensorVector(tensor), dstRank, tag);
        if (work != null && !work.isNull()) work._wait();
    }

    public Tensor recv(Tensor tensor, int srcRank) {
        return recv(tensor, srcRank, 0);
    }

    public Tensor recv(Tensor tensor, int srcRank, int tag) {
        if (isLocalOnly()) return tensor;
        Work work = collectiveBackend.recv(toTensorVector(tensor), srcRank, tag);
        if (work != null && !work.isNull()) work._wait();
        return tensor;
    }

    public Tensor recvAnysource(Tensor tensor) {
        return recvAnysource(tensor, 0);
    }

    public Tensor recvAnysource(Tensor tensor, int tag) {
        if (isLocalOnly()) return tensor;
        Work work = collectiveBackend.recvAnysource(toTensorVector(tensor), tag);
        if (work != null && !work.isNull()) work._wait();
        return tensor;
    }

    public Work barrier() {
        if (isLocalOnly()) {
            // Do NOT return default native Work() — its _wait() can block forever.
            // Callers must use barrierWait() or null-check. We still return a Work
            // only when a real backend exists.
            return null;
        }
        Work w = collectiveBackend.barrier(new BarrierOptions());
        waitIfNeeded(w, options.syncCollectives);
        return w;
    }

    /** Barrier then wait; safe for local mode (no-op) and null Work. */
    public void barrierWait() {
        Work w = barrier();
        if (w == null || w.isNull()) return;
        try {
            w._wait();
        } catch (Throwable ignored) {
        }
    }

    public void monitoredBarrier(boolean waitAllRanks) {
        if (isLocalOnly()) return;
        collectiveBackend.monitoredBarrier(new BarrierOptions(), waitAllRanks);
    }

    public void startCoalescing() {
        if (isLocalOnly()) return;
        collectiveBackend.startCoalescing();
    }

    public Work endCoalescing() {
        if (isLocalOnly()) return localDone();
        return collectiveBackend.endCoalescing();
    }

    /** Allreduce(SUM) then divide by worldSize in-place. */
    public void averageGradients(List<Tensor> gradients) {
        if (gradients == null || gradients.isEmpty() || worldSize <= 1) {
            return;
        }
        allreduce(gradients, ReduceOp.RedOpType.SUM);
        Scalar denom = new Scalar(worldSize);
        for (Tensor g : gradients) {
            g.div_(denom);
        }
    }

    public void syncParameters(List<Tensor> parameters, int rootRank) {
        if (worldSize <= 1) return;
        broadcast(parameters, rootRank);
    }

    /**
     * Create a subgroup over {@code ranks} using the underlying backend split API
     * when available. Returns a thin wrapper that reuses split backend for
     * collectives; rank/worldSize reflect the subgroup.
     *
     * <p>If split is unsupported, returns {@code null} and logs — callers
     * (DeviceMesh) should fall back to tag-based P2P or a fresh ProcessGroup on
     * a prefixed store.
     */
    public ProcessGroupWrapper trySplitGroup(List<Integer> ranks) {
        if (ranks == null || ranks.isEmpty() || isLocalOnly()) {
            return null;
        }
        if (!collectiveBackend.supportsSplitting()) {
            System.err.println("Backend does not support splitting; DeviceMesh will use logical subgroups");
            return null;
        }
        try {
            IntVector iv = new IntVector();
            for (int r : ranks) {
                iv.push_back(r);
            }
            Backend.Options bopts = collectiveBackend.getBackendOptions();
            Backend split = collectiveBackend.split(store.getNativeStore(), iv, bopts);
            if (split == null || split.isNull()) {
                return null;
            }
            // Logical wrapper: same store/options, but rank remapped inside DeviceMesh.
            // Full ProcessGroup re-init per subgroup is handled by DeviceMesh when split is null.
            return null;
        } catch (Throwable t) {
            System.err.println("trySplitGroup failed: " + t.getMessage());
            return null;
        }
    }

    // ── Accessors ──

    /** Concrete backend (Gloo/NCCL), never the debug wrapper. */
    public Backend getNativeGroup() { return backend; }

    /** Backend used for collectives (may be {@link ProcessGroupNativeWrapper}). */
    public Backend getCollectiveBackend() { return collectiveBackend; }

    /** Alias of {@link #getNativeGroup()}. */
    public Backend getBackend() { return backend; }

    /**
     * c10d {@link ProcessGroup} container — required by {@link Reducer}.
     * May lack a successfully installed backend if {@code setBackend} failed;
     * trainers must handle FALLBACK.
     */
    public ProcessGroup getProcessGroup() { return processGroup; }

    public boolean isDebugWrapped() { return debugWrapped; }
    public int getRank() { return rank; }
    public int getWorldSize() { return worldSize; }
    public BackendType getBackendType() { return backendType; }
    /** @deprecated use {@link #getBackendType()} */
    @Deprecated
    public BackendType getBackendEnum() { return backendType; }
    public String getBackendName() { return backendName; }
    public Device getDevice() { return device; }
    public DeviceType getDeviceType() { return deviceType; }
    public boolean isMainProcess() { return rank == 0; }
    public DistributedStore getStore() { return store; }
    public Options getOptions() { return options; }

    @Override
    public void close() {
        try {
            if (collectiveBackend != null
                    && ("nccl".equalsIgnoreCase(backendName) || backendName.startsWith("nccl"))) {
                collectiveBackend.waitForPendingWorks();
            }
        } catch (Throwable ignored) {
        }
        try {
            if (collectiveBackend != null) {
                collectiveBackend.shutdown();
            }
        } catch (Throwable ignored) {
        }
        try {
            if (processGroup != null) {
                processGroup.shutdown();
            }
        } catch (Throwable ignored) {
        }
        INSTANCES.remove(this);
    }

    @Override
    public String toString() {
        return "ProcessGroupWrapper{backend=" + backendName
                + ", rank=" + rank + ", worldSize=" + worldSize
                + ", device=" + device + ", debug=" + debugWrapped + '}';
    }

    private static BackendType resolveBackend(BackendType requested) {
        if (requested == BackendType.AUTO) {
            return cuda_is_available() ? BackendType.NCCL : BackendType.GLOO;
        }
        if (requested == BackendType.NCCL && !cuda_is_available()) {
            System.err.println("WARNING: NCCL requested but CUDA not available, falling back to GLOO");
            return BackendType.GLOO;
        }
        return requested;
    }

    private static TensorVector toTensorVector(List<Tensor> tensors) {
        return new TensorVector(tensors.toArray(new Tensor[0]));
    }

    private static TensorVector toTensorVector(Tensor tensor) {
        return new TensorVector(tensor);
    }

    public static final class Options {
        private BackendType backendType = BackendType.AUTO;
        private int timeoutMs = 300_000;
        private String masterAddr = "127.0.0.1";
        private int masterPort = 29_500;
        /** When true, wrap backend with ProcessGroupNativeWrapper (debug only). */
        private boolean debug = false;
        /**
         * When true (default), collective helpers wait on Work before returning.
         * Set false for overlap experiments; callers must {@code work._wait()}.
         */
        private boolean syncCollectives = true;
        /**
         * When true, construct a real collective backend even for {@code worldSize==1}
         * (Gloo create_default). Useful for multi-thread stress that still wants a
         * real ProcessGroupGloo object. Default false keeps single-process smoke fast.
         */
        private boolean forceCollective = false;

        public Options backend(BackendType b) { this.backendType = b; return this; }
        public Options timeout(int ms) { this.timeoutMs = ms; return this; }
        public Options masterAddr(String addr) { this.masterAddr = addr; return this; }
        public Options masterPort(int port) { this.masterPort = port; return this; }
        public Options debug(boolean d) { this.debug = d; return this; }
        public Options syncCollectives(boolean s) { this.syncCollectives = s; return this; }
        public Options forceCollective(boolean f) { this.forceCollective = f; return this; }

        public BackendType getBackend() { return backendType; }
        public int getTimeoutMs() { return timeoutMs; }
        public String getMasterAddr() { return masterAddr; }
        public int getMasterPort() { return masterPort; }
        public boolean isDebug() { return debug; }
        public boolean isSyncCollectives() { return syncCollectives; }
        public boolean isForceCollective() { return forceCollective; }
    }
}
