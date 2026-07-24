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
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.nccl.ProcessGroupNCCL;

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
 * High-level wrapper around {@link Backend} (Gloo / NCCL) that exposes
 * common collectives with Java-friendly overloads.
 *
 * <pre>{@code
 * try (DistributedStore store = DistributedStore.create(rank, worldSize);
 *      ProcessGroupWrapper pg = ProcessGroupWrapper.create(rank, worldSize, store)) {
 *     pg.allreduce(grad);
 *     pg.barrier()._wait();
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
    private final Backend processGroup;

    public ProcessGroupWrapper(Options options, int rank, int worldSize, DistributedStore store) {
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(store, "store");
        this.rank = rank;
        this.worldSize = worldSize;
        this.backendType = resolveBackend(options.backendType);
        Milliseconds timeout = new Milliseconds(options.timeoutMs);

        if (backendType == BackendType.NCCL && cuda_is_available()) {
            this.device = new Device(DeviceType.CUDA, (byte) rank);
            ProcessGroupNCCL.Options pgOpts = ProcessGroupNCCL.Options.create(true);
            pgOpts.timeout(timeout);
            this.processGroup = new ProcessGroupNCCL(store.getNativeStore(), rank, worldSize, pgOpts);
            this.backendName = "nccl";
        } else {
            this.device = new Device(DeviceType.CPU, (byte) 0);
            ProcessGroupGloo.Options pgOpts = ProcessGroupGloo.Options.create();
            pgOpts.timeout(timeout);
            GlooDeviceVector devices = new GlooDeviceVector();
            devices.push_back(ProcessGroupGloo.createDeviceForHostname(options.masterAddr));
            pgOpts.devices(devices);
            this.processGroup = new ProcessGroupGloo(store.getNativeStore(), rank, worldSize, pgOpts);
            this.backendName = "gloo";
        }
        System.out.printf("[Rank %d] ProcessGroup initialized with backend=%s, device=%s%n",
                rank, backendName, device);
        INSTANCES.add(this);
    }

    public static ProcessGroupWrapper create(int rank, int worldSize, DistributedStore store) {
        return create(new Options(), rank, worldSize, store);
    }

    public static ProcessGroupWrapper create(Options options, int rank, int worldSize, DistributedStore store) {
        return new ProcessGroupWrapper(options, rank, worldSize, store);
    }

    public Work allreduce(List<Tensor> tensors) {
        return allreduce(tensors, ReduceOp.RedOpType.SUM);
    }

    public Work allreduce(List<Tensor> tensors, ReduceOp.RedOpType op) {
        AllreduceOptions opts = new AllreduceOptions();
        opts.reduceOp(new ReduceOp(op));
        return processGroup.allreduce(toTensorVector(tensors), opts);
    }

    public Work allreduce(Tensor tensor) {
        return allreduce(Collections.singletonList(tensor));
    }

    public Work broadcast(Tensor tensor, int rootRank) {
        BroadcastOptions opts = new BroadcastOptions();
        opts.rootRank(rootRank);
        return processGroup.broadcast(toTensorVector(tensor), opts);
    }

    public Work broadcast(List<Tensor> tensors, int rootRank) {
        BroadcastOptions opts = new BroadcastOptions();
        opts.rootRank(rootRank);
        return processGroup.broadcast(toTensorVector(tensors), opts);
    }

    /**
     * Contiguous all-gather via {@code _allgather_base}.
     * {@code output} must hold {@code worldSize * input.numel()} elements;
     * {@code input} is this rank's shard.
     *
     * <p>Preferred over the nested {@code vector<vector<Tensor>>} allgather,
     * which JavaCPP maps poorly for helper code.
     */
    public Work allgatherBase(Tensor output, Tensor input) {
        return processGroup._allgather_base(output, input, new AllgatherOptions());
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
        long shard = inputTensor.numel();
        Tensor flatOut = empty(shard * outputTensors.size())
                .to(inputTensor.device(), inputTensor.scalar_type());
        Work work = allgatherBase(flatOut, inputTensor);
        work._wait();
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
     */
    public Work reduceScatterBase(Tensor output, Tensor input) {
        ReduceScatterOptions opts = new ReduceScatterOptions();
        opts.reduceOp(new ReduceOp(ReduceOp.RedOpType.SUM));
        return processGroup._reduce_scatter_base(output, input, opts);
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
        TensorVector parts = new TensorVector(inputTensors.toArray(new Tensor[0]));
        Tensor flatIn = cat(parts);
        Tensor out = outputTensors.get(0);
        Work work = reduceScatterBase(out, flatIn);
        work._wait();
        flatIn.close();
        return work;
    }

    public void send(Tensor tensor, int dstRank) {
        send(tensor, dstRank, 0);
    }

    public void send(Tensor tensor, int dstRank, int tag) {
        Work work = processGroup.send(toTensorVector(tensor), dstRank, tag);
        work._wait();
    }

    public Tensor recv(Tensor tensor, int srcRank) {
        return recv(tensor, srcRank, 0);
    }

    public Tensor recv(Tensor tensor, int srcRank, int tag) {
        Work work = processGroup.recv(toTensorVector(tensor), srcRank, tag);
        work._wait();
        return tensor;
    }

    public Tensor recvAnysource(Tensor tensor) {
        return recvAnysource(tensor, 0);
    }

    public Tensor recvAnysource(Tensor tensor, int tag) {
        Work work = processGroup.recvAnysource(toTensorVector(tensor), tag);
        work._wait();
        return tensor;
    }

    public Work barrier() {
        return processGroup.barrier(new BarrierOptions());
    }

    /** Allreduce(SUM) then divide by worldSize in-place. */
    public void averageGradients(List<Tensor> gradients) {
        if (gradients == null || gradients.isEmpty()) {
            return;
        }
        allreduce(gradients, ReduceOp.RedOpType.SUM);
        Scalar denom = new Scalar(worldSize);
        for (Tensor g : gradients) {
            g.div_(denom);
        }
    }

    public void syncParameters(List<Tensor> parameters, int rootRank) {
        broadcast(parameters, rootRank);
    }

    public Backend getNativeGroup() { return processGroup; }
    public int getRank() { return rank; }
    public int getWorldSize() { return worldSize; }
    public BackendType getBackend() { return backendType; }
    public String getBackendName() { return backendName; }
    public Device getDevice() { return device; }
    public boolean isMainProcess() { return rank == 0; }

    @Override
    public void close() {
        if ("nccl".equalsIgnoreCase(backendName)) {
            processGroup.waitForPendingWorks();
        }
        processGroup.shutdown();
        INSTANCES.remove(this);
    }

    @Override
    public String toString() {
        return "ProcessGroupWrapper{backend=" + backendName
                + ", rank=" + rank + ", worldSize=" + worldSize + ", device=" + device + '}';
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

        public Options backend(BackendType b) { this.backendType = b; return this; }
        public Options timeout(int ms) { this.timeoutMs = ms; return this; }
        public Options masterAddr(String addr) { this.masterAddr = addr; return this; }
        public Options masterPort(int port) { this.masterPort = port; return this; }

        public BackendType getBackend() { return backendType; }
        public int getTimeoutMs() { return timeoutMs; }
        public String getMasterAddr() { return masterAddr; }
        public int getMasterPort() { return masterPort; }
    }
}
