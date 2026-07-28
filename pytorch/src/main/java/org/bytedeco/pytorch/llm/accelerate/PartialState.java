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
import org.bytedeco.pytorch.distributed.DistributedStore;
import org.bytedeco.pytorch.distributed.ProcessGroupWrapper;
import org.bytedeco.pytorch.distributed.StoreType;
import org.bytedeco.pytorch.llm.accelerate.utils.MultiProcessLauncher;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.DeviceType;
import static org.bytedeco.pytorch.global.torch.cuda_is_available;
import static org.bytedeco.pytorch.global.torch.hasMPS;

/**
 * Lightweight distributed process state (HF {@code accelerate.PartialState}).
 *
 * <p>Reads {@link MultiProcessLauncher} env vars when present; otherwise single-process.
 * Optionally owns a {@link ProcessGroupWrapper} for collectives.
 *
 * <pre>{@code
 * try (PartialState state = PartialState.fromEnv()) {
 *     state.print("hello from rank " + state.processIndex());
 *     state.waitForEveryone();
 * }
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class PartialState implements AutoCloseable {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final int processIndex;
    private final int numProcesses;
    private final int localProcessIndex;
    private final Device device;
    private final ProcessGroupWrapper processGroup;
    private final DistributedStore store;
    private final boolean ownsGroup;
    private final String mixedPrecision;

    public PartialState(int processIndex, int numProcesses, Device device,
                        ProcessGroupWrapper processGroup, DistributedStore store,
                        boolean ownsGroup, String mixedPrecision) {
        this.processIndex = processIndex;
        this.numProcesses = Math.max(1, numProcesses);
        this.localProcessIndex = processIndex;
        this.device = device == null ? autoDevice(false) : device;
        this.processGroup = processGroup;
        this.store = store;
        this.ownsGroup = ownsGroup;
        this.mixedPrecision = mixedPrecision == null ? "no" : mixedPrecision;
    }

    public static PartialState single() {
        return new PartialState(0, 1, autoDevice(false), null, null, false, "no");
    }

    /** Build from RANK/WORLD_SIZE env without opening a process group. */
    public static PartialState fromEnv() {
        int rank = MultiProcessLauncher.envRank();
        int world = MultiProcessLauncher.envWorldSize();
        Device dev = autoDevice(false);
        if (world > 1 && cuda_is_available()) {
            try {
                dev = new Device(DeviceType.CUDA, (byte) MultiProcessLauncher.envLocalRank());
            } catch (Throwable ignored) {
                dev = autoDevice(false);
            }
        }
        return new PartialState(rank, world, dev, null, null, false, "no");
    }

    /**
     * Open FileStore + Gloo/NCCL process group from launcher env.
     * Caller must close the returned state.
     */
    public static PartialState initProcessGroupFromEnv() {
        int rank = MultiProcessLauncher.envRank();
        int world = MultiProcessLauncher.envWorldSize();
        if (world <= 1) {
            return single();
        }
        DistributedStore.Options sopts = new DistributedStore.Options()
                .type(StoreType.FILE)
                .masterAddr(MultiProcessLauncher.envMasterAddr())
                .masterPort(MultiProcessLauncher.envMasterPort())
                .numWorkers(world);
        DistributedStore store = DistributedStore.create(sopts, rank, world);
        ProcessGroupWrapper.Options popts = new ProcessGroupWrapper.Options()
                .masterAddr(MultiProcessLauncher.envMasterAddr())
                .masterPort(MultiProcessLauncher.envMasterPort());
        ProcessGroupWrapper pg = ProcessGroupWrapper.create(popts, rank, world, store);
        Device dev = pg.getDevice();
        return new PartialState(rank, world, dev, pg, store, true, "no");
    }

    public static PartialState of(ProcessGroupWrapper pg) {
        Objects.requireNonNull(pg, "processGroup");
        return new PartialState(pg.getRank(), pg.getWorldSize(), pg.getDevice(),
                pg, null, false, "no");
    }

    public static Builder builder() {
        return new Builder();
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

    public int processIndex() { return processIndex; }
    public int numProcesses() { return numProcesses; }
    public int localProcessIndex() { return localProcessIndex; }
    public Device device() { return device; }
    public ProcessGroupWrapper processGroup() { return processGroup; }
    public DistributedStore store() { return store; }
    public String mixedPrecision() { return mixedPrecision; }
    public boolean isMainProcess() { return processIndex == 0; }
    public boolean isLocalMainProcess() { return localProcessIndex == 0; }
    public boolean isLastProcess() { return processIndex == numProcesses - 1; }
    public boolean distributedType() { return numProcesses > 1; }
    public boolean useDistributed() { return numProcesses > 1 && processGroup != null; }

    public void waitForEveryone() {
        if (processGroup != null && numProcesses > 1) {
            try {
                processGroup.barrier();
            } catch (Exception ignored) {
            }
        }
    }

    public void print(String msg) {
        if (isMainProcess()) {
            System.out.println(msg);
        }
    }

    public void print(String fmt, Object... args) {
        if (isMainProcess()) {
            System.out.printf(fmt, args);
            if (fmt == null || !fmt.endsWith("\n")) System.out.println();
        }
    }

    @Override
    public void close() {
        if (ownsGroup) {
            try {
                if (processGroup != null) processGroup.close();
            } catch (Exception ignored) {}
            try {
                if (store != null) store.close();
            } catch (Exception ignored) {}
        }
    }

    @Override
    public String toString() {
        return "PartialState{rank=" + processIndex + "/" + numProcesses
                + ", device=" + device + ", mixedPrecision=" + mixedPrecision + '}';
    }

    public static final class Builder {
        private int processIndex = 0;
        private int numProcesses = 1;
        private Device device;
        private ProcessGroupWrapper processGroup;
        private DistributedStore store;
        private boolean ownsGroup;
        private String mixedPrecision = "no";
        private boolean cpu;

        public Builder processIndex(int v) { this.processIndex = v; return this; }
        public Builder numProcesses(int v) { this.numProcesses = v; return this; }
        public Builder device(Device d) { this.device = d; return this; }
        public Builder processGroup(ProcessGroupWrapper pg) { this.processGroup = pg; return this; }
        public Builder store(DistributedStore s) { this.store = s; return this; }
        public Builder ownsGroup(boolean v) { this.ownsGroup = v; return this; }
        public Builder mixedPrecision(String v) { this.mixedPrecision = v; return this; }
        public Builder cpu(boolean v) { this.cpu = v; return this; }

        public PartialState build() {
            Device d = device != null ? device : autoDevice(cpu);
            if (processGroup != null) {
                processIndex = processGroup.getRank();
                numProcesses = processGroup.getWorldSize();
                d = processGroup.getDevice();
            }
            return new PartialState(processIndex, numProcesses, d, processGroup, store,
                    ownsGroup, mixedPrecision);
        }
    }
}
