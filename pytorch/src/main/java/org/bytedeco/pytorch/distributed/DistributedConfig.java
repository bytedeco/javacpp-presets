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
import org.bytedeco.pytorch.Device;

import static org.bytedeco.pytorch.global.torch.DeviceType;
import static org.bytedeco.pytorch.global.torch.cuda_is_available;

/**
 * Rank / world-size / device / backend snapshot for a distributed job.
 *
 * <pre>{@code
 * DistributedConfig cfg = DistributedConfig.builder()
 *     .rank(0).worldSize(4).backend("NCCL").build();
 * }</pre>
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DistributedConfig {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final int rank;
    private final int worldSize;
    private final Device device;
    private final String backend;

    public DistributedConfig(int rank, int worldSize, Device device, String backend) {
        this.rank = rank;
        this.worldSize = worldSize;
        this.device = device;
        this.backend = backend;
    }

    public int rank() { return rank; }
    public int worldSize() { return worldSize; }
    public Device device() { return device; }
    public String backend() { return backend; }
    public boolean isMainProcess() { return rank == 0; }
    public boolean isDistributed() { return worldSize > 1; }

    @Override
    public String toString() {
        return "DistributedConfig{rank=" + rank + ", worldSize=" + worldSize
                + ", device=" + device + ", backend=" + backend + '}';
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private int rank = 0;
        private int worldSize = 1;
        private String backend = "GLOO";

        public Builder rank(int r) { this.rank = r; return this; }
        public Builder worldSize(int ws) { this.worldSize = ws; return this; }
        public Builder backend(String b) { this.backend = b; return this; }

        public DistributedConfig build() {
            String effective = "NCCL".equalsIgnoreCase(backend) && cuda_is_available()
                    ? "NCCL" : "GLOO";
            Device dev = "NCCL".equals(effective) && cuda_is_available()
                    ? new Device(DeviceType.CUDA, (byte) rank)
                    : new Device(DeviceType.CPU, (byte) 0);
            return new DistributedConfig(rank, worldSize, dev, effective);
        }
    }
}
