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

package org.bytedeco.pytorch.llm.unsloth.studio.model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public final class HardwareProfile {
    private final String osName;
    private final String arch;
    private final int cpuCores;
    private final long systemMemoryMb;
    private final boolean cudaAvailable;
    private final boolean mpsAvailable;
    private final boolean rocmAvailable;
    private final List<GpuDevice> gpus;
    private final String recommendedDevice;

    public HardwareProfile(String osName, String arch, int cpuCores, long systemMemoryMb,
                           boolean cudaAvailable, boolean mpsAvailable, boolean rocmAvailable,
                           List<GpuDevice> gpus, String recommendedDevice) {
        this.osName = osName;
        this.arch = arch;
        this.cpuCores = cpuCores;
        this.systemMemoryMb = systemMemoryMb;
        this.cudaAvailable = cudaAvailable;
        this.mpsAvailable = mpsAvailable;
        this.rocmAvailable = rocmAvailable;
        this.gpus = List.copyOf(gpus);
        this.recommendedDevice = recommendedDevice;
    }

    public String osName() { return osName; }
    public String arch() { return arch; }
    public int cpuCores() { return cpuCores; }
    public long systemMemoryMb() { return systemMemoryMb; }
    public boolean cudaAvailable() { return cudaAvailable; }
    public boolean mpsAvailable() { return mpsAvailable; }
    public boolean rocmAvailable() { return rocmAvailable; }
    public List<GpuDevice> gpus() { return gpus; }
    public String recommendedDevice() { return recommendedDevice; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("os", osName);
        m.put("arch", arch);
        m.put("cpu_cores", cpuCores);
        m.put("system_memory_mb", systemMemoryMb);
        m.put("cuda_available", cudaAvailable);
        m.put("mps_available", mpsAvailable);
        m.put("rocm_available", rocmAvailable);
        m.put("recommended_device", recommendedDevice);
        List<Map<String, Object>> gs = new ArrayList<>();
        for (GpuDevice g : gpus) gs.add(g.toMap());
        m.put("gpus", gs);
        return m;
    }

    public static final class GpuDevice {
        private final int index;
        private final String name;
        private final long totalMemoryMb;
        private final String backend;

        public GpuDevice(int index, String name, long totalMemoryMb, String backend) {
            this.index = index;
            this.name = name;
            this.totalMemoryMb = totalMemoryMb;
            this.backend = backend;
        }

        public int index() { return index; }
        public String name() { return name; }
        public long totalMemoryMb() { return totalMemoryMb; }
        public String backend() { return backend; }

        public Map<String, Object> toMap() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("index", index);
            m.put("name", name);
            m.put("total_memory_mb", totalMemoryMb);
            m.put("backend", backend);
            return m;
        }
    }
}
