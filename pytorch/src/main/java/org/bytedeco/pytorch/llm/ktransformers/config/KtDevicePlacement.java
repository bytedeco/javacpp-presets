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
package org.bytedeco.pytorch.llm.ktransformers.config;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Device placement policy for layers and MoE experts (GPU / CPU / Disk tiers).
 *
 * <p>Maps are logical ids → device strings understood by {@code new Device(spec)}
 * (e.g. {@code "cuda:0"}, {@code "cpu"}). Empty maps mean "engine default".
 */
public final class KtDevicePlacement {

    public enum DefaultCompute {
        AUTO,
        CUDA,
        CPU,
        MPS
    }

    private final DefaultCompute defaultCompute;
    private final Map<Integer, String> layerDeviceMap;
    private final Map<Integer, String> expertDeviceMap;
    private final boolean pinMemory;
    private final boolean offloadFrozenToCpu;
    private final double gpuMemFraction;
    private final long cpuExpertBytesBudget;
    private final long diskBytesBudget;

    private KtDevicePlacement(Builder b) {
        this.defaultCompute = Objects.requireNonNull(b.defaultCompute, "defaultCompute");
        this.layerDeviceMap = Collections.unmodifiableMap(new LinkedHashMap<>(b.layerDeviceMap));
        this.expertDeviceMap = Collections.unmodifiableMap(new LinkedHashMap<>(b.expertDeviceMap));
        this.pinMemory = b.pinMemory;
        this.offloadFrozenToCpu = b.offloadFrozenToCpu;
        if (b.gpuMemFraction <= 0.0 || b.gpuMemFraction > 1.0) {
            throw new IllegalArgumentException("gpuMemFraction must be in (0, 1]");
        }
        this.gpuMemFraction = b.gpuMemFraction;
        this.cpuExpertBytesBudget = b.cpuExpertBytesBudget;
        this.diskBytesBudget = b.diskBytesBudget;
    }

    public DefaultCompute defaultCompute() { return defaultCompute; }
    public Map<Integer, String> layerDeviceMap() { return layerDeviceMap; }
    public Map<Integer, String> expertDeviceMap() { return expertDeviceMap; }
    public boolean pinMemory() { return pinMemory; }
    public boolean offloadFrozenToCpu() { return offloadFrozenToCpu; }
    public double gpuMemFraction() { return gpuMemFraction; }
    public long cpuExpertBytesBudget() { return cpuExpertBytesBudget; }
    public long diskBytesBudget() { return diskBytesBudget; }

    public String deviceForLayer(int layerIdx, String fallback) {
        return layerDeviceMap.getOrDefault(layerIdx, fallback);
    }

    public String deviceForExpert(int expertId, String fallback) {
        return expertDeviceMap.getOrDefault(expertId, fallback);
    }

    public static Builder builder() { return new Builder(); }

    public static KtDevicePlacement defaults() {
        return builder().build();
    }

    public static final class Builder {
        private DefaultCompute defaultCompute = DefaultCompute.AUTO;
        private final Map<Integer, String> layerDeviceMap = new LinkedHashMap<>();
        private final Map<Integer, String> expertDeviceMap = new LinkedHashMap<>();
        private boolean pinMemory = true;
        private boolean offloadFrozenToCpu = true;
        private double gpuMemFraction = 0.90;
        private long cpuExpertBytesBudget = 64L * 1024 * 1024 * 1024; // 64 GiB logical
        private long diskBytesBudget = 512L * 1024 * 1024 * 1024;

        public Builder defaultCompute(DefaultCompute v) { this.defaultCompute = v; return this; }
        public Builder layerDevice(int layer, String device) {
            this.layerDeviceMap.put(layer, device);
            return this;
        }
        public Builder expertDevice(int expertId, String device) {
            this.expertDeviceMap.put(expertId, device);
            return this;
        }
        public Builder layerDeviceMap(Map<Integer, String> m) {
            this.layerDeviceMap.clear();
            if (m != null) this.layerDeviceMap.putAll(m);
            return this;
        }
        public Builder expertDeviceMap(Map<Integer, String> m) {
            this.expertDeviceMap.clear();
            if (m != null) this.expertDeviceMap.putAll(m);
            return this;
        }
        public Builder pinMemory(boolean v) { this.pinMemory = v; return this; }
        public Builder offloadFrozenToCpu(boolean v) { this.offloadFrozenToCpu = v; return this; }
        public Builder gpuMemFraction(double v) { this.gpuMemFraction = v; return this; }
        public Builder cpuExpertBytesBudget(long v) { this.cpuExpertBytesBudget = v; return this; }
        public Builder diskBytesBudget(long v) { this.diskBytesBudget = v; return this; }

        public KtDevicePlacement build() { return new KtDevicePlacement(this); }
    }
}
