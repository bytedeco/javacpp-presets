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
package org.bytedeco.pytorch.llm.ktransformers.inference;

import org.bytedeco.pytorch.llm.ktransformers.config.KtDevicePlacement;
import org.bytedeco.pytorch.llm.ktransformers.config.KtInferenceConfig;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Layer-wise / expert-wise multi-GPU device map (config-driven).
 *
 * <p>Does not launch NCCL; documents placement for host meshes and optional
 * future {@code module.to(device)} hooks. When {@code multiGpu=false}, all layers
 * map to {@code cuda:0} / rank 0.
 */
public final class MultiGpuCoordinator {

    private final int worldSize;
    private final int tensorParallel;
    private final Map<Integer, String> layerToDevice;
    private final Map<Integer, String> expertToDevice;
    private final boolean enabled;

    public MultiGpuCoordinator(KtInferenceConfig inference, KtDevicePlacement placement,
                               int numLayers, int numExperts) {
        Objects.requireNonNull(inference, "inference");
        this.enabled = inference.multiGpu();
        this.tensorParallel = Math.max(1, inference.tensorParallel());
        this.worldSize = this.enabled ? Math.max(1, this.tensorParallel) : 1;

        Map<Integer, String> layers = new LinkedHashMap<>();
        Map<Integer, String> experts = new LinkedHashMap<>();
        if (placement != null && placement.layerDeviceMap() != null && !placement.layerDeviceMap().isEmpty()) {
            layers.putAll(placement.layerDeviceMap());
        } else {
            for (int i = 0; i < numLayers; i++) {
                layers.put(i, deviceString(enabled ? (i % worldSize) : 0));
            }
        }
        if (placement != null && placement.expertDeviceMap() != null && !placement.expertDeviceMap().isEmpty()) {
            experts.putAll(placement.expertDeviceMap());
        } else {
            for (int e = 0; e < numExperts; e++) {
                experts.put(e, deviceString(enabled ? (e % worldSize) : 0));
            }
        }
        this.layerToDevice = Collections.unmodifiableMap(layers);
        this.expertToDevice = Collections.unmodifiableMap(experts);
    }

    private static String deviceString(int rank) {
        return rank <= 0 ? "cuda:0" : ("cuda:" + rank);
    }

    public static MultiGpuCoordinator single() {
        return new MultiGpuCoordinator(
                KtInferenceConfig.defaults(),
                KtDevicePlacement.defaults(),
                1, 1);
    }

    public boolean enabled() { return enabled; }
    public int worldSize() { return worldSize; }
    public int tensorParallel() { return tensorParallel; }
    public Map<Integer, String> layerToDevice() { return layerToDevice; }
    public Map<Integer, String> expertToDevice() { return expertToDevice; }

    public String deviceForLayer(int layer) {
        return layerToDevice.getOrDefault(layer, "cuda:0");
    }

    public String deviceForExpert(int expert) {
        return expertToDevice.getOrDefault(expert, "cuda:0");
    }

    /** Parse trailing rank from {@code cuda:N} / {@code cpu}; default 0. */
    public static int rankOf(String device) {
        if (device == null || device.isEmpty() || device.startsWith("cpu")) return 0;
        int colon = device.lastIndexOf(':');
        if (colon < 0) return 0;
        try {
            return Integer.parseInt(device.substring(colon + 1).trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    @Override
    public String toString() {
        return "MultiGpuCoordinator{enabled=" + enabled + ", worldSize=" + worldSize
                + ", tp=" + tensorParallel + ", layers=" + layerToDevice.size()
                + ", experts=" + expertToDevice.size() + "}";
    }
}
