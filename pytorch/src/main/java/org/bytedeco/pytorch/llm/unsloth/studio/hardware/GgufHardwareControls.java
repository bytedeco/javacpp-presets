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

package org.bytedeco.pytorch.llm.unsloth.studio.hardware;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * GGUF runtime hardware knobs: GPU layers, MoE expert offload, multi-GPU, tensor parallel.
 */
public final class GgufHardwareControls {
    private final int nGpuLayers;
    private final boolean offloadMoeExperts;
    private final List<Integer> gpuIds;
    private final boolean tensorParallel;
    private final int tensorParallelSize;
    private final String cacheTypeK;
    private final String cacheTypeV;
    private final int mainGpu;
    private final boolean flashAttn;
    private final int threads;

    private GgufHardwareControls(Builder b) {
        this.nGpuLayers = b.nGpuLayers;
        this.offloadMoeExperts = b.offloadMoeExperts;
        this.gpuIds = List.copyOf(b.gpuIds);
        this.tensorParallel = b.tensorParallel;
        this.tensorParallelSize = b.tensorParallelSize > 0 ? b.tensorParallelSize
                : Math.max(1, b.gpuIds.size());
        this.cacheTypeK = b.cacheTypeK;
        this.cacheTypeV = b.cacheTypeV;
        this.mainGpu = b.mainGpu;
        this.flashAttn = b.flashAttn;
        this.threads = b.threads;
    }

    public static Builder builder() { return new Builder(); }
    public static GgufHardwareControls defaults() { return builder().build(); }

    public int nGpuLayers() { return nGpuLayers; }
    public boolean offloadMoeExperts() { return offloadMoeExperts; }
    public List<Integer> gpuIds() { return gpuIds; }
    public boolean tensorParallel() { return tensorParallel; }
    public int tensorParallelSize() { return tensorParallelSize; }
    public Optional<String> cacheTypeK() { return Optional.ofNullable(cacheTypeK); }
    public Optional<String> cacheTypeV() { return Optional.ofNullable(cacheTypeV); }
    public int mainGpu() { return mainGpu; }
    public boolean flashAttn() { return flashAttn; }
    public int threads() { return threads; }

    /** CLI-style argument map for an external GGUF runner (llama.cpp compatible names). */
    public Map<String, String> toRunnerArgs() {
        Map<String, String> m = new LinkedHashMap<>();
        m.put("n-gpu-layers", String.valueOf(nGpuLayers));
        if (offloadMoeExperts) m.put("offload-moe", "1");
        if (tensorParallel) m.put("tensor-split", String.valueOf(tensorParallelSize));
        if (cacheTypeK != null) m.put("cache-type-k", cacheTypeK);
        if (cacheTypeV != null) m.put("cache-type-v", cacheTypeV);
        m.put("main-gpu", String.valueOf(mainGpu));
        if (flashAttn) m.put("flash-attn", "1");
        if (threads > 0) m.put("threads", String.valueOf(threads));
        if (!gpuIds.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < gpuIds.size(); i++) {
                if (i > 0) sb.append(',');
                sb.append(gpuIds.get(i));
            }
            m.put("gpu-ids", sb.toString());
        }
        return m;
    }

    public static final class Builder {
        private int nGpuLayers = -1; // -1 = all
        private boolean offloadMoeExperts = false;
        private List<Integer> gpuIds = List.of();
        private boolean tensorParallel = false;
        private int tensorParallelSize = 0;
        private String cacheTypeK;
        private String cacheTypeV;
        private int mainGpu = 0;
        private boolean flashAttn = true;
        private int threads = 0;

        public Builder nGpuLayers(int v) { this.nGpuLayers = v; return this; }
        public Builder offloadMoeExperts(boolean v) { this.offloadMoeExperts = v; return this; }
        public Builder gpuIds(List<Integer> v) { this.gpuIds = v != null ? new ArrayList<>(v) : List.of(); return this; }
        public Builder tensorParallel(boolean v) { this.tensorParallel = v; return this; }
        public Builder tensorParallelSize(int v) { this.tensorParallelSize = v; return this; }
        public Builder cacheTypeK(String v) { this.cacheTypeK = v; return this; }
        public Builder cacheTypeV(String v) { this.cacheTypeV = v; return this; }
        public Builder cacheTypeKv(String v) { this.cacheTypeK = v; this.cacheTypeV = v; return this; }
        public Builder mainGpu(int v) { this.mainGpu = v; return this; }
        public Builder flashAttn(boolean v) { this.flashAttn = v; return this; }
        public Builder threads(int v) { this.threads = v; return this; }
        public GgufHardwareControls build() { return new GgufHardwareControls(this); }
    }
}
