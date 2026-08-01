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

import org.bytedeco.pytorch.llm.unsloth.studio.util.Validate;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Request to load a model for inference (upstream LoadRequest aligned). */
public final class LoadRequest {
    private final String modelPath;
    private final String hfToken;
    private final int maxSeqLength;
    private final boolean loadIn4bit;
    private final boolean loadIn8bit;
    private final boolean isLora;
    private final String loraPath;
    private final String ggufVariant;
    private final boolean trustRemoteCode;
    private final String chatTemplateOverride;
    private final String cacheTypeKv;
    private final List<Integer> gpuIds;
    private final String speculativeType;
    private final Integer nParallel;
    private final Integer nGpuLayers;
    private final boolean tensorParallel;
    private final boolean offloadMoeExperts;

    private LoadRequest(Builder b) {
        this.modelPath = Objects.requireNonNull(b.modelPath, "model_path");
        Validate.requireNonBlank("model_path", b.modelPath);
        this.hfToken = b.hfToken;
        this.maxSeqLength = b.maxSeqLength; // 0 = model default
        if (b.maxSeqLength < 0 || b.maxSeqLength > 1_048_576) {
            throw new org.bytedeco.pytorch.llm.unsloth.studio.util.StudioValidationException(
                    "max_seq_length must be in [0, 1048576], got " + b.maxSeqLength);
        }
        this.loadIn4bit = b.loadIn4bit;
        this.loadIn8bit = b.loadIn8bit;
        this.isLora = b.isLora;
        this.loraPath = b.loraPath;
        this.ggufVariant = b.ggufVariant;
        this.trustRemoteCode = b.trustRemoteCode;
        this.chatTemplateOverride = Validate.chatTemplateOverride(b.chatTemplateOverride);
        this.cacheTypeKv = b.cacheTypeKv;
        this.gpuIds = List.copyOf(b.gpuIds);
        this.speculativeType = b.speculativeType;
        this.nParallel = b.nParallel;
        this.nGpuLayers = b.nGpuLayers;
        this.tensorParallel = b.tensorParallel;
        this.offloadMoeExperts = b.offloadMoeExperts;
    }

    public static Builder builder() { return new Builder(); }

    public String modelPath() { return modelPath; }
    public Optional<String> hfToken() { return Optional.ofNullable(hfToken); }
    public int maxSeqLength() { return maxSeqLength; }
    public boolean loadIn4bit() { return loadIn4bit; }
    public boolean loadIn8bit() { return loadIn8bit; }
    public boolean isLora() { return isLora; }
    public Optional<String> loraPath() { return Optional.ofNullable(loraPath); }
    public Optional<String> ggufVariant() { return Optional.ofNullable(ggufVariant); }
    public boolean trustRemoteCode() { return trustRemoteCode; }
    public Optional<String> chatTemplateOverride() { return Optional.ofNullable(chatTemplateOverride); }
    public Optional<String> cacheTypeKv() { return Optional.ofNullable(cacheTypeKv); }
    public List<Integer> gpuIds() { return gpuIds; }
    public Optional<String> speculativeType() { return Optional.ofNullable(speculativeType); }
    public Optional<Integer> nParallel() { return Optional.ofNullable(nParallel); }
    public Optional<Integer> nGpuLayers() { return Optional.ofNullable(nGpuLayers); }
    public boolean tensorParallel() { return tensorParallel; }
    public boolean offloadMoeExperts() { return offloadMoeExperts; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("model_path", modelPath);
        m.put("max_seq_length", maxSeqLength);
        m.put("load_in_4bit", loadIn4bit);
        m.put("load_in_8bit", loadIn8bit);
        m.put("is_lora", isLora);
        if (loraPath != null) m.put("lora_path", loraPath);
        if (ggufVariant != null) m.put("gguf_variant", ggufVariant);
        m.put("trust_remote_code", trustRemoteCode);
        if (chatTemplateOverride != null) m.put("chat_template_override", chatTemplateOverride);
        if (cacheTypeKv != null) m.put("cache_type_kv", cacheTypeKv);
        if (!gpuIds.isEmpty()) m.put("gpu_ids", gpuIds);
        if (speculativeType != null) m.put("speculative_type", speculativeType);
        if (nParallel != null) m.put("n_parallel", nParallel);
        if (nGpuLayers != null) m.put("n_gpu_layers", nGpuLayers);
        m.put("tensor_parallel", tensorParallel);
        m.put("offload_moe_experts", offloadMoeExperts);
        return m;
    }

    public static final class Builder {
        private String modelPath;
        private String hfToken;
        private int maxSeqLength = 0;
        private boolean loadIn4bit = true;
        private boolean loadIn8bit = false;
        private boolean isLora = false;
        private String loraPath;
        private String ggufVariant;
        private boolean trustRemoteCode = false;
        private String chatTemplateOverride;
        private String cacheTypeKv;
        private List<Integer> gpuIds = List.of();
        private String speculativeType;
        private Integer nParallel;
        private Integer nGpuLayers;
        private boolean tensorParallel = false;
        private boolean offloadMoeExperts = false;

        public Builder modelPath(String v) { this.modelPath = v; return this; }
        public Builder hfToken(String v) { this.hfToken = v; return this; }
        public Builder maxSeqLength(int v) { this.maxSeqLength = v; return this; }
        public Builder loadIn4bit(boolean v) { this.loadIn4bit = v; return this; }
        public Builder loadIn8bit(boolean v) { this.loadIn8bit = v; return this; }
        public Builder isLora(boolean v) { this.isLora = v; return this; }
        public Builder loraPath(String v) { this.loraPath = v; return this; }
        public Builder ggufVariant(String v) { this.ggufVariant = v; return this; }
        public Builder trustRemoteCode(boolean v) { this.trustRemoteCode = v; return this; }
        public Builder chatTemplateOverride(String v) { this.chatTemplateOverride = v; return this; }
        public Builder cacheTypeKv(String v) { this.cacheTypeKv = v; return this; }
        public Builder gpuIds(List<Integer> v) { this.gpuIds = v != null ? new ArrayList<>(v) : List.of(); return this; }
        public Builder speculativeType(String v) { this.speculativeType = v; return this; }
        public Builder nParallel(Integer v) { this.nParallel = v; return this; }
        public Builder nGpuLayers(Integer v) { this.nGpuLayers = v; return this; }
        public Builder tensorParallel(boolean v) { this.tensorParallel = v; return this; }
        public Builder offloadMoeExperts(boolean v) { this.offloadMoeExperts = v; return this; }
        public LoadRequest build() { return new LoadRequest(this); }
    }
}
