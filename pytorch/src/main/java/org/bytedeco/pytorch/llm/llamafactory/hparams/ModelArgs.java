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
package org.bytedeco.pytorch.llm.llamafactory.hparams;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Model / tokenizer / quant / rope / backend args (LLaMA-Factory model section). */
public final class ModelArgs {
    private final String modelNameOrPath;
    private final String adapterNameOrPath;
    private final List<String> adapterToMerge;
    private final boolean useFastTokenizer;
    private final boolean resizeVocab;
    private final RopeScalingType ropeScaling;
    private final double ropeScalingFactor;
    private final boolean flashAttn;
    private final boolean shiftAttn;
    private final String mixedPrecision;
    private final QuantizationMethod quantizationMethod;
    private final int quantizationBit;
    private final boolean doubleQuantization;
    private final String quantType;
    private final boolean lowCpuMemUsage;
    private final boolean useUnsloth;
    private final boolean useLigerKernel;
    private final boolean useKtransformers;
    private final String modelRevision;
    private final boolean trustRemoteCode;
    private final double neftuneAlpha;
    private final String inferBackend;
    private final double vllmGpuMemoryUtilization;
    private final int vllmMaxModelLen;

    private ModelArgs(Builder b) {
        this.modelNameOrPath = Objects.requireNonNull(b.modelNameOrPath, "modelNameOrPath");
        this.adapterNameOrPath = b.adapterNameOrPath;
        this.adapterToMerge = Collections.unmodifiableList(new ArrayList<>(b.adapterToMerge));
        this.useFastTokenizer = b.useFastTokenizer;
        this.resizeVocab = b.resizeVocab;
        this.ropeScaling = b.ropeScaling == null ? RopeScalingType.NONE : b.ropeScaling;
        this.ropeScalingFactor = b.ropeScalingFactor;
        this.flashAttn = b.flashAttn;
        this.shiftAttn = b.shiftAttn;
        this.mixedPrecision = b.mixedPrecision == null ? "bf16" : b.mixedPrecision;
        this.quantizationMethod = b.quantizationMethod == null ? QuantizationMethod.NONE : b.quantizationMethod;
        this.quantizationBit = b.quantizationBit;
        this.doubleQuantization = b.doubleQuantization;
        this.quantType = b.quantType == null ? "nf4" : b.quantType;
        this.lowCpuMemUsage = b.lowCpuMemUsage;
        this.useUnsloth = b.useUnsloth;
        this.useLigerKernel = b.useLigerKernel;
        this.useKtransformers = b.useKtransformers;
        this.modelRevision = b.modelRevision == null ? "main" : b.modelRevision;
        this.trustRemoteCode = b.trustRemoteCode;
        this.neftuneAlpha = b.neftuneAlpha;
        this.inferBackend = b.inferBackend == null ? "huggingface" : b.inferBackend;
        this.vllmGpuMemoryUtilization = b.vllmGpuMemoryUtilization;
        this.vllmMaxModelLen = b.vllmMaxModelLen;
    }

    public String modelNameOrPath() { return modelNameOrPath; }
    public String adapterNameOrPath() { return adapterNameOrPath; }
    public List<String> adapterToMerge() { return adapterToMerge; }
    public boolean useFastTokenizer() { return useFastTokenizer; }
    public boolean resizeVocab() { return resizeVocab; }
    public RopeScalingType ropeScaling() { return ropeScaling; }
    public double ropeScalingFactor() { return ropeScalingFactor; }
    public boolean flashAttn() { return flashAttn; }
    public boolean shiftAttn() { return shiftAttn; }
    public String mixedPrecision() { return mixedPrecision; }
    public QuantizationMethod quantizationMethod() { return quantizationMethod; }
    public int quantizationBit() { return quantizationBit; }
    public boolean doubleQuantization() { return doubleQuantization; }
    public String quantType() { return quantType; }
    public boolean lowCpuMemUsage() { return lowCpuMemUsage; }
    public boolean useUnsloth() { return useUnsloth; }
    public boolean useLigerKernel() { return useLigerKernel; }
    public boolean useKtransformers() { return useKtransformers; }
    public String modelRevision() { return modelRevision; }
    public boolean trustRemoteCode() { return trustRemoteCode; }
    public double neftuneAlpha() { return neftuneAlpha; }
    public boolean neftuneEnabled() { return neftuneAlpha > 0.0; }
    public String inferBackend() { return inferBackend; }
    public double vllmGpuMemoryUtilization() { return vllmGpuMemoryUtilization; }
    public int vllmMaxModelLen() { return vllmMaxModelLen; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        HparamsMaps.put(m, "model_name_or_path", modelNameOrPath);
        HparamsMaps.put(m, "adapter_name_or_path", adapterNameOrPath);
        HparamsMaps.put(m, "adapter_to_merge", adapterToMerge);
        HparamsMaps.put(m, "use_fast_tokenizer", useFastTokenizer);
        HparamsMaps.put(m, "resize_vocab", resizeVocab);
        HparamsMaps.put(m, "rope_scaling", ropeScaling.wireName());
        HparamsMaps.put(m, "rope_scaling_factor", ropeScalingFactor);
        HparamsMaps.put(m, "flash_attn", flashAttn);
        HparamsMaps.put(m, "shift_attn", shiftAttn);
        HparamsMaps.put(m, "mixed_precision", mixedPrecision);
        HparamsMaps.put(m, "quantization_method", quantizationMethod.wireName());
        HparamsMaps.put(m, "quantization_bit", quantizationBit);
        HparamsMaps.put(m, "double_quantization", doubleQuantization);
        HparamsMaps.put(m, "quant_type", quantType);
        HparamsMaps.put(m, "low_cpu_mem_usage", lowCpuMemUsage);
        HparamsMaps.put(m, "use_unsloth", useUnsloth);
        HparamsMaps.put(m, "use_liger_kernel", useLigerKernel);
        HparamsMaps.put(m, "use_ktransformers", useKtransformers);
        HparamsMaps.put(m, "model_revision", modelRevision);
        HparamsMaps.put(m, "trust_remote_code", trustRemoteCode);
        HparamsMaps.put(m, "neftune_alpha", neftuneAlpha);
        HparamsMaps.put(m, "infer_backend", inferBackend);
        HparamsMaps.put(m, "vllm_gpu_memory_utilization", vllmGpuMemoryUtilization);
        HparamsMaps.put(m, "vllm_maxlen", vllmMaxModelLen);
        return m;
    }

    public static ModelArgs defaults() { return builder().build(); }

    public static ModelArgs fromMap(Map<String, ?> m) {
        if (m == null || m.isEmpty()) return defaults();
        Builder b = builder();
        b.modelNameOrPath(HparamsMaps.str(m, b.modelNameOrPath, "model_name_or_path", "model"));
        b.adapterNameOrPath(HparamsMaps.strOrNull(m, "adapter_name_or_path", "adapter_path"));
        List<String> merge = HparamsMaps.stringList(m, "adapter_to_merge", "adapters");
        if (!merge.isEmpty()) b.adapterToMerge(merge);
        b.useFastTokenizer(HparamsMaps.bool(m, b.useFastTokenizer, "use_fast_tokenizer"));
        b.resizeVocab(HparamsMaps.bool(m, b.resizeVocab, "resize_vocab"));
        String rs = HparamsMaps.strOrNull(m, "rope_scaling");
        if (rs != null) b.ropeScaling(RopeScalingType.parse(rs));
        b.ropeScalingFactor(HparamsMaps.dbl(m, b.ropeScalingFactor, "rope_scaling_factor", "rope_factor"));
        b.flashAttn(HparamsMaps.bool(m, b.flashAttn, "flash_attn", "flash_attention"));
        b.shiftAttn(HparamsMaps.bool(m, b.shiftAttn, "shift_attn", "shift_attention"));
        b.mixedPrecision(HparamsMaps.str(m, b.mixedPrecision, "mixed_precision", "precision"));
        String qm = HparamsMaps.strOrNull(m, "quantization_method", "quant_method");
        if (qm != null) b.quantizationMethod(QuantizationMethod.parse(qm));
        b.quantizationBit(HparamsMaps.integer(m, b.quantizationBit, "quantization_bit", "quant_bit"));
        b.doubleQuantization(HparamsMaps.bool(m, b.doubleQuantization, "double_quantization", "bnb_4bit_use_double_quant"));
        b.quantType(HparamsMaps.str(m, b.quantType, "quant_type", "bnb_4bit_quant_type"));
        b.lowCpuMemUsage(HparamsMaps.bool(m, b.lowCpuMemUsage, "low_cpu_mem_usage"));
        b.useUnsloth(HparamsMaps.bool(m, b.useUnsloth, "use_unsloth", "unsloth"));
        b.useLigerKernel(HparamsMaps.bool(m, b.useLigerKernel, "use_liger_kernel", "liger_kernel"));
        b.useKtransformers(HparamsMaps.bool(m, b.useKtransformers, "use_ktransformers", "ktransformers"));
        b.modelRevision(HparamsMaps.str(m, b.modelRevision, "model_revision", "revision"));
        b.trustRemoteCode(HparamsMaps.bool(m, b.trustRemoteCode, "trust_remote_code"));
        b.neftuneAlpha(HparamsMaps.dbl(m, b.neftuneAlpha, "neftune_alpha", "neftune_noise_alpha"));
        b.inferBackend(HparamsMaps.str(m, b.inferBackend, "infer_backend", "infer_backend_type"));
        b.vllmGpuMemoryUtilization(HparamsMaps.dbl(m, b.vllmGpuMemoryUtilization, "vllm_gpu_memory_utilization"));
        b.vllmMaxModelLen(HparamsMaps.integer(m, b.vllmMaxModelLen, "vllm_maxlen", "vllm_max_model_len"));
        return b.build();
    }

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private String modelNameOrPath = "gpt2";
        private String adapterNameOrPath;
        private List<String> adapterToMerge = new ArrayList<>();
        private boolean useFastTokenizer = true;
        private boolean resizeVocab;
        private RopeScalingType ropeScaling = RopeScalingType.NONE;
        private double ropeScalingFactor = 1.0;
        private boolean flashAttn;
        private boolean shiftAttn;
        private String mixedPrecision = "bf16";
        private QuantizationMethod quantizationMethod = QuantizationMethod.NONE;
        private int quantizationBit = 4;
        private boolean doubleQuantization = true;
        private String quantType = "nf4";
        private boolean lowCpuMemUsage = true;
        private boolean useUnsloth;
        private boolean useLigerKernel;
        private boolean useKtransformers;
        private String modelRevision = "main";
        private boolean trustRemoteCode;
        private double neftuneAlpha;
        private String inferBackend = "huggingface";
        private double vllmGpuMemoryUtilization = 0.9;
        private int vllmMaxModelLen;

        public Builder modelNameOrPath(String v) { this.modelNameOrPath = v; return this; }
        public Builder adapterNameOrPath(String v) { this.adapterNameOrPath = v; return this; }
        public Builder adapterToMerge(List<String> v) {
            this.adapterToMerge = v == null ? new ArrayList<>() : new ArrayList<>(v);
            return this;
        }
        public Builder useFastTokenizer(boolean v) { this.useFastTokenizer = v; return this; }
        public Builder resizeVocab(boolean v) { this.resizeVocab = v; return this; }
        public Builder ropeScaling(RopeScalingType v) { this.ropeScaling = v; return this; }
        public Builder ropeScalingFactor(double v) { this.ropeScalingFactor = v; return this; }
        public Builder flashAttn(boolean v) { this.flashAttn = v; return this; }
        public Builder shiftAttn(boolean v) { this.shiftAttn = v; return this; }
        public Builder mixedPrecision(String v) { this.mixedPrecision = v; return this; }
        public Builder quantizationMethod(QuantizationMethod v) { this.quantizationMethod = v; return this; }
        public Builder quantizationBit(int v) { this.quantizationBit = v; return this; }
        public Builder doubleQuantization(boolean v) { this.doubleQuantization = v; return this; }
        public Builder quantType(String v) { this.quantType = v; return this; }
        public Builder lowCpuMemUsage(boolean v) { this.lowCpuMemUsage = v; return this; }
        public Builder useUnsloth(boolean v) { this.useUnsloth = v; return this; }
        public Builder useLigerKernel(boolean v) { this.useLigerKernel = v; return this; }
        public Builder useKtransformers(boolean v) { this.useKtransformers = v; return this; }
        public Builder modelRevision(String v) { this.modelRevision = v; return this; }
        public Builder trustRemoteCode(boolean v) { this.trustRemoteCode = v; return this; }
        public Builder neftuneAlpha(double v) { this.neftuneAlpha = v; return this; }
        public Builder inferBackend(String v) { this.inferBackend = v; return this; }
        public Builder vllmGpuMemoryUtilization(double v) { this.vllmGpuMemoryUtilization = v; return this; }
        public Builder vllmMaxModelLen(int v) { this.vllmMaxModelLen = v; return this; }
        public ModelArgs build() { return new ModelArgs(this); }
    }
}
