package org.bytedeco.pytorch.llm.quantization;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * BitsAndBytes quantization config aligned with HuggingFace
 * {@code transformers.BitsAndBytesConfig} field names and defaults.
 *
 * <pre>{@code
 * BitsAndBytesConfig cfg = BitsAndBytesConfig.builder()
 *     .loadIn4Bit(true)
 *     .bnb4BitQuantType("nf4")
 *     .bnb4BitUseDoubleQuant(true)
 *     .bnb4BitComputeDtype("bfloat16")
 *     .build();
 * }</pre>
 */
public final class BitsAndBytesConfig {
    private final boolean loadIn4Bit;
    private final boolean loadIn8Bit;
    private final String bnb4BitComputeDtype;
    private final boolean bnb4BitUseDoubleQuant;
    private final String bnb4BitQuantType;
    private final String bnb8BitQuantType;
    private final String deviceMap;
    private final double llmInt8Threshold;
    private final boolean llmInt8SkipModules;
    private final List<String> llmInt8SkipModuleNames;
    private final boolean llmInt8EnableFp32CpuOffload;
    private final boolean llmInt8HasFp16Weight;
    private final int bnb4BitQuantStorage; // bits of storage type hint (4 or 8)
    private final int blocksize;

    private BitsAndBytesConfig(Builder builder) {
        this.loadIn4Bit = builder.loadIn4Bit;
        this.loadIn8Bit = builder.loadIn8Bit;
        this.bnb4BitComputeDtype = builder.bnb4BitComputeDtype;
        this.bnb4BitUseDoubleQuant = builder.bnb4BitUseDoubleQuant;
        this.bnb4BitQuantType = builder.bnb4BitQuantType;
        this.bnb8BitQuantType = builder.bnb8BitQuantType;
        this.deviceMap = builder.deviceMap;
        this.llmInt8Threshold = builder.llmInt8Threshold;
        this.llmInt8SkipModules = builder.llmInt8SkipModules;
        this.llmInt8SkipModuleNames = Collections.unmodifiableList(new ArrayList<>(builder.llmInt8SkipModuleNames));
        this.llmInt8EnableFp32CpuOffload = builder.llmInt8EnableFp32CpuOffload;
        this.llmInt8HasFp16Weight = builder.llmInt8HasFp16Weight;
        this.bnb4BitQuantStorage = builder.bnb4BitQuantStorage;
        this.blocksize = builder.blocksize;
        validate();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** HF-style 4-bit QLoRA defaults (nf4 + double quant + bf16 compute). */
    public static BitsAndBytesConfig qloraDefaults() {
        return builder()
                .loadIn4Bit(true)
                .bnb4BitQuantType("nf4")
                .bnb4BitUseDoubleQuant(true)
                .bnb4BitComputeDtype("bfloat16")
                .build();
    }

    /** HF-style 8-bit load defaults. */
    public static BitsAndBytesConfig int8Defaults() {
        return builder()
                .loadIn8Bit(true)
                .llmInt8Threshold(6.0)
                .build();
    }

    private void validate() {
        if (loadIn4Bit && loadIn8Bit) {
            throw new IllegalArgumentException("loadIn4Bit and loadIn8Bit cannot both be true");
        }
        if (loadIn4Bit && !"nf4".equalsIgnoreCase(bnb4BitQuantType) && !"fp4".equalsIgnoreCase(bnb4BitQuantType)) {
            throw new IllegalArgumentException("bnb4BitQuantType must be one of [nf4, fp4]");
        }
        if (deviceMap == null || deviceMap.trim().isEmpty()) {
            throw new IllegalArgumentException("deviceMap must not be empty");
        }
        if (blocksize <= 0) {
            throw new IllegalArgumentException("blocksize must be positive");
        }
    }

    public boolean isLoadIn4Bit() {
        return loadIn4Bit;
    }

    public boolean isLoadIn8Bit() {
        return loadIn8Bit;
    }

    /** True when either 4-bit or 8-bit load is requested. */
    public boolean isQuantized() {
        return loadIn4Bit || loadIn8Bit;
    }

    public String getBnb4BitComputeDtype() {
        return bnb4BitComputeDtype;
    }

    public boolean isBnb4BitUseDoubleQuant() {
        return bnb4BitUseDoubleQuant;
    }

    public String getBnb4BitQuantType() {
        return bnb4BitQuantType;
    }

    public String getBnb8BitQuantType() {
        return bnb8BitQuantType;
    }

    public String getDeviceMap() {
        return deviceMap;
    }

    /** Outlier threshold for LLM.int8() (HF default 6.0). */
    public double getLlmInt8Threshold() {
        return llmInt8Threshold;
    }

    public boolean isLlmInt8SkipModules() {
        return llmInt8SkipModules;
    }

    public List<String> getLlmInt8SkipModuleNames() {
        return llmInt8SkipModuleNames;
    }

    public boolean isLlmInt8EnableFp32CpuOffload() {
        return llmInt8EnableFp32CpuOffload;
    }

    public boolean isLlmInt8HasFp16Weight() {
        return llmInt8HasFp16Weight;
    }

    public int getBnb4BitQuantStorage() {
        return bnb4BitQuantStorage;
    }

    /** Block size for blockwise quant (HF/bnb default 64). */
    public int getBlocksize() {
        return blocksize;
    }

    /** Whether {@code name} should be skipped for int8 quantization. */
    public boolean shouldSkipModule(String name) {
        if (name == null) return false;
        if (!llmInt8SkipModules && llmInt8SkipModuleNames.isEmpty()) return false;
        String lower = name.toLowerCase();
        // Always skip lm_head-like modules by HF convention when skip list empty but flag set
        if (llmInt8SkipModuleNames.isEmpty()) {
            return lower.endsWith("lm_head") || lower.equals("lm_head")
                    || lower.endsWith("embed_out") || lower.contains("lmhead");
        }
        for (String s : llmInt8SkipModuleNames) {
            if (s == null) continue;
            String t = s.toLowerCase();
            if (lower.equals(t) || lower.endsWith("." + t) || lower.endsWith("/" + t)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public String toString() {
        return "BitsAndBytesConfig{loadIn4Bit=" + loadIn4Bit
                + ", loadIn8Bit=" + loadIn8Bit
                + ", quantType=" + (loadIn4Bit ? bnb4BitQuantType : bnb8BitQuantType)
                + ", doubleQuant=" + bnb4BitUseDoubleQuant
                + ", computeDtype=" + bnb4BitComputeDtype
                + ", blocksize=" + blocksize
                + ", deviceMap=" + deviceMap + '}';
    }

    public static final class Builder {
        private boolean loadIn4Bit;
        private boolean loadIn8Bit;
        private String bnb4BitComputeDtype = "bfloat16";
        private boolean bnb4BitUseDoubleQuant = true;
        private String bnb4BitQuantType = "nf4";
        private String bnb8BitQuantType = "int8";
        private String deviceMap = "auto";
        private double llmInt8Threshold = 6.0;
        private boolean llmInt8SkipModules = false;
        private List<String> llmInt8SkipModuleNames = new ArrayList<>();
        private boolean llmInt8EnableFp32CpuOffload = false;
        private boolean llmInt8HasFp16Weight = false;
        private int bnb4BitQuantStorage = 4;
        private int blocksize = 64;

        private Builder() {
        }

        public Builder loadIn4Bit(boolean loadIn4Bit) {
            this.loadIn4Bit = loadIn4Bit;
            return this;
        }

        /** Snake alias matching Python {@code load_in_4bit}. */
        public Builder load_in_4bit(boolean v) {
            return loadIn4Bit(v);
        }

        public Builder loadIn8Bit(boolean loadIn8Bit) {
            this.loadIn8Bit = loadIn8Bit;
            return this;
        }

        public Builder load_in_8bit(boolean v) {
            return loadIn8Bit(v);
        }

        public Builder bnb4BitComputeDtype(String bnb4BitComputeDtype) {
            this.bnb4BitComputeDtype = Objects.requireNonNull(bnb4BitComputeDtype, "bnb4BitComputeDtype");
            return this;
        }

        public Builder bnb_4bit_compute_dtype(String v) {
            return bnb4BitComputeDtype(v);
        }

        public Builder bnb4BitUseDoubleQuant(boolean bnb4BitUseDoubleQuant) {
            this.bnb4BitUseDoubleQuant = bnb4BitUseDoubleQuant;
            return this;
        }

        public Builder bnb_4bit_use_double_quant(boolean v) {
            return bnb4BitUseDoubleQuant(v);
        }

        public Builder bnb4BitQuantType(String bnb4BitQuantType) {
            this.bnb4BitQuantType = Objects.requireNonNull(bnb4BitQuantType, "bnb4BitQuantType");
            return this;
        }

        public Builder bnb_4bit_quant_type(String v) {
            return bnb4BitQuantType(v);
        }

        public Builder bnb8BitQuantType(String bnb8BitQuantType) {
            this.bnb8BitQuantType = Objects.requireNonNull(bnb8BitQuantType, "bnb8BitQuantType");
            return this;
        }

        public Builder deviceMap(String deviceMap) {
            this.deviceMap = Objects.requireNonNull(deviceMap, "deviceMap");
            return this;
        }

        public Builder device_map(String v) {
            return deviceMap(v);
        }

        public Builder llmInt8Threshold(double llmInt8Threshold) {
            this.llmInt8Threshold = llmInt8Threshold;
            return this;
        }

        public Builder llm_int8_threshold(double v) {
            return llmInt8Threshold(v);
        }

        public Builder llmInt8SkipModules(boolean llmInt8SkipModules) {
            this.llmInt8SkipModules = llmInt8SkipModules;
            return this;
        }

        public Builder llmInt8SkipModuleNames(List<String> names) {
            this.llmInt8SkipModuleNames = names == null ? new ArrayList<>() : new ArrayList<>(names);
            return this;
        }

        public Builder llm_int8_skip_modules(String... names) {
            this.llmInt8SkipModuleNames = new ArrayList<>();
            if (names != null) {
                for (String n : names) if (n != null) this.llmInt8SkipModuleNames.add(n);
            }
            this.llmInt8SkipModules = !this.llmInt8SkipModuleNames.isEmpty();
            return this;
        }

        public Builder llmInt8EnableFp32CpuOffload(boolean v) {
            this.llmInt8EnableFp32CpuOffload = v;
            return this;
        }

        public Builder llm_int8_enable_fp32_cpu_offload(boolean v) {
            return llmInt8EnableFp32CpuOffload(v);
        }

        public Builder llmInt8HasFp16Weight(boolean v) {
            this.llmInt8HasFp16Weight = v;
            return this;
        }

        public Builder llm_int8_has_fp16_weight(boolean v) {
            return llmInt8HasFp16Weight(v);
        }

        public Builder bnb4BitQuantStorage(int bits) {
            this.bnb4BitQuantStorage = bits;
            return this;
        }

        public Builder blocksize(int blocksize) {
            this.blocksize = blocksize;
            return this;
        }

        public BitsAndBytesConfig build() {
            return new BitsAndBytesConfig(this);
        }
    }
}
