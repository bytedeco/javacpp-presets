package org.bytedeco.pytorch.serving.tensorrt;

import org.bytedeco.pytorch.serving.tensorrt.enums.BuilderFlag;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;

import java.util.*;

/**
 * High-level build / runtime options for TensorRT.
 *
 * <p>Corresponds to the knobs typically set on Python
 * {@code tensorrt.Builder} / {@code IBuilderConfig} / {@code IRuntime}
 * (workspace, precision flags, DLA core, device index). Defaults match common
 * TensorRT samples: 1 GiB workspace, no extra precision flags, device 0.
 */
public final class TrtOption {
    /** Default workspace pool size: 1 GiB ({@code MemoryPoolType.WORKSPACE}). */
    public static final long DEFAULT_WORKSPACE_BYTES = 1L << 30;

    private TRTLogger logger = TRTLogger.getDefaultLogger();
    private int deviceIndex = 0;
    private long workspaceBytes = DEFAULT_WORKSPACE_BYTES;
    private int maxThreads = 0;
    private int dlaCore = -1;
    private final Set<BuilderFlag> builderFlags = new LinkedHashSet<>();
    private boolean stronglyTyped = false;
    /** ONNX parser verbosity passed to {@code parseFromFile} (0–4 typical). */
    private int onnxVerbosity = 0;

    public TrtOption() {}

    public TrtOption(TrtOption other) {
        Objects.requireNonNull(other, "other");
        this.logger = other.logger;
        this.deviceIndex = other.deviceIndex;
        this.workspaceBytes = other.workspaceBytes;
        this.maxThreads = other.maxThreads;
        this.dlaCore = other.dlaCore;
        this.builderFlags.addAll(other.builderFlags);
        this.stronglyTyped = other.stronglyTyped;
        this.onnxVerbosity = other.onnxVerbosity;
    }

    public static Builder builder() {
        return new Builder();
    }

    public TRTLogger logger() {
        return logger;
    }

    public TrtOption logger(TRTLogger logger) {
        this.logger = Objects.requireNonNull(logger, "logger");
        return this;
    }

    public int deviceIndex() {
        return deviceIndex;
    }

    public TrtOption deviceIndex(int deviceIndex) {
        if (deviceIndex < 0) {
            throw new TrtInvalidArgumentException("deviceIndex must be >= 0");
        }
        this.deviceIndex = deviceIndex;
        return this;
    }

    public long workspaceBytes() {
        return workspaceBytes;
    }

    public TrtOption workspaceBytes(long workspaceBytes) {
        if (workspaceBytes < 0) {
            throw new TrtInvalidArgumentException("workspaceBytes must be >= 0");
        }
        this.workspaceBytes = workspaceBytes;
        return this;
    }

    public int maxThreads() {
        return maxThreads;
    }

    public TrtOption maxThreads(int maxThreads) {
        if (maxThreads < 0) {
            throw new TrtInvalidArgumentException("maxThreads must be >= 0");
        }
        this.maxThreads = maxThreads;
        return this;
    }

    public int dlaCore() {
        return dlaCore;
    }

    public TrtOption dlaCore(int dlaCore) {
        this.dlaCore = dlaCore;
        return this;
    }

    public Set<BuilderFlag> builderFlags() {
        return Collections.unmodifiableSet(builderFlags);
    }

    public TrtOption addBuilderFlag(BuilderFlag flag) {
        builderFlags.add(Objects.requireNonNull(flag, "flag"));
        return this;
    }

    public TrtOption enableFp16(boolean enable) {
        if (enable) {
            builderFlags.add(BuilderFlag.FP16);
        } else {
            builderFlags.remove(BuilderFlag.FP16);
        }
        return this;
    }

    public TrtOption enableInt8(boolean enable) {
        if (enable) {
            builderFlags.add(BuilderFlag.INT8);
        } else {
            builderFlags.remove(BuilderFlag.INT8);
        }
        return this;
    }

    public TrtOption enableBf16(boolean enable) {
        if (enable) {
            builderFlags.add(BuilderFlag.BF16);
        } else {
            builderFlags.remove(BuilderFlag.BF16);
        }
        return this;
    }

    public boolean stronglyTyped() {
        return stronglyTyped;
    }

    public TrtOption stronglyTyped(boolean stronglyTyped) {
        this.stronglyTyped = stronglyTyped;
        return this;
    }

    public int onnxVerbosity() {
        return onnxVerbosity;
    }

    public TrtOption onnxVerbosity(int onnxVerbosity) {
        this.onnxVerbosity = onnxVerbosity;
        return this;
    }

    public List<BuilderFlag> builderFlagList() {
        return new ArrayList<>(builderFlags);
    }

    public static final class Builder {
        private final TrtOption trtOptions = new TrtOption();

        public Builder logger(TRTLogger logger) {
            trtOptions.logger(logger);
            return this;
        }

        public Builder deviceIndex(int deviceIndex) {
            trtOptions.deviceIndex(deviceIndex);
            return this;
        }

        public Builder workspaceBytes(long workspaceBytes) {
            trtOptions.workspaceBytes(workspaceBytes);
            return this;
        }

        public Builder maxThreads(int maxThreads) {
            trtOptions.maxThreads(maxThreads);
            return this;
        }

        public Builder dlaCore(int dlaCore) {
            trtOptions.dlaCore(dlaCore);
            return this;
        }

        public Builder addBuilderFlag(BuilderFlag flag) {
            trtOptions.addBuilderFlag(flag);
            return this;
        }

        public Builder enableFp16(boolean enable) {
            trtOptions.enableFp16(enable);
            return this;
        }

        public Builder enableInt8(boolean enable) {
            trtOptions.enableInt8(enable);
            return this;
        }

        public Builder enableBf16(boolean enable) {
            trtOptions.enableBf16(enable);
            return this;
        }

        public Builder stronglyTyped(boolean stronglyTyped) {
            trtOptions.stronglyTyped(stronglyTyped);
            return this;
        }

        public Builder onnxVerbosity(int onnxVerbosity) {
            trtOptions.onnxVerbosity(onnxVerbosity);
            return this;
        }

        public TrtOption build() {
            return new TrtOption(trtOptions);
        }
    }
}
