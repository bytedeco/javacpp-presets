package org.bytedeco.pytorch.serving.tensorrt;

import org.bytedeco.pytorch.serving.tensorrt.enums.TRTDataType;
import org.bytedeco.pytorch.serving.tensorrt.enums.TRTTensorIOMode;

import java.util.Arrays;
import java.util.Objects;

/**
 * Metadata for one engine I/O tensor.
 *
 * <p>Filled from {@code ICudaEngine.getIOTensorName} /
 * {@code getTensorIOMode} / {@code getTensorShape} / {@code getTensorDataType}.
 */
public final class TRTTensorInfo {
    private final String name;
    private final TRTTensorIOMode ioMode;
    private final TRTDataType dataType;
    private final long[] shape;

    public TRTTensorInfo(String name, TRTTensorIOMode ioMode, TRTDataType dataType, long[] shape) {
        this.name = Objects.requireNonNull(name, "name");
        this.ioMode = Objects.requireNonNull(ioMode, "ioMode");
        this.dataType = Objects.requireNonNull(dataType, "dataType");
        this.shape = Arrays.copyOf(Objects.requireNonNull(shape, "shape"), shape.length);
    }

    public String name() {
        return name;
    }

    public TRTTensorIOMode ioMode() {
        return ioMode;
    }

    public TRTDataType dataType() {
        return dataType;
    }

    public long[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public boolean input() {
        return ioMode == TRTTensorIOMode.INPUT;
    }

    public boolean output() {
        return ioMode == TRTTensorIOMode.OUTPUT;
    }

    @Override
    public String toString() {
        return "TensorInfo{name='" + name + "', io=" + ioMode
                + ", type=" + dataType + ", shape=" + Arrays.toString(shape) + '}';
    }
}
