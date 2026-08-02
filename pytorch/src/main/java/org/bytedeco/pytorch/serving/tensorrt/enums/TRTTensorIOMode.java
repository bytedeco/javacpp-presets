package org.bytedeco.pytorch.serving.tensorrt.enums;

import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;

/**
 * Tensor I/O mode.
 *
 * <p>Matches {@code nvinfer1::TensorIOMode} / Python {@code tensorrt.TensorIOMode}.
 */
public enum TRTTensorIOMode {
    NONE(0),
    INPUT(1),
    OUTPUT(2);

    private final int code;

    TRTTensorIOMode(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static TRTTensorIOMode fromCode(int code) {
        for (TRTTensorIOMode m : values()) {
            if (m.code == code) {
                return m;
            }
        }
        throw new TrtInvalidArgumentException("Unknown TensorIOMode code: " + code);
    }
}
