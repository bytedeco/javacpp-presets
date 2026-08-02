package org.bytedeco.pytorch.serving.tensorrt.enums;

/**
 * TensorRT error codes.
 *
 * <p>Matches {@code nvinfer1::ErrorCode} / Python {@code tensorrt.ErrorCode}
 * (bytedeco {@code org.bytedeco.tensorrt.global.nvinfer.ErrorCode}).
 */
public enum ErrorCode {
    SUCCESS(0),
    UNSPECIFIED_ERROR(1),
    INTERNAL_ERROR(2),
    INVALID_ARGUMENT(3),
    INVALID_CONFIG(4),
    FAILED_ALLOCATION(5),
    FAILED_INITIALIZATION(6),
    FAILED_EXECUTION(7),
    FAILED_COMPUTATION(8),
    INVALID_STATE(9),
    UNSUPPORTED_STATE(10);

    private final int code;

    ErrorCode(int code) {
        this.code = code;
    }

    public int code() {
        return code;
    }

    public static ErrorCode fromCode(int code) {
        for (ErrorCode e : values()) {
            if (e.code == code) {
                return e;
            }
        }
        return UNSPECIFIED_ERROR;
    }
}
