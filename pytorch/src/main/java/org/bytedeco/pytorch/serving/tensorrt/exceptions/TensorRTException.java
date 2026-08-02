package org.bytedeco.pytorch.serving.tensorrt.exceptions;

import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;

/**
 * Base exception for TensorRT high-level API errors.
 *
 * <p>Corresponds to failures surfaced via {@code nvinfer1::IErrorRecorder} /
 * {@code nvinfer1::ErrorCode}, and to Python exceptions raised by the
 * {@code tensorrt} package on failed builder/runtime calls.
 */
public class TensorRTException extends RuntimeException {
    private final ErrorCode errorCode;

    public TensorRTException(String message) {
        this(message, ErrorCode.UNSPECIFIED_ERROR, null);
    }

    public TensorRTException(String message, Throwable cause) {
        this(message, ErrorCode.UNSPECIFIED_ERROR, cause);
    }

    public TensorRTException(String message, ErrorCode errorCode) {
        this(message, errorCode, null);
    }

    public TensorRTException(String message, ErrorCode errorCode, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode == null ? ErrorCode.UNSPECIFIED_ERROR : errorCode;
    }

    public ErrorCode errorCode() {
        return errorCode;
    }
}
