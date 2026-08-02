package org.bytedeco.pytorch.serving.tensorrt.exceptions;

import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;

/**
 * Exception for invalid arguments (API contract violations).
 *
 * <p>Maps to {@code nvinfer1::ErrorCode::kINVALID_ARGUMENT}.
 */
public class TrtInvalidArgumentException extends TensorRTException {
    public TrtInvalidArgumentException(String message) {
        super(message, ErrorCode.INVALID_ARGUMENT);
    }

    public TrtInvalidArgumentException(String message, Throwable cause) {
        super(message, ErrorCode.INVALID_ARGUMENT, cause);
    }
}
