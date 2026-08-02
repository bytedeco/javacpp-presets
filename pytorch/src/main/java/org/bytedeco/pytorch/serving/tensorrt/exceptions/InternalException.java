package org.bytedeco.pytorch.serving.tensorrt.exceptions;

import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;

/**
 * Exception for internal TensorRT failures.
 *
 * <p>Maps to {@code nvinfer1::ErrorCode::kINTERNAL_ERROR}.
 */
public class InternalException extends TensorRTException {
    public InternalException(String message) {
        super(message, ErrorCode.INTERNAL_ERROR);
    }

    public InternalException(String message, Throwable cause) {
        super(message, ErrorCode.INTERNAL_ERROR, cause);
    }
}
