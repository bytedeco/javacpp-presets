package org.bytedeco.pytorch.serving.tensorrt.exceptions;

import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;

/**
 * Exception for failed CUDA / TensorRT memory allocation.
 *
 * <p>Maps to {@code nvinfer1::ErrorCode::kFAILED_ALLOCATION}.
 */
public class TrtAllocationException extends TensorRTException {
    public TrtAllocationException(String message) {
        super(message, ErrorCode.FAILED_ALLOCATION);
    }

    public TrtAllocationException(String message, Throwable cause) {
        super(message, ErrorCode.FAILED_ALLOCATION, cause);
    }
}
