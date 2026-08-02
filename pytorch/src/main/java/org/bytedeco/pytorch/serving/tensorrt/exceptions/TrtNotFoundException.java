package org.bytedeco.pytorch.serving.tensorrt.exceptions;

import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;

/**
 * Exception thrown when a requested tensor or engine is not found.
 */
public class TrtNotFoundException extends TensorRTException {
    public TrtNotFoundException(String message) {
        super(message, ErrorCode.UNSPECIFIED_ERROR);
    }

    public TrtNotFoundException(String message, Throwable cause) {
        super(message, ErrorCode.UNSPECIFIED_ERROR, cause);
    }
}
