package org.bytedeco.pytorch.serving.tensorrt.exceptions;

import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;

/**
 * Exception for failed engine build, deserialize, or enqueue execution.
 *
 * <p>Maps to {@code kFAILED_EXECUTION} / {@code kFAILED_INITIALIZATION} /
 * {@code kINVALID_CONFIG} depending on call site.
 */
public class ExecutionException extends TensorRTException {
    public ExecutionException(String message) {
        super(message, ErrorCode.FAILED_EXECUTION);
    }

    public ExecutionException(String message, ErrorCode code) {
        super(message, code);
    }

    public ExecutionException(String message, ErrorCode code, Throwable cause) {
        super(message, code, cause);
    }

    public ExecutionException(String message, Throwable cause) {
        super(message, ErrorCode.FAILED_EXECUTION, cause);
    }
}
