package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/**
 * Base exception for Triton in-process API errors.
 *
 * <p>Corresponds to Python {@code tritonserver.TritonError}.
 */
public class TritonException extends RuntimeException {
    private final int errorCode;

    public TritonException(String message) {
        this(message, -1, null);
    }

    public TritonException(String message, Throwable cause) {
        this(message, -1, cause);
    }

    public TritonException(String message, int errorCode) {
        this(message, errorCode, null);
    }

    public TritonException(String message, int errorCode, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    /** Native {@code TRITONSERVER_Error_Code}, or {@code -1} if not from C API. */
    public int errorCode() {
        return errorCode;
    }
}
