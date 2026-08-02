package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/** Python {@code InternalError} / {@code TRITONSERVER_ERROR_INTERNAL}. */
public class TritonInternalException extends TritonException {
    public TritonInternalException(String message) {
        super(message);
    }

    public TritonInternalException(String message, Throwable cause) {
        super(message, cause);
    }

    public TritonInternalException(String message, int errorCode) {
        super(message, errorCode);
    }

    public TritonInternalException(String message, int errorCode, Throwable cause) {
        super(message, errorCode, cause);
    }
}
