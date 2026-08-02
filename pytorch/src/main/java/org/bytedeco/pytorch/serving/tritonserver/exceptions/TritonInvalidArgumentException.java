package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/** Python {@code InvalidArgumentError} / {@code TRITONSERVER_ERROR_INVALID_ARG}. */
public class TritonInvalidArgumentException extends TritonException {
    public TritonInvalidArgumentException(String message) {
        super(message);
    }

    public TritonInvalidArgumentException(String message, int errorCode) {
        super(message, errorCode);
    }
}
