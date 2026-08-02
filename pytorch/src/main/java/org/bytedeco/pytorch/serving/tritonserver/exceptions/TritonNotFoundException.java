package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/** Python {@code NotFoundError} / {@code TRITONSERVER_ERROR_NOT_FOUND}. */
public class TritonNotFoundException extends TritonException {
    public TritonNotFoundException(String message) {
        super(message);
    }

    public TritonNotFoundException(String message, int errorCode) {
        super(message, errorCode);
    }
}
