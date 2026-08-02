package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/** Python {@code AlreadyExistsError} / {@code TRITONSERVER_ERROR_ALREADY_EXISTS}. */
public class AlreadyExistsException extends TritonException {
    public AlreadyExistsException(String message) {
        super(message);
    }

    public AlreadyExistsException(String message, int errorCode) {
        super(message, errorCode);
    }
}
