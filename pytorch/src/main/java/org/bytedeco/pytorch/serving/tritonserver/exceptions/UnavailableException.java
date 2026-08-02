package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/** Python {@code UnavailableError} / {@code TRITONSERVER_ERROR_UNAVAILABLE}. */
public class UnavailableException extends TritonException {
    public UnavailableException(String message) {
        super(message);
    }

    public UnavailableException(String message, int errorCode) {
        super(message, errorCode);
    }
}
