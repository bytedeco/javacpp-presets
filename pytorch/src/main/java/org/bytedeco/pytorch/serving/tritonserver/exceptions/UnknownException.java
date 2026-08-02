package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/** Python {@code UnknownError} / {@code TRITONSERVER_ERROR_UNKNOWN}. */
public class UnknownException extends TritonException {
    public UnknownException(String message) {
        super(message);
    }

    public UnknownException(String message, int errorCode) {
        super(message, errorCode);
    }
}
