package org.bytedeco.pytorch.serving.tritonserver.exceptions;

/** Python {@code UnsupportedError} / {@code TRITONSERVER_ERROR_UNSUPPORTED}. */
public class UnsupportedException extends TritonException {
    public UnsupportedException(String message) {
        super(message);
    }

    public UnsupportedException(String message, int errorCode) {
        super(message, errorCode);
    }
}
