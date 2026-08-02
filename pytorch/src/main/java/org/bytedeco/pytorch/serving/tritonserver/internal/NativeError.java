package org.bytedeco.pytorch.serving.tritonserver.internal;

import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Error;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
import static org.bytedeco.tritonserver.global.tritonserver.*;

/**
 * Maps {@code TRITONSERVER_Error*} to the Java exception hierarchy.
 *
 * <p>Corresponds to Python bindings that raise {@code TritonError} subclasses
 * from C API error codes.
 */
public final class NativeError {
    private NativeError() {}

    /**
     * If {@code err} is non-null, delete it and throw the matching exception.
     *
     * @param err result of a Triton C API call ({@code null} means success)
     * @param context short description used in the exception message prefix
     */
    public static void check(TRITONSERVER_Error err, String context) {
        if (err == null || err.isNull()) {
            return;
        }
        int code = TRITONSERVER_ErrorCode(err);
        String codeStr = TRITONSERVER_ErrorCodeString(err);
        String message = TRITONSERVER_ErrorMessage(err);
        TRITONSERVER_ErrorDelete(err);

        String full = context == null || context.isEmpty()
                ? codeStr + " - " + message
                : context + ": " + codeStr + " - " + message;
        throw create(code, full);
    }

    /** Convenience overload without context prefix. */
    public static void check(TRITONSERVER_Error err) {
        check(err, null);
    }

    /**
     * Build exception from an already-read code/message without owning a native error.
     * Used when parsing response-level errors that the caller still owns.
     */
    public static TritonException create(int code, String message) {
        return switch (code) {
            case TRITONSERVER_ERROR_NOT_FOUND -> new TritonNotFoundException(message, code);
            case TRITONSERVER_ERROR_INVALID_ARG -> new TritonInvalidArgumentException(message, code);
            case TRITONSERVER_ERROR_UNAVAILABLE -> new UnavailableException(message, code);
            case TRITONSERVER_ERROR_ALREADY_EXISTS -> new AlreadyExistsException(message, code);
            case TRITONSERVER_ERROR_UNSUPPORTED -> new UnsupportedException(message, code);
            case TRITONSERVER_ERROR_INTERNAL -> new TritonInternalException(message, code);
            case TRITONSERVER_ERROR_UNKNOWN -> new UnknownException(message, code);
            default -> new UnknownException(message, code);
        };
    }

    /**
     * Read error from a response without deleting the response.
     *
     * @return exception instance, or {@code null} if no error
     */
    public static TritonException fromResponseError(TRITONSERVER_Error err) {
        if (err == null || err.isNull()) {
            return null;
        }
        int code = TRITONSERVER_ErrorCode(err);
        String codeStr = TRITONSERVER_ErrorCodeString(err);
        String message = TRITONSERVER_ErrorMessage(err);
        // Do NOT delete: lifetime tied to the response object.
        return create(code, codeStr + " - " + message);
    }
}
