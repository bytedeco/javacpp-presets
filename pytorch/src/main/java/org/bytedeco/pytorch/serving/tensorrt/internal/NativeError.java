package org.bytedeco.pytorch.serving.tensorrt.internal;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.tensorrt.nvinfer.IHostMemory;
import org.bytedeco.tensorrt.nvonnxparser.IParser;
import org.bytedeco.tensorrt.nvonnxparser.IParserError;
import org.bytedeco.pytorch.serving.tensorrt.enums.ErrorCode;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.*;

import static org.bytedeco.cuda.global.cudart.cudaSuccess;

/**
 * Native / CUDA error helpers for the high-level TensorRT API.
 *
 * <p>TensorRT C++ APIs generally return {@code null}/false rather than
 * {@code TRITONSERVER_Error*}; ONNX parser errors are enumerated via
 * {@code IParser::getNbErrors}. CUDA uses {@code cudaError_t}.
 */
public final class NativeError {
    private NativeError() {}

    public static void checkCuda(int cudaError, String context) {
        if (cudaError == cudaSuccess) {
            return;
        }
        String full = (context == null || context.isEmpty())
                ? "CUDA error code " + cudaError
                : context + ": CUDA error code " + cudaError;
        throw new TrtAllocationException(full);
    }

    public static void requireNonNull(Pointer ptr, String context) {
        if (ptr == null || ptr.isNull()) {
            throw new ExecutionException(
                    context + ": native object is null",
                    ErrorCode.FAILED_INITIALIZATION);
        }
    }

    public static void requireTrue(boolean ok, String context) {
        if (!ok) {
            throw new ExecutionException(context + ": returned false", ErrorCode.FAILED_EXECUTION);
        }
    }

    /**
     * After a failed {@code IParser.parse*} call, collect parser diagnostics.
     */
    public static void throwOnParserErrors(IParser parser, String context) {
        if (parser == null || parser.isNull()) {
            throw new ExecutionException(context + ": ONNX parser is null", ErrorCode.FAILED_INITIALIZATION);
        }
        int n = parser.getNbErrors();
        if (n <= 0) {
            throw new ExecutionException(context + ": ONNX parse failed (no parser diagnostics)",
                    ErrorCode.INVALID_CONFIG);
        }
        StringBuilder sb = new StringBuilder(context).append(": ONNX parse failed:");
        for (int i = 0; i < n; i++) {
            IParserError err = parser.getError(i);
            if (err == null || err.isNull()) {
                continue;
            }
            sb.append("\n  [").append(i).append("] ");
            String desc = err.desc();
            if (desc != null) {
                sb.append(desc);
            }
            Object code = err.code();
            sb.append(" (code=").append(code == null ? "?" : code.toString())
                    .append(", file=").append(err.file())
                    .append(", line=").append(err.line())
                    .append(", func=").append(err.func()).append(')');
        }
        throw new ExecutionException(sb.toString(), ErrorCode.INVALID_CONFIG);
    }

    public static void requireHostMemory(IHostMemory mem, String context) {
        requireNonNull(mem, context);
        if (mem.data() == null || mem.data().isNull() || mem.size() <= 0) {
            throw new ExecutionException(context + ": empty serialized engine", ErrorCode.FAILED_EXECUTION);
        }
    }

    public static TensorRTException map(ErrorCode code, String message) {
        return switch (code) {
            case INVALID_ARGUMENT -> new TrtInvalidArgumentException(message);
            case INTERNAL_ERROR -> new InternalException(message);
            case FAILED_ALLOCATION -> new TrtAllocationException(message);
            case FAILED_EXECUTION, FAILED_COMPUTATION, FAILED_INITIALIZATION,
                    INVALID_CONFIG, INVALID_STATE, UNSUPPORTED_STATE ->
                    new ExecutionException(message, code);
            default -> new TensorRTException(message, code);
        };
    }
}
