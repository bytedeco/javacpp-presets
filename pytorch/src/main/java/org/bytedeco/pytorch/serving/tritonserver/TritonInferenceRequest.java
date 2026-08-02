package org.bytedeco.pytorch.serving.tritonserver;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonMemoryType;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequest;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_ResponseAllocator;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;
import org.bytedeco.pytorch.serving.tritonserver.internal.InferCallbacks;
import org.bytedeco.pytorch.serving.tritonserver.internal.NativeError;
import org.bytedeco.pytorch.serving.tritonserver.internal.TRTResponseAllocators;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.tritonserver.global.tritonserver.*;

/**
 * Inference request bound to a {@link TritonModel}.
 *
 * <p>Corresponds to Python {@code tritonserver.InferenceRequest}. Inputs may be
 * {@link TritonTensor} or primitive arrays (converted via {@link TritonTensor#fromObject}).
 */
public final class TritonInferenceRequest {
    private final TritonModel tritonModel;
    private String requestId = "";
    private int flags;
    private long correlationId;
    private String correlationIdString;
    private long priority;
    private long timeoutMicros;
    private final Map<String, Object> inputs = new LinkedHashMap<>();
    private final Map<String, Object> parameters = new LinkedHashMap<>();
    private TritonMemoryType outputTritonMemoryType = TritonMemoryType.CPU;

    public TritonInferenceRequest(TritonModel tritonModel) {
        this.tritonModel = Objects.requireNonNull(tritonModel, "model");
    }

    public TritonModel model() {
        return tritonModel;
    }

    public String requestId() {
        return requestId;
    }

    public TritonInferenceRequest requestId(String id) {
        this.requestId = id == null ? "" : id;
        return this;
    }

    public int flags() {
        return flags;
    }

    public TritonInferenceRequest flags(int flags) {
        this.flags = flags;
        return this;
    }

    public long correlationId() {
        return correlationId;
    }

    public TritonInferenceRequest correlationId(long id) {
        this.correlationId = id;
        this.correlationIdString = null;
        return this;
    }

    public String correlationIdString() {
        return correlationIdString;
    }

    public TritonInferenceRequest correlationIdString(String id) {
        this.correlationIdString = id;
        return this;
    }

    public long priority() {
        return priority;
    }

    public TritonInferenceRequest priority(long priority) {
        this.priority = priority;
        return this;
    }

    public long timeoutMicros() {
        return timeoutMicros;
    }

    public TritonInferenceRequest timeoutMicros(long timeoutMicros) {
        this.timeoutMicros = timeoutMicros;
        return this;
    }

    /** Mutable inputs map: name → {@link TritonTensor} or primitive array. */
    public Map<String, Object> inputs() {
        return inputs;
    }

    public TritonInferenceRequest putInput(String name, Object value) {
        inputs.put(Objects.requireNonNull(name, "name"), Objects.requireNonNull(value, "value"));
        return this;
    }

    public Map<String, Object> parameters() {
        return parameters;
    }

    public TritonInferenceRequest putParameter(String key, Object value) {
        parameters.put(Objects.requireNonNull(key, "key"), value);
        return this;
    }

    public TritonMemoryType outputMemoryType() {
        return outputTritonMemoryType;
    }

    /**
     * Preferred output memory type. MVP allocator still falls back to CPU, but the
     * preference is retained for Phase 2/3 GPU paths.
     */
    public TritonInferenceRequest outputMemoryType(TritonMemoryType type) {
        this.outputTritonMemoryType = Objects.requireNonNull(type, "type");
        return this;
    }

    /**
     * Build native request, wire callbacks, and return the live sink.
     *
     * <p>Caller must keep the returned  reachable
     * until request release.
     */
    NativeBundle createNativeRequest(InferCallbacks.InferSink sink) {
        TServer TServer = tritonModel.server();
        TRITONSERVER_Server nativeServer = TServer.requireNative();

        TRITONSERVER_InferenceRequest request =
                new TRITONSERVER_InferenceRequest((Pointer) null);
        NativeError.check(
                TRITONSERVER_InferenceRequestNew(
                        request, nativeServer, tritonModel.name(), tritonModel.version()),
                "InferenceRequestNew");

        try {
            if (requestId != null && !requestId.isEmpty()) {
                NativeError.check(
                        TRITONSERVER_InferenceRequestSetId(request, requestId), "set request id");
            }
            if (flags != 0) {
                NativeError.check(
                        TRITONSERVER_InferenceRequestSetFlags(request, flags), "set flags");
            }
            if (correlationIdString != null && !correlationIdString.isEmpty()) {
                NativeError.check(
                        TRITONSERVER_InferenceRequestSetCorrelationIdString(
                                request, correlationIdString),
                        "set correlation id string");
            } else if (correlationId != 0L) {
                NativeError.check(
                        TRITONSERVER_InferenceRequestSetCorrelationId(request, correlationId),
                        "set correlation id");
            }
            if (priority != 0L) {
                NativeError.check(
                        TRITONSERVER_InferenceRequestSetPriorityUInt64(request, priority),
                        "set priority");
            }
            if (timeoutMicros != 0L) {
                NativeError.check(
                        TRITONSERVER_InferenceRequestSetTimeoutMicroseconds(request, timeoutMicros),
                        "set timeout");
            }

            // Materialize inputs as Tensors and keep them reachable.
            Map<String, TritonTensor> materialized = new LinkedHashMap<>();
            for (Map.Entry<String, Object> e : inputs.entrySet()) {
                TritonTensor t = TritonTensor.fromObject(e.getValue());
                materialized.put(e.getKey(), t);
                long[] shape = t.shape();
                NativeError.check(
                        TRITONSERVER_InferenceRequestAddInput(
                                request, e.getKey(), t.dataType().code(), shape, shape.length),
                        "AddInput " + e.getKey());
                Pointer base = pointerFor(t);
                NativeError.check(
                        TRITONSERVER_InferenceRequestAppendInputData(
                                request,
                                e.getKey(),
                                base,
                                t.size(),
                                t.memoryType().code(),
                                t.memoryTypeId()),
                        "AppendInputData " + e.getKey());
            }

            for (Map.Entry<String, Object> e : parameters.entrySet()) {
                setParameter(request, e.getKey(), e.getValue());
            }

            Pointer userp = InferCallbacks.retain(sink);
            sink.userpHandle = userp;
            sink.inputKeepAlive = materialized;

            NativeError.check(
                    TRITONSERVER_InferenceRequestSetReleaseCallback(
                            request, InferCallbacks.requestReleaseFn(), userp),
                    "SetReleaseCallback");

            TRITONSERVER_ResponseAllocator allocator = TRTResponseAllocators.sharedCpu();
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetResponseCallback(
                            request,
                            allocator,
                            userp,
                            InferCallbacks.responseCompleteFn(),
                            userp),
                    "SetResponseCallback");

            return new NativeBundle(request, sink, materialized, userp);
        } catch (RuntimeException ex) {
            // Best-effort delete; if callbacks not set yet Triton won't call release.
            try {
                org.bytedeco.tritonserver.global.tritonserver.TRITONSERVER_InferenceRequestDelete(
                        request);
            } catch (Throwable ignored) {
                // ignore
            }
            throw ex;
        }
    }

    private static void setParameter(TRITONSERVER_InferenceRequest request, String key, Object value) {
        if (value == null) {
            throw new TritonInvalidArgumentException("parameter '" + key + "' value must not be null");
        }
        if (value instanceof String s) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetStringParameter(request, key, s),
                    "set string param " + key);
        } else if (value instanceof Boolean b) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetBoolParameter(request, key, b),
                    "set bool param " + key);
        } else if (value instanceof Integer i) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetIntParameter(request, key, i.longValue()),
                    "set int param " + key);
        } else if (value instanceof Long l) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetIntParameter(request, key, l),
                    "set int param " + key);
        } else if (value instanceof Short s) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetIntParameter(request, key, s.longValue()),
                    "set int param " + key);
        } else if (value instanceof Byte b) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetIntParameter(request, key, b.longValue()),
                    "set int param " + key);
        } else if (value instanceof Double d) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetDoubleParameter(request, key, d),
                    "set double param " + key);
        } else if (value instanceof Float f) {
            NativeError.check(
                    TRITONSERVER_InferenceRequestSetDoubleParameter(request, key, f.doubleValue()),
                    "set double param " + key);
        } else {
            throw new TritonInvalidArgumentException(
                    "unsupported parameter type for '" + key + "': " + value.getClass().getName());
        }
    }

    private static Pointer pointerFor(TritonTensor t) {
        Object owner = t.memoryBuffer().owner();
        if (owner instanceof Pointer p && !p.isNull()) {
            return p;
        }
        long addr = t.dataPtr();
        long size = t.size();
        return new Pointer() {
            {
                address = addr;
                limit = size;
                capacity = size;
            }
        };
    }

    /** Package of native request + GC keep-alives for one infer submission. */
    static final class NativeBundle {
        final TRITONSERVER_InferenceRequest request;
        final InferCallbacks.InferSink sink;
        final Map<String, TritonTensor> inputs;
        final Pointer userp;

        NativeBundle(
                TRITONSERVER_InferenceRequest request,
                InferCallbacks.InferSink sink,
                Map<String, TritonTensor> inputs,
                Pointer userp) {
            this.request = request;
            this.sink = sink;
            this.inputs = inputs;
            this.userp = userp;
        }
    }
}
