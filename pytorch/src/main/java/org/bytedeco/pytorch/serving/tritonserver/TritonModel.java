package org.bytedeco.pytorch.serving.tritonserver;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequest;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Message;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;
import org.bytedeco.pytorch.serving.tritonserver.internal.InferCallbacks;
import org.bytedeco.pytorch.serving.tritonserver.internal.NativeError;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.tritonserver.global.tritonserver.*;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
/**
 * Model handle returned by {@link TServer#model}.
 *
 * <p>Corresponds to Python {@code tritonserver.Model}. A model handle is cheap to
 * construct (no native call) but most operations require a running server.
 */
public final class TritonModel {
    private final TServer TServer;
    private final String name;
    private final long version;
    private final String state;
    private final String reason;

    TritonModel(TServer TServer, String name, long version, String state, String reason) {
        this.TServer = Objects.requireNonNull(TServer, "server");
        this.name = Objects.requireNonNull(name, "name");
        this.version = version;
        this.state = state;
        this.reason = reason;
    }

    public TServer server() {
        return TServer;
    }

    public String name() {
        return name;
    }

    public long version() {
        return version;
    }

    public String state() {
        return state;
    }

    public String reason() {
        return reason;
    }

    /** Create a new request for this model. */
    public TritonInferenceRequest createRequest() {
        return new TritonInferenceRequest(this);
    }

    /**
     * Submit an inference request (sync-like iterator; wraps {@link TritonInferenceRequest}).
     *
     * <p>If {@code request} is passed, it is assumed to be already configured with
     * inputs/parameters and will be submitted directly. Otherwise a new request
     * is created from {@code inputs} map (common Python-style usage).
     */
    public TRTResponseIterator infer(TritonInferenceRequest request) {
        Objects.requireNonNull(request, "request");
        if (request.model() != this) {
            throw new TritonInvalidArgumentException("InferenceRequest bound to different model");
        }

        InferCallbacks.InferSink sink = new InferCallbacks.InferSink(this);
        TRITONSERVER_InferenceRequest nativeRequest =
                request.createNativeRequest(sink).request;
        TServer.inferAsync(nativeRequest);
        return new TRTResponseIterator(sink, true);
    }

    /**
     * Convenience: create and submit a request with inputs map (no parameters).
     * Equivalent to {@code model.createRequest().inputs(inputs).infer(request)}.
     */
    public TRTResponseIterator infer(Map<String, Object> inputs) {
        Objects.requireNonNull(inputs, "inputs");
        TritonInferenceRequest req = createRequest();
        for (Map.Entry<String, Object> e : inputs.entrySet()) {
            req.putInput(e.getKey(), e.getValue());
        }
        return infer(req);
    }

    /**
     * Create a request and populate it with inputs (chainable).
     * Equivalent to {@code model.createRequest().inputs(inputs)}.
     */
    public TritonInferenceRequest createRequestWithInputs(Map<String, Object> inputs) {
        Objects.requireNonNull(inputs, "inputs");
        TritonInferenceRequest req = createRequest();
        for (Map.Entry<String, Object> e : inputs.entrySet()) {
            req.putInput(e.getKey(), e.getValue());
        }
        return req;
    }

    public boolean ready() {
        TRITONSERVER_Server nativeServer = TServer.requireNative();
        boolean[] out = new boolean[1];
        NativeError.check(
                TRITONSERVER_ServerModelIsReady(nativeServer, name, version, out),
                "ServerModelIsReady");
        return out[0];
    }

    /** Model metadata as a JSON-derived map. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> metadata() {
        return metadata(null);
    }

    public Map<String, Object> metadata(Long configVersion) {
        TRITONSERVER_Server nativeServer = TServer.requireNative();
        TRITONSERVER_Message message = new TRITONSERVER_Message((Pointer) null);
        NativeError.check(
                TRITONSERVER_ServerModelMetadata(nativeServer, name, version, message),
                "ServerModelMetadata");
        try {
            Object parsed = TServer.messageToObject(message);
            if (parsed instanceof Map<?, ?> map) {
                return (Map<String, Object>) map;
            }
            Map<String, Object> wrap = new LinkedHashMap<>();
            wrap.put("value", parsed);
            return wrap;
        } finally {
            NativeError.check(TRITONSERVER_MessageDelete(message), "MessageDelete");
        }
    }

    /** Model config as a JSON-derived map. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> config() {
        TRITONSERVER_Server nativeServer = TServer.requireNative();
        TRITONSERVER_Message message = new TRITONSERVER_Message((Pointer) null);
        NativeError.check(
                TRITONSERVER_ServerModelConfig(nativeServer, name, version, 0, message),
                "ServerModelConfig");
        try {
            Object parsed = TServer.messageToObject(message);
            if (parsed instanceof Map<?, ?> map) {
                return (Map<String, Object>) map;
            }
            Map<String, Object> wrap = new LinkedHashMap<>();
            wrap.put("value", parsed);
            return wrap;
        } finally {
            NativeError.check(TRITONSERVER_MessageDelete(message), "MessageDelete");
        }
    }

    /** Model statistics as a JSON-derived map. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> statistics() {
        TRITONSERVER_Server nativeServer = TServer.requireNative();
        TRITONSERVER_Message message = new TRITONSERVER_Message((Pointer) null);
        NativeError.check(
                TRITONSERVER_ServerModelStatistics(nativeServer, name, version, message),
                "ServerModelStatistics");
        try {
            Object parsed = TServer.messageToObject(message);
            if (parsed instanceof Map<?, ?> map) {
                return (Map<String, Object>) map;
            }
            Map<String, Object> wrap = new LinkedHashMap<>();
            wrap.put("value", parsed);
            return wrap;
        } finally {
            NativeError.check(TRITONSERVER_MessageDelete(message), "MessageDelete");
        }
    }

    /**
     * Batch properties flags (bit mask from
     */
    public int batchProperties() {
        TRITONSERVER_Server nativeServer = TServer.requireNative();
        IntPointer flags = new IntPointer(1);
        PointerPointer<Pointer> voidp = new PointerPointer<>(1);
        NativeError.check(
                TRITONSERVER_ServerModelBatchProperties(nativeServer, name, version, flags, voidp),
                "ServerModelBatchProperties");
        return flags.get();
    }

    /**
     * Transaction properties flags (bit mask from }).
     */
    public int transactionProperties() {
        TRITONSERVER_Server nativeServer = TServer.requireNative();
        IntPointer flags = new IntPointer(1);
        PointerPointer<Pointer> voidp = new PointerPointer<>(1);
        NativeError.check(
                TRITONSERVER_ServerModelTransactionProperties(
                        nativeServer, name, version, flags, voidp),
                "ServerModelTransactionProperties");
        return flags.get();
    }

    /**
     * Unload this model. Convenience for {@link TServer#unload(TritonModel)}.
     */
    public void unload() {
        TServer.unload(this);
    }

    /**
     * Unload this model and dependents. Convenience for {@link TServer#unload(TritonModel, boolean)}.
     */
    public void unloadDependents() {
        TServer.unload(this, true);
    }

    @Override
    public String toString() {
        return "Model{name='" + name + "', version=" + version + "}";
    }
}
