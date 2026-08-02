package org.bytedeco.pytorch.serving.tritonserver;
import org.bytedeco.pytorch.nn.*;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonMetricFormat;
import org.bytedeco.tritonserver.tritonserver.*;
import org.bytedeco.pytorch.serving.tritonserver.internal.NativeError;

import java.util.*;

import static org.bytedeco.tritonserver.global.tritonserver.*;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
/**
 * In-process Triton Inference Server handle.
 *
 * <p>Corresponds to Python {@code tritonserver.Server}. Lifecycle:
 *
 * <pre>
 * [Unstarted] --start()--> [Running] --stop()--> [Unstarted]
 * </pre>
 *
 * <p>Repeat {@link #start} while running raises {@link TritonInvalidArgumentException}. After {@link
 * #stop}, {@link #start} may be called again (new native server).
 */
public final class TServer implements AutoCloseable {
    private final TritonOption tritonOptions;
    private volatile TRITONSERVER_Server nativeServer;
    private volatile boolean started;

    public TServer(TritonOption tritonOptions) {
        this.tritonOptions = new TritonOption(Objects.requireNonNull(tritonOptions, "options"));
    }

    /** Fluent entry: {@code Server.builder().modelRepository(...).build()} then wrap. */
    public static TritonOption.Builder builder() {
        return TritonOption.builder();
    }

    public TritonOption options() {
        return new TritonOption(tritonOptions);
    }

    public boolean isStarted() {
        return started;
    }

    /** Start and wait until ready (0.1s poll, 30s timeout). */
    public void start() {
        start(true, 0.1, 30.0);
    }

    /** Start with readiness wait control; default poll 0.1s / timeout 30s. */
    public void start(boolean waitUntilReady) {
        start(waitUntilReady, 0.1, 30.0);
    }

    /**
     * Start the server.
     *
     * @param waitUntilReady poll {@link #ready()} until true or timeout
     * @param pollingIntervalSec interval between readiness polls
     * @param timeoutSec max wait seconds when {@code waitUntilReady}; &le;0 means unbounded
     */
    public synchronized void start(
            boolean waitUntilReady, double pollingIntervalSec, double timeoutSec) {
        if (started) {
            throw new TritonInvalidArgumentException("Server already started");
        }

        TRITONSERVER_ServerOptions nativeOptions = tritonOptions.createNativeOptions();
        TRITONSERVER_Server server = new TRITONSERVER_Server((Pointer) null);
        try {
            NativeError.check(TRITONSERVER_ServerNew(server, nativeOptions), "ServerNew");
        } catch (RuntimeException ex) {
            safeDeleteOptions(nativeOptions);
            throw ex;
        }
        // Options are consumed/copied by ServerNew; free immediately.
        safeDeleteOptions(nativeOptions);

        this.nativeServer = server;
        this.started = true;

        if (waitUntilReady) {
            try {
                waitReady(pollingIntervalSec, timeoutSec);
            } catch (RuntimeException ex) {
                // Roll back to unstarted so caller can retry or inspect.
                try {
                    stop();
                } catch (RuntimeException ignored) {
                    // prefer original failure
                }
                throw ex;
            }
        }
    }

    private void waitReady(double pollingIntervalSec, double timeoutSec) {
        long intervalMs = Math.max(1L, Math.round(pollingIntervalSec * 1000.0));
        long deadlineNs =
                timeoutSec > 0
                        ? System.nanoTime() + Math.round(timeoutSec * 1_000_000_000.0)
                        : Long.MAX_VALUE;

        while (!ready()) {
            if (System.nanoTime() >= deadlineNs) {
                throw new UnavailableException(
                        "Timed out waiting for server to become ready (timeoutSec="
                                + timeoutSec
                                + ")");
            }
            try {
                Thread.sleep(intervalMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new TritonInternalException("interrupted while waiting for server ready", e);
            }
        }
    }

    /**
     * Stop and delete the native server. Returns to unstarted; may {@link #start} again. Aligns
     * with Python: stop replaces running instance with unstarted sentinel.
     */
    public synchronized void stop() {
        if (!started || nativeServer == null) {
            started = false;
            nativeServer = null;
            return;
        }
        TRITONSERVER_Server server = nativeServer;
        try {
            try {
                NativeError.check(TRITONSERVER_ServerStop(server), "ServerStop");
            } catch (RuntimeException stopEx) {
                // Still attempt delete so we do not leak the native handle.
                try {
                    NativeError.check(TRITONSERVER_ServerDelete(server), "ServerDelete");
                } catch (RuntimeException ignored) {
                    // prefer stop failure
                }
                throw stopEx;
            }
            NativeError.check(TRITONSERVER_ServerDelete(server), "ServerDelete");
        } finally {
            nativeServer = null;
            started = false;
        }
    }

    @Override
    public void close() {
        stop();
    }

    public boolean live() {
        TRITONSERVER_Server server = requireNative();
        boolean[] out = new boolean[1];
        NativeError.check(TRITONSERVER_ServerIsLive(server, out), "ServerIsLive");
        return out[0];
    }

    public boolean ready() {
        TRITONSERVER_Server server = requireNative();
        boolean[] out = new boolean[1];
        NativeError.check(TRITONSERVER_ServerIsReady(server, out), "ServerIsReady");
        return out[0];
    }

    /** Server metadata as a JSON-derived map (Python returns dict). */
    @SuppressWarnings("unchecked")
    public Map<String, Object> metadata() {
        TRITONSERVER_Server server = requireNative();
        TRITONSERVER_Message message = new TRITONSERVER_Message((Pointer) null);
        NativeError.check(TRITONSERVER_ServerMetadata(server, message), "ServerMetadata");
        try {
            Object parsed = messageToObject(message);
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
     * Lightweight model handle (does not query readiness).
     *
     * @param name model name
     * @param version model version; {@code -1} means server-selected / latest
     */
    public TritonModel model(String name, long version) {
        Objects.requireNonNull(name, "name");
        if (name.isEmpty()) {
            throw new TritonInvalidArgumentException("model name must not be empty");
        }
        return new TritonModel(this, name, version, null, null);
    }

    public TritonModel model(String name) {
        return model(name, -1L);
    }

    /**
     * Index of models known to the server.
     *
     * @param excludeNotReady if true, only READY models ({@code INDEX_FLAG_READY})
     */
    public TritonModelDictionary models(boolean excludeNotReady) {
        TRITONSERVER_Server server = requireNative();
        int flags = excludeNotReady ? TRITONSERVER_INDEX_FLAG_READY : 0;
        TRITONSERVER_Message message = new TRITONSERVER_Message((Pointer) null);
        NativeError.check(TRITONSERVER_ServerModelIndex(server, flags, message), "ServerModelIndex");
        try {
            return modelDictionaryFromIndex(messageToObject(message));
        } finally {
            NativeError.check(TRITONSERVER_MessageDelete(message), "MessageDelete");
        }
    }

    public TritonModelDictionary models() {
        return models(false);
    }

    public TritonModel load(String name) {
        Objects.requireNonNull(name, "name");
        TRITONSERVER_Server server = requireNative();
        NativeError.check(TRITONSERVER_ServerLoadModel(server, name), "ServerLoadModel");
        return model(name, -1L);
    }

    /** Load with string/bool/int/double parameters (Python {@code load(name, parameters=...)}). */
    public TritonModel load(String name, Map<String, Object> parameters) {
        Objects.requireNonNull(name, "name");
        if (parameters == null || parameters.isEmpty()) {
            return load(name);
        }
        TRITONSERVER_Server server = requireNative();
        TRITONSERVER_Parameter[] params = new TRITONSERVER_Parameter[parameters.size()];
        // Keep value pointers reachable while native Parameter holds them.
        Pointer[] valueKeepAlive = new Pointer[parameters.size()];
        int i = 0;
        try {
            for (Map.Entry<String, Object> e : parameters.entrySet()) {
                Object v = e.getValue();
                if (v == null) {
                    throw new TritonInvalidArgumentException(
                            "load parameter '" + e.getKey() + "' is null");
                }
                if (v instanceof String s) {
                    BytePointer sp = new BytePointer(s);
                    valueKeepAlive[i] = sp;
                    params[i] =
                            TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_STRING, sp);
                } else if (v instanceof Boolean b) {
                    BoolPointer bp = new BoolPointer(1);
                    bp.put(b);
                    valueKeepAlive[i] = bp;
                    params[i] =
                            TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_BOOL, bp);
                } else if (v instanceof Integer n) {
                    LongPointer lp = new LongPointer(1);
                    lp.put(n.longValue());
                    valueKeepAlive[i] = lp;
                    params[i] = TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_INT, lp);
                } else if (v instanceof Long n) {
                    LongPointer lp = new LongPointer(1);
                    lp.put(n);
                    valueKeepAlive[i] = lp;
                    params[i] = TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_INT, lp);
                } else if (v instanceof Short n) {
                    LongPointer lp = new LongPointer(1);
                    lp.put(n.longValue());
                    valueKeepAlive[i] = lp;
                    params[i] = TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_INT, lp);
                } else if (v instanceof Byte n) {
                    LongPointer lp = new LongPointer(1);
                    lp.put(n.longValue());
                    valueKeepAlive[i] = lp;
                    params[i] = TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_INT, lp);
                } else if (v instanceof Double d) {
                    DoublePointer dp = new DoublePointer(1);
                    dp.put(d);
                    valueKeepAlive[i] = dp;
                    params[i] =
                            TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_DOUBLE, dp);
                } else if (v instanceof Float f) {
                    DoublePointer dp = new DoublePointer(1);
                    dp.put(f.doubleValue());
                    valueKeepAlive[i] = dp;
                    params[i] =
                            TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_DOUBLE, dp);
                } else {
                    throw new TritonInvalidArgumentException(
                            "unsupported load parameter type for '"
                                    + e.getKey()
                                    + "': "
                                    + v.getClass().getName());
                }
                if (params[i] == null || params[i].isNull()) {
                    throw new TritonInternalException("ParameterNew failed for '" + e.getKey() + "'");
                }
                i++;
            }
            NativeError.check(
                    TRITONSERVER_ServerLoadModelWithParameters(
                            server, name, params[0], params.length),
                    "ServerLoadModelWithParameters");
            return model(name, -1L);
        } finally {
            for (TRITONSERVER_Parameter p : params) {
                if (p != null && !p.isNull()) {
                    try {
                        TRITONSERVER_ParameterDelete(p);
                    } catch (Throwable ignored) {
                        // best-effort
                    }
                }
            }
            // valueKeepAlive kept until parameters deleted
            for (int k = 0; k < valueKeepAlive.length; k++) {
                valueKeepAlive[k] = null;
            }
        }
    }

    public void unload(String name) {
        unload(name, false);
    }

    public void unload(TritonModel tritonModel) {
        Objects.requireNonNull(tritonModel, "model");
        unload(tritonModel.name(), false);
    }

    public void unload(String name, boolean unloadDependents) {
        Objects.requireNonNull(name, "name");
        TRITONSERVER_Server server = requireNative();
        if (unloadDependents) {
            NativeError.check(
                    TRITONSERVER_ServerUnloadModelAndDependents(server, name),
                    "ServerUnloadModelAndDependents");
        } else {
            NativeError.check(TRITONSERVER_ServerUnloadModel(server, name), "ServerUnloadModel");
        }
    }

    public void unload(TritonModel tritonModel, boolean unloadDependents) {
        Objects.requireNonNull(tritonModel, "model");
        unload(tritonModel.name(), unloadDependents);
    }

    /** Formatted metrics string (default Prometheus). */
    public String metrics() {
        return metrics(TritonMetricFormat.PROMETHEUS);
    }

    public String metrics(TritonMetricFormat format) {
        Objects.requireNonNull(format, "format");
        TRITONSERVER_Server server = requireNative();
        TRITONSERVER_Metrics metrics = new TRITONSERVER_Metrics((Pointer) null);
        NativeError.check(TRITONSERVER_ServerMetrics(server, metrics), "ServerMetrics");
        try {
            BytePointer base = new BytePointer((Pointer) null);
            SizeTPointer byteSize = new SizeTPointer(1);
            NativeError.check(
                    TRITONSERVER_MetricsFormatted(metrics, format.code(), base, byteSize),
                    "MetricsFormatted");
            return readCString(base, byteSize.get());
        } finally {
            NativeError.check(TRITONSERVER_MetricsDelete(metrics), "MetricsDelete");
        }
    }

    public void pollModelRepository() {
        TRITONSERVER_Server server = requireNative();
        NativeError.check(
                TRITONSERVER_ServerPollModelRepository(server), "ServerPollModelRepository");
    }

    /**
     * Register an additional model repository (EXPLICIT control mode).
     *
     * @param repositoryPath filesystem path
     * @param nameMapping optional original→override name map; may be null/empty
     */
    public void registerModelRepository(String repositoryPath, Map<String, String> nameMapping) {
        Objects.requireNonNull(repositoryPath, "repositoryPath");
        TRITONSERVER_Server server = requireNative();
        if (nameMapping == null || nameMapping.isEmpty()) {
            PointerPointer<Pointer> nullMap = new PointerPointer<>(1);
            nullMap.put(0, (Pointer) null);
            NativeError.check(
                    TRITONSERVER_ServerRegisterModelRepository(server, repositoryPath, nullMap, 0),
                    "ServerRegisterModelRepository");
            return;
        }

        TRITONSERVER_Parameter[] params = new TRITONSERVER_Parameter[nameMapping.size()];
        Pointer[] keep = new Pointer[nameMapping.size()];
        int i = 0;
        try {
            for (Map.Entry<String, String> e : nameMapping.entrySet()) {
                BytePointer value = new BytePointer(e.getValue() == null ? "" : e.getValue());
                keep[i] = value;
                params[i] =
                        TRITONSERVER_ParameterNew(e.getKey(), TRITONSERVER_PARAMETER_STRING, value);
                if (params[i] == null || params[i].isNull()) {
                    throw new TritonInternalException(
                            "ParameterNew failed for mapping '" + e.getKey() + "'");
                }
                i++;
            }
            NativeError.check(
                    TRITONSERVER_ServerRegisterModelRepository(
                            server, repositoryPath, params[0], params.length),
                    "ServerRegisterModelRepository");
        } finally {
            for (TRITONSERVER_Parameter p : params) {
                if (p != null && !p.isNull()) {
                    try {
                        TRITONSERVER_ParameterDelete(p);
                    } catch (Throwable ignored) {
                        // best-effort
                    }
                }
            }
        }
    }

    public void registerModelRepository(String repositoryPath) {
        registerModelRepository(repositoryPath, null);
    }

    public void unregisterModelRepository(String repositoryPath) {
        Objects.requireNonNull(repositoryPath, "repositoryPath");
        TRITONSERVER_Server server = requireNative();
        NativeError.check(
                TRITONSERVER_ServerUnregisterModelRepository(server, repositoryPath),
                "ServerUnregisterModelRepository");
    }

    /** Submit inference asynchronously. Package-private; used by {@link TritonModel#infer}. */
    void inferAsync(TRITONSERVER_InferenceRequest request) {
        TRITONSERVER_Server server = requireNative();
        NativeError.check(
                TRITONSERVER_ServerInferAsync(
                        server, request, (TRITONSERVER_InferenceTrace) null),
                "ServerInferAsync");
    }

    /** Native server pointer; throws if not started. */
    TRITONSERVER_Server requireNative() {
        TRITONSERVER_Server server = nativeServer;
        if (!started || server == null || server.isNull()) {
            throw new TritonInvalidArgumentException("Server not started");
        }
        return server;
    }

    private static void safeDeleteOptions(TRITONSERVER_ServerOptions options) {
        if (options == null || options.isNull()) {
            return;
        }
        try {
            NativeError.check(TRITONSERVER_ServerOptionsDelete(options), "ServerOptionsDelete");
        } catch (Throwable ignored) {
            // best-effort cleanup
        }
    }

    private TritonModelDictionary modelDictionaryFromIndex(Object parsed) {
        List<?> list;
        if (parsed instanceof List<?> l) {
            list = l;
        } else if (parsed instanceof Map<?, ?> map) {
            Object models = map.get("models");
            if (models == null) {
                models = map.get("model");
            }
            if (models instanceof List<?> l) {
                list = l;
            } else {
                return TritonModelDictionary.empty();
            }
        } else {
            return TritonModelDictionary.empty();
        }

        Map<String, TritonModel> out = new LinkedHashMap<>();
        for (Object item : list) {
            if (!(item instanceof Map<?, ?> m)) {
                continue;
            }
            Object nameObj = m.get("name");
            if (!(nameObj instanceof String name) || name.isEmpty()) {
                continue;
            }
            long version = -1L;
            Object vObj = m.get("version");
            if (vObj instanceof Number n) {
                version = n.longValue();
            } else if (vObj instanceof String s) {
                try {
                    version = Long.parseLong(s);
                } catch (NumberFormatException ignored) {
                    version = -1L;
                }
            }
            String state = m.get("state") instanceof String st ? st : null;
            String reason = m.get("reason") instanceof String r ? r : null;

            TritonModel tritonModel = new TritonModel(this, name, version, state, reason);
            out.put(name, tritonModel);
            if (version >= 0) {
                out.put(name + "/" + version, tritonModel);
            }
        }
        return new TritonModelDictionary(out);
    }

    static Object messageToObject(TRITONSERVER_Message message) {
        BytePointer base = new BytePointer((Pointer) null);
        SizeTPointer byteSize = new SizeTPointer(1);
        NativeError.check(
                TRITONSERVER_MessageSerializeToJson(message, base, byteSize),
                "MessageSerializeToJson");
        String json = readCString(base, byteSize.get());
        if (json.isEmpty()) {
            return Map.of();
        }
        return jsonElementToJava(JsonParser.parseString(json));
    }

    private static String readCString(BytePointer base, long n) {
        if (base == null || base.isNull()) {
            return "";
        }
        if (n > 0 && n <= Integer.MAX_VALUE) {
            byte[] bytes = new byte[(int) n];
            base.get(bytes);
            // Triton may or may not include trailing NUL in byte_size.
            int len = bytes.length;
            if (len > 0 && bytes[len - 1] == 0) {
                len--;
            }
            return new String(bytes, 0, len);
        }
        return base.getString();
    }

    private static Object jsonElementToJava(JsonElement el) {
        if (el == null || el.isJsonNull()) {
            return null;
        }
        if (el.isJsonPrimitive()) {
            var p = el.getAsJsonPrimitive();
            if (p.isBoolean()) {
                return p.getAsBoolean();
            }
            if (p.isNumber()) {
                String s = p.getAsString();
                if (s.indexOf('.') < 0 && s.indexOf('e') < 0 && s.indexOf('E') < 0) {
                    try {
                        return p.getAsLong();
                    } catch (NumberFormatException ignored) {
                        // fall through
                    }
                }
                return p.getAsDouble();
            }
            return p.getAsString();
        }
        if (el.isJsonArray()) {
            JsonArray arr = el.getAsJsonArray();
            List<Object> list = new ArrayList<>(arr.size());
            for (JsonElement child : arr) {
                list.add(jsonElementToJava(child));
            }
            return list;
        }
        if (el.isJsonObject()) {
            JsonObject obj = el.getAsJsonObject();
            Map<String, Object> map = new LinkedHashMap<>();
            for (Map.Entry<String, JsonElement> e : obj.entrySet()) {
                map.put(e.getKey(), jsonElementToJava(e.getValue()));
            }
            return map;
        }
        return null;
    }

    @Override
    public String toString() {
        return "Server{started=" + started + ", serverId=" + tritonOptions.serverId() + "}";
    }
}
