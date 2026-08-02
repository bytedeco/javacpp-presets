package org.bytedeco.pytorch.serving.tritonserver.internal;

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonMemoryType;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonDataType;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequest;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceRequestReleaseFn_t;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceResponse;
import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_InferenceResponseCompleteFn_t;
import org.bytedeco.pytorch.serving.tritonserver.*;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.bytedeco.tritonserver.global.tritonserver.*;

/**
 * Request/response complete callbacks + per-infer state.
 *
 * <p>Aligns with Python ResponseIterator / bytedeco Simple.java:
 * response complete callback owns the native response, parses outputs into
 * {@link TritonInferenceResponse}, and enqueues them for the Java iterator.
 * Request release callback deletes the native request once Triton is done.
 *
 * <p>FunctionPointers are static so GC cannot free them while native code holds them.
 * Per-request state is looked up via a synthetic userp id (pointer address encoding).
 */
public final class InferCallbacks {
    private static final TRITONSERVER_InferenceRequestReleaseFn_t REQUEST_RELEASE_FN =
            new RequestReleaseFn();
    private static final TRITONSERVER_InferenceResponseCompleteFn_t RESPONSE_COMPLETE_FN =
            new ResponseCompleteFn();

    private static final ConcurrentHashMap<Long, InferSink> SINKS = new ConcurrentHashMap<>();
    private static final AtomicLong NEXT_SINK_ID = new AtomicLong(1);

    private InferCallbacks() {}

    public static TRITONSERVER_InferenceRequestReleaseFn_t requestReleaseFn() {
        return REQUEST_RELEASE_FN;
    }

    public static TRITONSERVER_InferenceResponseCompleteFn_t responseCompleteFn() {
        return RESPONSE_COMPLETE_FN;
    }

    /**
     * Register a sink and return a Pointer whose address is the sink id.
     * Caller must pass the same Pointer as response_userp / request_release_userp
     * and keep it reachable until FINAL + request release.
     */
    public static Pointer retain(InferSink sink) {
        long id = NEXT_SINK_ID.getAndIncrement();
        sink.id = id;
        SINKS.put(id, sink);
        // Encode id as pointer address (no real allocation). JavaCPP has no Pointer(long) ctor.
        return new Pointer() {
            {
                address = id;
            }
        };
    }

    public static InferSink lookup(Pointer userp) {
        if (userp == null || userp.isNull()) {
            return null;
        }
        return SINKS.get(userp.address());
    }

    public static void release(long id) {
        SINKS.remove(id);
    }

    /**
     * Per-inference state shared between request release and response complete.
     */
    public static final class InferSink {
        volatile long id;
        final TritonModel tritonModel;
        final BlockingQueue<TritonInferenceResponse> queue;
        final AtomicBoolean finished = new AtomicBoolean(false);
        final AtomicBoolean requestReleased = new AtomicBoolean(false);
        /** Keep input tensors reachable until request is released. */
        public volatile Object inputKeepAlive;
        /** userp Pointer object itself must stay reachable. */
        public volatile Pointer userpHandle;
        public volatile Throwable callbackError;

        public InferSink(TritonModel tritonModel) {
            this.tritonModel = Objects.requireNonNull(tritonModel, "model");
            this.queue = new LinkedBlockingQueue<>();
        }

        public BlockingQueue<TritonInferenceResponse> queue() {
            return queue;
        }

        public boolean isFinished() {
            return finished.get();
        }

        void markFinished() {
            finished.set(true);
            // Sentinel so iterators blocked on take() wake up.
            queue.offer(TritonInferenceResponse.sentinel());
        }

        void tryCleanup() {
            if (finished.get() && requestReleased.get()) {
                inputKeepAlive = null;
                userpHandle = null;
                release(id);
            }
        }
    }

    private static final class RequestReleaseFn extends TRITONSERVER_InferenceRequestReleaseFn_t {
        @Override
        public void call(TRITONSERVER_InferenceRequest request, int flags, Pointer userp) {
            try {
                if ((flags & TRITONSERVER_REQUEST_RELEASE_ALL) != 0) {
                    if (request != null && !request.isNull()) {
                        TRITONSERVER_InferenceRequestDelete(request);
                    }
                }
            } catch (Throwable t) {
                InferSink sink = lookup(userp);
                if (sink != null) {
                    sink.callbackError = t;
                }
            } finally {
                InferSink sink = lookup(userp);
                if (sink != null) {
                    sink.requestReleased.set(true);
                    // Inputs may now be freed from Triton's perspective.
                    sink.inputKeepAlive = null;
                    sink.tryCleanup();
                }
            }
        }
    }

    private static final class ResponseCompleteFn extends TRITONSERVER_InferenceResponseCompleteFn_t {
        @Override
        public void call(TRITONSERVER_InferenceResponse response, int flags, Pointer userp) {
            InferSink sink = lookup(userp);
            boolean isFinal = (flags & TRITONSERVER_RESPONSE_COMPLETE_FINAL) != 0;
            try {
                if (response != null && !response.isNull()) {
                    TritonInferenceResponse parsed = parseResponse(sink != null ? sink.tritonModel : null, response, isFinal);
                    if (sink != null) {
                        sink.queue.offer(parsed);
                    } else {
                        // Orphan response — still must delete.
                        safeDeleteResponse(response);
                    }
                } else if (isFinal && sink != null) {
                    // Final with null response: no more payloads.
                }
            } catch (Throwable t) {
                if (sink != null) {
                    sink.callbackError = t;
                    sink.queue.offer(TritonInferenceResponse.errorOnly(
                            sink.tritonModel,
                            t instanceof TritonException te
                                    ? te
                                    : new TritonInternalException(
                                            "response callback failed: " + t.getMessage(), t)));
                    // Native response may still need delete if parse failed mid-way.
                    safeDeleteResponse(response);
                } else {
                    safeDeleteResponse(response);
                }
            } finally {
                if (isFinal && sink != null) {
                    sink.markFinished();
                    sink.tryCleanup();
                }
            }
        }
    }

    private static void safeDeleteResponse(TRITONSERVER_InferenceResponse response) {
        if (response == null || response.isNull()) {
            return;
        }
        try {
            TRITONSERVER_InferenceResponseDelete(response);
        } catch (Throwable ignored) {
            // best-effort
        }
    }

    /**
     * Parse native response into Java object and take ownership of output buffers.
     * Deletes the native response afterwards (release callbacks free allocator bookkeeping).
     */
    static TritonInferenceResponse parseResponse(
            TritonModel tritonModel, TRITONSERVER_InferenceResponse response, boolean isFinal) {
        TritonException error = NativeError.fromResponseError(TRITONSERVER_InferenceResponseError(response));

        String requestId = "";
        try {
            BytePointer idPtr = new BytePointer((Pointer) null);
            NativeError.check(TRITONSERVER_InferenceResponseId(response, idPtr), "response id");
            if (idPtr != null && !idPtr.isNull()) {
                requestId = idPtr.getString();
            }
        } catch (Throwable ignored) {
            // id is optional
        }

        Map<String, Object> parameters = readParameters(response);
        Map<String, TritonTensor> outputs = new LinkedHashMap<>();

        if (error == null) {
            IntPointer countPtr = new IntPointer(1);
            NativeError.check(
                    TRITONSERVER_InferenceResponseOutputCount(response, countPtr),
                    "response output count");
            int count = countPtr.get();
            for (int i = 0; i < count; i++) {
                BytePointer namePtr = new BytePointer((Pointer) null);
                IntPointer datatype = new IntPointer(1);
                LongPointer shapePtr = new LongPointer((Pointer) null);
                LongPointer dimCount = new LongPointer(1);
                Pointer base = new Pointer();
                SizeTPointer byteSize = new SizeTPointer(1);
                IntPointer memoryType = new IntPointer(1);
                LongPointer memoryTypeId = new LongPointer(1);
                Pointer userp = new Pointer();

                NativeError.check(
                        TRITONSERVER_InferenceResponseOutput(
                                response,
                                i,
                                namePtr,
                                datatype,
                                shapePtr,
                                dimCount,
                                base,
                                byteSize,
                                memoryType,
                                memoryTypeId,
                                userp),
                        "response output " + i);

                String name = namePtr == null || namePtr.isNull() ? ("output_" + i) : namePtr.getString();
                TritonDataType dt = TritonDataType.fromCode(datatype.get());
                long dims = dimCount.get();
                long[] shape = new long[(int) dims];
                if (dims > 0 && shapePtr != null && !shapePtr.isNull()) {
                    LongPointer sp = new LongPointer(shapePtr);
                    for (int d = 0; d < dims; d++) {
                        shape[d] = sp.get(d);
                    }
                }
                long size = byteSize.get();
                TritonMemoryType mt = TritonMemoryType.fromCode(memoryType.get());
                long mtId = memoryTypeId.get();

                // Take ownership of allocator buffer so Tensor keeps data after response delete.
                Object owner;
                long address = (base == null || base.isNull()) ? 0L : base.address();
                if (userp != null && !userp.isNull()) {
                    long uid = userp.address();
                    BytePointer owned = TRTResponseAllocators.takeOwnership(uid);
                    if (owned != null) {
                        owner = owned;
                        address = owned.address();
                    } else {
                        // Copy to stable CPU buffer if we cannot claim ownership.
                        owner = copyToOwned(base, size, mt);
                        if (owner instanceof BytePointer bp) {
                            address = bp.address();
                        }
                        mt = TritonMemoryType.CPU;
                        mtId = 0;
                    }
                } else if (size > 0 && address != 0L) {
                    owner = copyToOwned(base, size, mt);
                    if (owner instanceof BytePointer bp) {
                        address = bp.address();
                    }
                    mt = TritonMemoryType.CPU;
                    mtId = 0;
                } else {
                    owner = null;
                }

                TritonMemoryBuffer buf = new TritonMemoryBuffer(address, mt, mtId, size, owner);
                outputs.put(name, new TritonTensor(dt, shape, buf));
            }
        }

        // Deleting response triggers ReleaseFn for any buffers still tracked by allocator.
        // Buffers we tookOwnership of are no longer in the map, so release is a no-op for them.
        TRITONSERVER_InferenceResponseDelete(response);

        return new TritonInferenceResponse(tritonModel, requestId, parameters, outputs, error, isFinal, false);
    }

    private static Object copyToOwned(Pointer base, long size, TritonMemoryType mt) {
        if (size <= 0 || base == null || base.isNull()) {
            return null;
        }
        if (mt != TritonMemoryType.CPU && mt != TritonMemoryType.CPU_PINNED) {
            // GPU copy not in MVP — leave empty owner; Tensor.to* will fail with Unsupported.
            return null;
        }
        BytePointer dst = new BytePointer(size);
        BytePointer src = new BytePointer(base);
        src.limit(size).capacity(size);
        dst.put(src.limit(size));
        dst.limit(size).capacity(size);
        return dst;
    }

    private static Map<String, Object> readParameters(TRITONSERVER_InferenceResponse response) {
        Map<String, Object> out = new LinkedHashMap<>();
        try {
            IntPointer countPtr = new IntPointer(1);
            NativeError.check(
                    TRITONSERVER_InferenceResponseParameterCount(response, countPtr),
                    "response parameter count");
            int count = countPtr.get();
            for (int i = 0; i < count; i++) {
                BytePointer namePtr = new BytePointer((Pointer) null);
                IntPointer typePtr = new IntPointer(1);
                Pointer valuePtr = new Pointer();
                NativeError.check(
                        TRITONSERVER_InferenceResponseParameter(response, i, namePtr, typePtr, valuePtr),
                        "response parameter " + i);
                String name = namePtr == null || namePtr.isNull() ? ("param_" + i) : namePtr.getString();
                int type = typePtr.get();
                Object value = switch (type) {
                    case TRITONSERVER_PARAMETER_STRING -> {
                        BytePointer bp = new BytePointer(valuePtr);
                        yield bp.isNull() ? null : bp.getString();
                    }
                    case TRITONSERVER_PARAMETER_INT -> new LongPointer(valuePtr).get();
                    case TRITONSERVER_PARAMETER_BOOL -> new org.bytedeco.javacpp.BoolPointer(valuePtr).get();
                    case TRITONSERVER_PARAMETER_DOUBLE -> new org.bytedeco.javacpp.DoublePointer(valuePtr).get();
                    default -> null;
                };
                out.put(name, value);
            }
        } catch (Throwable ignored) {
            // parameters are best-effort in MVP
        }
        return out;
    }
}
