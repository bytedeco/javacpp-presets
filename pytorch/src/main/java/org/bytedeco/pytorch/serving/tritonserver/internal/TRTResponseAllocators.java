package org.bytedeco.pytorch.serving.tritonserver.internal;

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonMemoryType;
import org.bytedeco.tritonserver.tritonserver.*;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.TritonInternalException;
import org.bytedeco.pytorch.serving.tritonserver.TritonModel;

import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import static org.bytedeco.tritonserver.global.tritonserver.*;

/**
 * Default CPU response allocator used by {@link TritonModel#infer}.
 *
 * <p>Mirrors bytedeco {@code Simple.java} / Python default allocator behaviour:
 * allocate host memory with {@link Pointer#malloc}, free on release. Preferred
 * GPU/PINNED types fall back to CPU for the MVP path.
 *
 * <p>Native callbacks are held as static FunctionPointer instances so the GC
 * cannot collect them while Triton still invokes them.
 */
public final class TRTResponseAllocators {
    /** Shared default allocator (CPU only). Created lazily, never deleted for process lifetime. */
    private static volatile TRITONSERVER_ResponseAllocator shared;

    private static final TRITONSERVER_ResponseAllocatorAllocFn_t ALLOC_FN = new AllocFn();
    private static final TRITONSERVER_ResponseAllocatorReleaseFn_t RELEASE_FN = new ReleaseFn();
    /** Optional; not required for one-to-one models. */
    private static final TRITONSERVER_ResponseAllocatorStartFn_t START_FN = null;

    /**
     * Maps buffer_userp id → owning BytePointer so release can free correctly
     * and GC keeps the pointer alive between alloc and release.
     */
    private static final ConcurrentHashMap<Long, BytePointer> LIVE_BUFFERS = new ConcurrentHashMap<>();
    private static final AtomicLong NEXT_BUFFER_ID = new AtomicLong(1);

    private TRTResponseAllocators() {}

    /**
     * Process-wide shared CPU allocator. Safe to reuse across requests.
     *
     * <p>Must only be called after Triton native libraries are loadable.
     */
    public static TRITONSERVER_ResponseAllocator sharedCpu() {
        TRITONSERVER_ResponseAllocator local = shared;
        if (local != null) {
            return local;
        }
        synchronized (TRTResponseAllocators.class) {
            if (shared == null) {
                TRITONSERVER_ResponseAllocator allocator =
                        new TRITONSERVER_ResponseAllocator((Pointer) null);
                NativeError.check(
                        TRITONSERVER_ResponseAllocatorNew(allocator, ALLOC_FN, RELEASE_FN, START_FN),
                        "creating response allocator");
                shared = allocator;
            }
            return shared;
        }
    }

    /**
     * Allocate a CPU buffer and register it under a synthetic buffer_userp id.
     * Used by unit tests / callers that need the same layout without going through Triton.
     */
    public static OwnedBuffer allocateCpu(long byteSize) {
        if (byteSize < 0) {
            throw new TritonInternalException("byteSize must be >= 0");
        }
        if (byteSize == 0) {
            return new OwnedBuffer(0L, 0L, null);
        }
        BytePointer ptr = new BytePointer(byteSize);
        ptr.limit(byteSize).capacity(byteSize);
        long id = NEXT_BUFFER_ID.getAndIncrement();
        LIVE_BUFFERS.put(id, ptr);
        return new OwnedBuffer(ptr.address(), id, ptr);
    }

    /** Free a buffer previously returned by {@link #allocateCpu}. */
    public static void freeOwned(OwnedBuffer buf) {
        if (buf == null || buf.userpId == 0L) {
            return;
        }
        BytePointer ptr = LIVE_BUFFERS.remove(buf.userpId);
        if (ptr != null) {
            ptr.deallocate();
        }
    }

    public static final class OwnedBuffer {
        public final long address;
        public final long userpId;
        public final BytePointer pointer;

        OwnedBuffer(long address, long userpId, BytePointer pointer) {
            this.address = address;
            this.userpId = userpId;
            this.pointer = pointer;
        }
    }

    // ---- native callbacks ----

    private static final class AllocFn extends TRITONSERVER_ResponseAllocatorAllocFn_t {
        @Override
        public TRITONSERVER_Error call(
                TRITONSERVER_ResponseAllocator allocator,
                String tensorName,
                long byteSize,
                int preferredMemoryType,
                long preferredMemoryTypeId,
                Pointer userp,
                PointerPointer buffer,
                PointerPointer bufferUserp,
                IntPointer actualMemoryType,
                LongPointer actualMemoryTypeId) {
            try {
                // MVP: always allocate CPU; ignore preferred GPU/PINNED.
                actualMemoryType.put(0, TRITONSERVER_MEMORY_CPU);
                actualMemoryTypeId.put(0, 0L);

                if (byteSize == 0) {
                    buffer.put(0, (Pointer) null);
                    bufferUserp.put(0, (Pointer) null);
                    return null;
                }

                BytePointer ptr = new BytePointer(byteSize);
                ptr.limit(byteSize).capacity(byteSize);
                long id = NEXT_BUFFER_ID.getAndIncrement();
                LIVE_BUFFERS.put(id, ptr);

                buffer.put(0, ptr);
                // Encode id as pointer value so release can look it up without a real object.
                // JavaCPP has no Pointer(long) constructor; set address in an anonymous subclass.
                bufferUserp.put(0, new Pointer() {
                    {
                        address = id;
                    }
                });
                return null;
            } catch (Throwable t) {
                return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL,
                        "alloc failed for " + tensorName + ": " + t.getMessage());
            }
        }
    }

    private static final class ReleaseFn extends TRITONSERVER_ResponseAllocatorReleaseFn_t {
        @Override
        public TRITONSERVER_Error call(
                TRITONSERVER_ResponseAllocator allocator,
                Pointer buffer,
                Pointer bufferUserp,
                long byteSize,
                int memoryType,
                long memoryTypeId) {
            try {
                if (bufferUserp != null && !bufferUserp.isNull()) {
                    long id = bufferUserp.address();
                    BytePointer ptr = LIVE_BUFFERS.remove(id);
                    if (ptr != null) {
                        ptr.deallocate();
                    } else if (buffer != null && !buffer.isNull()
                            && memoryType == TRITONSERVER_MEMORY_CPU) {
                        // Fallback: free raw address if map entry already gone.
                        Pointer.free(buffer);
                    }
                } else if (buffer != null && !buffer.isNull()
                        && memoryType == TRITONSERVER_MEMORY_CPU) {
                    Pointer.free(buffer);
                }
                return null;
            } catch (Throwable t) {
                return TRITONSERVER_ErrorNew(
                        TRITONSERVER_ERROR_INTERNAL, "release failed: " + t.getMessage());
            }
        }
    }

    /** Prefer memory type string for logging. */
    public static String memoryTypeName(int code) {
        return TritonMemoryType.fromCode(code).typeString();
    }

    /**
     * Look up a live buffer by buffer_userp id (used when parsing response outputs
     * so the Tensor can keep the owner reachable until the response is closed).
     */
    public static BytePointer lookup(long userpId) {
        return LIVE_BUFFERS.get(userpId);
    }

    /**
     * Transfer ownership of a live buffer out of the allocator map into a
     * response-owned Tensor. After transfer, release callback is a no-op for this id
     * (response close / GC owns the BytePointer).
     */
    public static BytePointer takeOwnership(long userpId) {
        return LIVE_BUFFERS.remove(userpId);
    }

    /** Test/debug: number of buffers still tracked. */
    public static int liveBufferCount() {
        return LIVE_BUFFERS.size();
    }

    /**
     * Explicitly destroy the shared allocator (normally never needed).
     * Not thread-safe vs concurrent infer.
     */
    public static synchronized void shutdownShared() {
        if (shared != null) {
            TRITONSERVER_ResponseAllocatorDelete(shared);
            shared = null;
        }
        // Do not clear LIVE_BUFFERS blindly — in-flight responses may still hold ids.
    }

    static {
        // Touch FunctionPointers so class init fails fast if Loader cannot resolve symbols
        // only when first used via sharedCpu(); keep static init free of Loader.load.
        Objects.requireNonNull(ALLOC_FN);
        Objects.requireNonNull(RELEASE_FN);
    }
}
