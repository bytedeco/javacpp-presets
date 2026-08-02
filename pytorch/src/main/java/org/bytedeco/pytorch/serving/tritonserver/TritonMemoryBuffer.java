package org.bytedeco.pytorch.serving.tritonserver;

import org.bytedeco.pytorch.serving.tritonserver.enums.TritonMemoryType;
import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
/**
 * Memory allocated for a tensor.
 *
 * <p>Corresponds to Python {@code tritonserver.MemoryBuffer}. Does not own the
 * memory by itself; holds a reference to {@link #owner()} so the backing storage
 * is not reclaimed while the buffer is reachable.
 */
public final class TritonMemoryBuffer {
    private final long dataPtr;
    private final TritonMemoryType tritonMemoryType;
    private final long memoryTypeId;
    private final long size;
    private final Object owner;

    public TritonMemoryBuffer(
            long dataPtr,
            TritonMemoryType tritonMemoryType,
            long memoryTypeId,
            long size,
            Object owner) {
        if (tritonMemoryType == null) {
            throw new TritonInvalidArgumentException("memoryType must not be null");
        }
        if (size < 0) {
            throw new TritonInvalidArgumentException("size must be >= 0");
        }
        this.dataPtr = dataPtr;
        this.tritonMemoryType = tritonMemoryType;
        this.memoryTypeId = memoryTypeId;
        this.size = size;
        this.owner = owner;
    }

    public long dataPtr() {
        return dataPtr;
    }

    public TritonMemoryType memoryType() {
        return tritonMemoryType;
    }

    public long memoryTypeId() {
        return memoryTypeId;
    }

    public long size() {
        return size;
    }

    /**
     * Object that keeps the underlying storage reachable (e.g. {@code byte[]} or
     * a JavaCPP {@code Pointer}).
     */
    public Object owner() {
        return owner;
    }

    @Override
    public String toString() {
        return "MemoryBuffer{dataPtr=0x" + Long.toHexString(dataPtr)
                + ", memoryType=" + tritonMemoryType
                + ", memoryTypeId=" + memoryTypeId
                + ", size=" + size
                + "}";
    }
}
