package org.bytedeco.pytorch.serving.tritonserver;

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonMemoryType;
import org.bytedeco.pytorch.serving.tritonserver.enums.TritonDataType;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

import org.bytedeco.pytorch.serving.tritonserver.exceptions.*;
/**
 * Tensor value for inference inputs/outputs.
 *
 * <p>Corresponds to Python {@code tritonserver.Tensor}. MVP focuses on CPU
 * buffers backed by Java arrays or JavaCPP pointers.
 */
public final class TritonTensor {
    private final TritonDataType tritonDataType;
    private final long[] shape;
    private final TritonMemoryBuffer tritonMemoryBuffer;

    public TritonTensor(TritonDataType tritonDataType, long[] shape, TritonMemoryBuffer tritonMemoryBuffer) {
        if (tritonDataType == null) {
            throw new TritonInvalidArgumentException("dataType must not be null");
        }
        if (shape == null) {
            throw new TritonInvalidArgumentException("shape must not be null");
        }
        if (tritonMemoryBuffer == null) {
            throw new TritonInvalidArgumentException("memoryBuffer must not be null");
        }
        this.tritonDataType = tritonDataType;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.tritonMemoryBuffer = tritonMemoryBuffer;
        validateSize();
    }

    private void validateSize() {
        if (tritonDataType == TritonDataType.BYTES || tritonDataType == TritonDataType.INVALID) {
            return;
        }
        long elements = elementCount(shape);
        long expected = elements * tritonDataType.byteSize();
        if (tritonMemoryBuffer.size() != expected) {
            throw new TritonInvalidArgumentException(
                    "buffer size " + tritonMemoryBuffer.size()
                            + " does not match shape " + Arrays.toString(shape)
                            + " and type " + tritonDataType + " (expected " + expected + " bytes)");
        }
    }

    public TritonDataType dataType() {
        return tritonDataType;
    }

    public long[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public TritonMemoryBuffer memoryBuffer() {
        return tritonMemoryBuffer;
    }

    public long dataPtr() {
        return tritonMemoryBuffer.dataPtr();
    }

    public TritonMemoryType memoryType() {
        return tritonMemoryBuffer.memoryType();
    }

    public long memoryTypeId() {
        return tritonMemoryBuffer.memoryTypeId();
    }

    public long size() {
        return tritonMemoryBuffer.size();
    }

    public static long elementCount(long[] shape) {
        long n = 1;
        for (long d : shape) {
            if (d < 0) {
                throw new TritonInvalidArgumentException("shape dimension must be >= 0, got " + d);
            }
            n = Math.multiplyExact(n, d);
        }
        return n;
    }

    // ---- factories from primitive arrays (CPU) ----

    public static TritonTensor of(boolean[] data, long... shape) {
        long[] resolved = resolveShape(shape, data.length);
        byte[] bytes = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            bytes[i] = (byte) (data[i] ? 1 : 0);
        }
        return fromHeapBytes(TritonDataType.BOOL, resolved, bytes);
    }

    public static TritonTensor of(byte[] data, long... shape) {
        long[] resolved = resolveShape(shape, data.length);
        return fromHeapBytes(TritonDataType.INT8, resolved, Arrays.copyOf(data, data.length));
    }

    public static TritonTensor of(short[] data, long... shape) {
        long[] resolved = resolveShape(shape, data.length);
        ByteBuffer bb = ByteBuffer.allocate(data.length * Short.BYTES).order(ByteOrder.nativeOrder());
        for (short v : data) {
            bb.putShort(v);
        }
        return fromHeapBytes(TritonDataType.INT16, resolved, bb.array());
    }

    public static TritonTensor of(int[] data, long... shape) {
        long[] resolved = resolveShape(shape, data.length);
        ByteBuffer bb = ByteBuffer.allocate(data.length * Integer.BYTES).order(ByteOrder.nativeOrder());
        for (int v : data) {
            bb.putInt(v);
        }
        return fromHeapBytes(TritonDataType.INT32, resolved, bb.array());
    }

    public static TritonTensor of(long[] data, long... shape) {
        long[] resolved = resolveShape(shape, data.length);
        ByteBuffer bb = ByteBuffer.allocate(data.length * Long.BYTES).order(ByteOrder.nativeOrder());
        for (long v : data) {
            bb.putLong(v);
        }
        return fromHeapBytes(TritonDataType.INT64, resolved, bb.array());
    }

    public static TritonTensor of(float[] data, long... shape) {
        long[] resolved = resolveShape(shape, data.length);
        ByteBuffer bb = ByteBuffer.allocate(data.length * Float.BYTES).order(ByteOrder.nativeOrder());
        for (float v : data) {
            bb.putFloat(v);
        }
        return fromHeapBytes(TritonDataType.FP32, resolved, bb.array());
    }

    public static TritonTensor of(double[] data, long... shape) {
        long[] resolved = resolveShape(shape, data.length);
        ByteBuffer bb = ByteBuffer.allocate(data.length * Double.BYTES).order(ByteOrder.nativeOrder());
        for (double v : data) {
            bb.putDouble(v);
        }
        return fromHeapBytes(TritonDataType.FP64, resolved, bb.array());
    }

    /**
     * Generic factory: raw bytes with explicit datatype and shape.
     */
    public static TritonTensor of(TritonDataType tritonDataType, long[] shape, byte[] data) {
        Objects.requireNonNull(tritonDataType, "dataType");
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(data, "data");
        return fromHeapBytes(tritonDataType, Arrays.copyOf(shape, shape.length), Arrays.copyOf(data, data.length));
    }

    /**
     * Wrap an existing native pointer without copying.
     *
     * <p>{@code owner} must keep the storage alive (typically the Pointer itself).
     */
    public static TritonTensor wrap(TritonDataType tritonDataType, long[] shape, Pointer pointer, long byteSize, TritonMemoryType tritonMemoryType, long memoryTypeId) {
        if (pointer == null || pointer.isNull()) {
            throw new TritonInvalidArgumentException("pointer must not be null");
        }
        TritonMemoryBuffer buf = new TritonMemoryBuffer(pointer.address(), tritonMemoryType, memoryTypeId, byteSize, pointer);
        return new TritonTensor(tritonDataType, shape, buf);
    }

    private static TritonTensor fromHeapBytes(TritonDataType tritonDataType, long[] shape, byte[] data) {
        // Allocate off-heap so Triton C API can read a stable address.
        BytePointer ptr = new BytePointer(data.length);
        ptr.put(data);
        TritonMemoryBuffer buf = new TritonMemoryBuffer(ptr.address(), TritonMemoryType.CPU, 0, data.length, ptr);
        return new TritonTensor(tritonDataType, shape, buf);
    }

    private static long[] resolveShape(long[] shape, int elementCount) {
        if (shape == null || shape.length == 0) {
            return new long[] {elementCount};
        }
        long n = elementCount(shape);
        if (n != elementCount) {
            throw new TritonInvalidArgumentException(
                    "shape " + Arrays.toString(shape) + " has " + n
                            + " elements but data has " + elementCount);
        }
        return Arrays.copyOf(shape, shape.length);
    }

    // ---- host reads ----

    public void requireCpu() {
        if (memoryType() != TritonMemoryType.CPU && memoryType() != TritonMemoryType.CPU_PINNED) {
            throw new UnsupportedException(
                    "tensor is in " + memoryType() + "; copy to host is not implemented in MVP");
        }
    }

    public byte[] toByteArray() {
        requireCpu();
        byte[] out = new byte[Math.toIntExact(size())];
        BytePointer bp = new BytePointer(asPointer());
        bp.limit(size()).get(out);
        return out;
    }

    public int[] toIntArray() {
        requireCpu();
        requireType(TritonDataType.INT32);
        int n = Math.toIntExact(elementCount(shape));
        int[] out = new int[n];
        new IntPointer(asPointer()).get(out);
        return out;
    }

    public long[] toLongArray() {
        requireCpu();
        requireType(TritonDataType.INT64);
        int n = Math.toIntExact(elementCount(shape));
        long[] out = new long[n];
        new LongPointer(asPointer()).get(out);
        return out;
    }

    public float[] toFloatArray() {
        requireCpu();
        requireType(TritonDataType.FP32);
        int n = Math.toIntExact(elementCount(shape));
        float[] out = new float[n];
        new FloatPointer(asPointer()).get(out);
        return out;
    }

    public double[] toDoubleArray() {
        requireCpu();
        requireType(TritonDataType.FP64);
        int n = Math.toIntExact(elementCount(shape));
        double[] out = new double[n];
        new DoublePointer(asPointer()).get(out);
        return out;
    }

    public short[] toShortArray() {
        requireCpu();
        requireType(TritonDataType.INT16);
        int n = Math.toIntExact(elementCount(shape));
        short[] out = new short[n];
        new ShortPointer(asPointer()).get(out);
        return out;
    }

    public boolean[] toBooleanArray() {
        requireCpu();
        requireType(TritonDataType.BOOL);
        byte[] raw = toByteArray();
        boolean[] out = new boolean[raw.length];
        for (int i = 0; i < raw.length; i++) {
            out[i] = raw[i] != 0;
        }
        return out;
    }

    private void requireType(TritonDataType expected) {
        if (tritonDataType != expected) {
            throw new TritonInvalidArgumentException(
                    "tensor datatype is " + tritonDataType + ", expected " + expected);
        }
    }

    private Pointer asPointer() {
        Object owner = tritonMemoryBuffer.owner();
        if (owner instanceof Pointer p && !p.isNull()) {
            return p;
        }
        return new Pointer() {
            {
                address = dataPtr();
                limit = size();
                capacity = size();
            }
        };
    }

    /**
     * Convert loosely typed input objects used in {@link TritonInferenceRequest#inputs()}.
     */
    static TritonTensor fromObject(Object value) {
        if (value == null) {
            throw new TritonInvalidArgumentException("input value must not be null");
        }
        if (value instanceof TritonTensor t) {
            return t;
        }
        if (value instanceof int[] a) {
            return of(a);
        }
        if (value instanceof long[] a) {
            return of(a);
        }
        if (value instanceof float[] a) {
            return of(a);
        }
        if (value instanceof double[] a) {
            return of(a);
        }
        if (value instanceof short[] a) {
            return of(a);
        }
        if (value instanceof byte[] a) {
            return of(a);
        }
        if (value instanceof boolean[] a) {
            return of(a);
        }
        throw new TritonInvalidArgumentException(
                "unsupported input type: " + value.getClass().getName()
                        + "; use Tensor or a primitive array");
    }

    @Override
    public String toString() {
        return "Tensor{dataType=" + tritonDataType
                + ", shape=" + Arrays.toString(shape)
                + ", memoryBuffer=" + tritonMemoryBuffer
                + "}";
    }
}
