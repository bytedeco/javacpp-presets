package org.bytedeco.pytorch.serving.tensorrt;

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.serving.tensorrt.enums.TRTDataType;
import org.bytedeco.pytorch.serving.tensorrt.exceptions.TrtInvalidArgumentException;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/**
 * Host-side tensor value for TensorRT inputs/outputs.
 *
 * <p>Analogous to Python usage of NumPy arrays bound to engine I/O names.
 * MVP stores CPU (host) buffers; {@link TRTEngine} copies to/from device as needed.
 *
 * <p>Shape uses {@code long[]} to match {@code nvinfer1::Dims64}.
 */
public final class TrtTensor {
    private final TRTDataType dataType;
    private final long[] shape;
    private final Pointer hostData;
    private final long byteSize;
    /** Keeps backing Java array / buffer reachable when hostData views it. */
    private final Object owner;

    public TrtTensor(TRTDataType dataType, long[] shape, Pointer hostData, long byteSize, Object owner) {
        if (dataType == null) {
            throw new TrtInvalidArgumentException("dataType must not be null");
        }
        if (shape == null) {
            throw new TrtInvalidArgumentException("shape must not be null");
        }
        if (hostData == null || hostData.isNull()) {
            if (byteSize != 0) {
                throw new TrtInvalidArgumentException("hostData is null but byteSize=" + byteSize);
            }
        }
        if (byteSize < 0) {
            throw new TrtInvalidArgumentException("byteSize must be >= 0");
        }
        this.dataType = dataType;
        this.shape = Arrays.copyOf(shape, shape.length);
        this.hostData = hostData == null ? new Pointer() : hostData;
        this.byteSize = byteSize;
        this.owner = owner;
        validateSize();
    }

    private void validateSize() {
        if (dataType.byteSize() <= 0) {
            return; // packed / special types — skip strict check
        }
        long elements = elementCount(shape);
        long expected = elements * (long) dataType.byteSize();
        if (byteSize != expected) {
            throw new TrtInvalidArgumentException(
                    "buffer size " + byteSize
                            + " does not match shape " + Arrays.toString(shape)
                            + " and type " + dataType + " (expected " + expected + " bytes)");
        }
    }

    public TRTDataType dataType() {
        return dataType;
    }

    public long[] shape() {
        return Arrays.copyOf(shape, shape.length);
    }

    public Pointer hostData() {
        return hostData;
    }

    public long byteSize() {
        return byteSize;
    }

    public Object owner() {
        return owner;
    }

    public long elementCount() {
        return elementCount(shape);
    }

    public static long elementCount(long[] shape) {
        if (shape == null) {
            throw new TrtInvalidArgumentException("shape must not be null");
        }
        long n = 1;
        for (long d : shape) {
            if (d < 0) {
                throw new TrtInvalidArgumentException("negative dimension in shape: " + Arrays.toString(shape));
            }
            if (d != 0 && n > Long.MAX_VALUE / d) {
                throw new TrtInvalidArgumentException("shape element count overflow: " + Arrays.toString(shape));
            }
            n *= d;
        }
        return n;
    }

    public static long volumeBytes(TRTDataType type, long[] shape) {
        Objects.requireNonNull(type, "type");
        if (type.byteSize() <= 0) {
            throw new TrtInvalidArgumentException("cannot compute volume for packed type " + type);
        }
        return elementCount(shape) * (long) type.byteSize();
    }

    // ---- factories ----

    public static TrtTensor of(TRTDataType type, long[] shape, byte[] data) {
        Objects.requireNonNull(data, "data");
        long expected = type.byteSize() > 0 ? volumeBytes(type, shape) : data.length;
        if (data.length != expected) {
            throw new TrtInvalidArgumentException(
                    "byte[] length " + data.length + " != expected " + expected);
        }
        BytePointer ptr = new BytePointer(data);
        return new TrtTensor(type, shape, ptr, data.length, data);
    }

    public static TrtTensor of(float[] data, long... shape) {
        Objects.requireNonNull(data, "data");
        long[] s = normalizeShape(shape, data.length);
        long expected = volumeBytes(TRTDataType.FLOAT, s);
        if (expected != (long) data.length * Float.BYTES) {
            throw new TrtInvalidArgumentException("float[] length does not match shape");
        }
        FloatPointer ptr = new FloatPointer(data);
        return new TrtTensor(TRTDataType.FLOAT, s, ptr, expected, data);
    }

    public static TrtTensor of(int[] data, long... shape) {
        Objects.requireNonNull(data, "data");
        long[] s = normalizeShape(shape, data.length);
        long expected = volumeBytes(TRTDataType.INT32, s);
        if (expected != (long) data.length * Integer.BYTES) {
            throw new TrtInvalidArgumentException("int[] length does not match shape");
        }
        IntPointer ptr = new IntPointer(data);
        return new TrtTensor(TRTDataType.INT32, s, ptr, expected, data);
    }

    public static TrtTensor of(long[] data, long... shape) {
        Objects.requireNonNull(data, "data");
        long[] s = normalizeShape(shape, data.length);
        long expected = volumeBytes(TRTDataType.INT64, s);
        if (expected != (long) data.length * Long.BYTES) {
            throw new TrtInvalidArgumentException("long[] length does not match shape");
        }
        LongPointer ptr = new LongPointer(data);
        return new TrtTensor(TRTDataType.INT64, s, ptr, expected, data);
    }

    public static TrtTensor of(ByteBuffer buffer, TRTDataType type, long... shape) {
        Objects.requireNonNull(buffer, "buffer");
        Objects.requireNonNull(type, "type");
        ByteBuffer dup = buffer.duplicate().order(ByteOrder.nativeOrder());
        long expected = type.byteSize() > 0 ? volumeBytes(type, shape) : dup.remaining();
        if (dup.remaining() < expected) {
            throw new TrtInvalidArgumentException(
                    "ByteBuffer remaining " + dup.remaining() + " < expected " + expected);
        }
        byte[] copy = new byte[(int) expected];
        dup.get(copy);
        return of(type, shape.length == 0 ? new long[] {copy.length} : shape, copy);
    }

    /** Empty host tensor with allocated zeroed storage. */
    public static TrtTensor zeros(TRTDataType type, long... shape) {
        long bytes = volumeBytes(type, shape);
        if (bytes > Integer.MAX_VALUE) {
            throw new TrtInvalidArgumentException("tensor too large: " + bytes);
        }
        byte[] data = new byte[(int) bytes];
        return of(type, shape, data);
    }

    private static long[] normalizeShape(long[] shape, int elementLength) {
        if (shape == null || shape.length == 0) {
            return new long[] {elementLength};
        }
        return Arrays.copyOf(shape, shape.length);
    }

    // ---- host readbacks ----

    public byte[] toByteArray() {
        byte[] out = new byte[Math.toIntExact(byteSize)];
        if (byteSize == 0) {
            return out;
        }
        BytePointer bp = new BytePointer(hostData);
        bp.limit(byteSize).position(0);
        bp.get(out);
        return out;
    }

    public float[] toFloatArray() {
        requireType(TRTDataType.FLOAT);
        int n = Math.toIntExact(elementCount());
        float[] out = new float[n];
        FloatPointer fp = new FloatPointer(hostData);
        fp.limit(n).position(0);
        fp.get(out);
        return out;
    }

    public int[] toIntArray() {
        requireType(TRTDataType.INT32);
        int n = Math.toIntExact(elementCount());
        int[] out = new int[n];
        IntPointer ip = new IntPointer(hostData);
        ip.limit(n).position(0);
        ip.get(out);
        return out;
    }

    public long[] toLongArray() {
        requireType(TRTDataType.INT64);
        int n = Math.toIntExact(elementCount());
        long[] out = new long[n];
        LongPointer lp = new LongPointer(hostData);
        lp.limit(n).position(0);
        lp.get(out);
        return out;
    }

    public short[] toHalfBitsArray() {
        requireType(TRTDataType.HALF);
        int n = Math.toIntExact(elementCount());
        short[] out = new short[n];
        ShortPointer sp = new ShortPointer(hostData);
        sp.limit(n).position(0);
        sp.get(out);
        return out;
    }

    private void requireType(TRTDataType expected) {
        if (dataType != expected) {
            throw new TrtInvalidArgumentException(
                    "tensor type is " + dataType + ", expected " + expected);
        }
    }

    @Override
    public String toString() {
        return "Tensor{type=" + dataType + ", shape=" + Arrays.toString(shape)
                + ", byteSize=" + byteSize + '}';
    }
}
