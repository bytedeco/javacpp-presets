package org.bytedeco.pytorch.dataframe.dtype;
import org.bytedeco.pytorch.dataframe.enums.TensorDType;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;

import java.io.EOFException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * 张量数据容器（用于AI模型输入输出）
 * 支持多种数据类型，延迟转换以节省内存。
 */
public class TensorData extends AbstractDataValue implements StructuredData {

    // 数据类型枚举
//    public enum DType {
//        F64("F64", 8),
//        F32("F32", 4),
//        F16("F16", 2),
//        BF16("BF16", 2),
//        I64("I64", 8),
//        I32("I32", 4),
//        I16("I16", 2),
//        I8("I8", 1),
//        U8("U8", 1),
//        BOOL("BOOL", 1);
//
//        private final String name;
//        private final int bytesPerElement;
//
//        DType(String name, int bytesPerElement) {
//            this.name = name;
//            this.bytesPerElement = bytesPerElement;
//        }
//
//        public String getName() { return name; }
//        public int size() { return bytesPerElement; }
//
//        public static DType fromString(String s) {
//            if (s == null) return F32;
//            return switch (s.toUpperCase()) {
//                case "F64", "FLOAT64" -> F64;
//                case "F32", "FLOAT32" -> F32;
//                case "F16", "FLOAT16" -> F16;
//                case "BF16", "BFLOAT16" -> BF16;
//                case "I64", "INT64" -> I64;
//                case "I32", "INT32" -> I32;
//                case "I16", "INT16" -> I16;
//                case "I8", "INT8" -> I8;
//                case "U8", "UINT8" -> U8;
//                case "BOOL", "BOOLEAN" -> BOOL;
//                default -> F32;
//            };
//        }
//    }

    // 数据存储：可以是 float[] 或 ByteBuffer（延迟转换）
    private float[] floatData;
    private ByteBuffer rawData;
    private int[] shape;
    private TensorDType dtype;

    // 元数据（用于懒加载场景）
    private String filePath;
    private long dataOffset;
    private long dataLength;
    private boolean isLazy = false;

    // Native / zero-copy payloads
    private Tensor nativeTensor;
    private BytePointer nativePointer;
    private boolean ownsNativePointer;
    private MappedByteBuffer mappedData;

    private static final long LARGE_TENSOR_THRESHOLD_BYTES = 64L * 1024 * 1024; // mmap/zero-copy threshold

    // ==================== 构造函数 ====================

    /**
     * 从 float[] 构造（常规用法）
     */
    public TensorData(float[] data, int[] shape) {
        this(data, shape, TensorDType.F32);
    }

    /**
     * From float storage with explicit logical dtype (e.g. F16 metadata on F32 buffer).
     */
    public TensorData(float[] data, int[] shape, TensorDType dtype) {
        this.floatData = data;
        this.shape = shape == null ? new int[]{data == null ? 0 : data.length} : shape;
        this.dtype = dtype == null ? TensorDType.F32 : dtype;
    }

    /**
     * 从 double[] 构造
     */
    public TensorData(double[] data, int[] shape) {
        this.floatData = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            this.floatData[i] = (float) data[i];
        }
        this.shape = shape;
        this.dtype = TensorDType.F64;
    }

    /**
     * 只指定形状，数据初始化为零
     */
    public TensorData(int[] shape) {
        int size = 1;
        for (int dim : shape) size *= dim;
        this.floatData = new float[size];
        this.shape = shape;
        this.dtype = TensorDType.F32;
    }

    /**
     * 从原始 ByteBuffer 构造（支持多种数据类型，延迟转换）
     */
    public TensorData(ByteBuffer rawData, int[] shape, TensorDType dtype) {
        this.rawData = rawData.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        this.shape = shape;
        this.dtype = dtype;
        this.floatData = null; // 延迟转换
    }

    /**
     * 懒加载构造（只存储元数据，按需从文件加载）
     */
    public TensorData(String filePath, long dataOffset, long dataLength, int[] shape, TensorDType dtype) {
        this.filePath = filePath;
        this.dataOffset = dataOffset;
        this.dataLength = dataLength;
        this.shape = shape;
        this.dtype = dtype;
        this.isLazy = true;
        this.floatData = null;
        this.rawData = null;
    }

    // ==================== 核心方法 ====================

    /**
     * 获取 float[] 数据（自动转换）
     */
    public float[] getData() {
        if (floatData != null) {
            return floatData;
        }
        if (isLazy) {
            loadFromFile();
        }
        if (rawData != null) {
            floatData = convertToFloat();
            rawData = null; // 释放原始数据
        }
        return floatData;
    }

    /**
     * 获取原始 ByteBuffer 数据，无需转换为 float[]。
     * 对于懒加载的张量，这将从文件加载字节，但保持 rawData 不变。
     */
    public ByteBuffer getRawDataBuffer() {
        if (rawData == null && isLazy) {
            loadRawFromFile();
        }
        if (rawData == null) return null;
        ByteBuffer dup = rawData.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        dup.rewind();
        return dup;
    }

    /**
     * 获取形状
     */
    public int[] getShape() {
        return shape;
    }

    /**
     * 获取数据类型
     */
    public TensorDType getDType() {
        return dtype;
    }

    /**
     * 获取数据类型字符串
     */
    public String getDtypeString() {
        return dtype.getName();
    }

    /**
     * 获取元素总数
     */
    public int size() {
        int s = 1;
        for (int dim : shape) s *= dim;
        return s;
    }

    /**
     * 获取数据字节数
     */
    public long sizeBytes() {
        return (long) size() * dtype.size();
    }

    /**
     * 是否为懒加载模式
     */
    public boolean isLazy() {
        return isLazy;
    }

    /**
     * 是否已加载数据
     */
    public boolean isLoaded() {
        return floatData != null;
    }

    /**
     * 获取文件路径（懒加载场景）
     */
    public String getFilePath() {
        return filePath;
    }

    /**
     * 获取数据偏移（懒加载场景）
     */
    public long getDataOffset() {
        return dataOffset;
    }

    /**
     * 获取数据长度（懒加载场景）
     */
    public long getDataLength() {
        return dataLength;
    }

    /**
     * 强制加载数据（懒加载场景）
     */
    public TensorData load() {
        if (isLazy && floatData == null) {
            loadFromFile();
            if (rawData != null) {
                floatData = convertToFloat();
                rawData = null;
            }
        }
        return this;
    }

    /**
     * 释放数据，回到懒加载状态
     */
    public void unload() {
        if (isLazy) {
            floatData = null;
            rawData = null;
            mappedData = null;
            if (ownsNativePointer && nativePointer != null) {
                nativePointer.close();
            }
            nativePointer = null;
            nativeTensor = null;
        }
    }

    // ==================== 懒加载支持 ====================

    private void loadFromFile() {
        if (filePath == null) {
            throw new IllegalStateException("Cannot load: filePath is null");
        }
        try (java.io.RandomAccessFile raf = new java.io.RandomAccessFile(filePath, "r");
             FileChannel ch = raf.getChannel()) {

            // Validate file and offsets
            long fileSize = raf.length();
            if (dataOffset < 0 || dataOffset > fileSize) {
                throw new IllegalStateException(
                    String.format("Invalid offset: %d (file size: %d)", dataOffset, fileSize)
                );
            }
            if (dataOffset + dataLength > fileSize) {
                throw new IllegalStateException(
                    String.format("Data exceeds file: offset=%d, length=%d, file_size=%d",
                        dataOffset, dataLength, fileSize)
                );
            }

            if (isLargeTensor(dataLength)) {
                mappedData = ch.map(FileChannel.MapMode.READ_ONLY, dataOffset, dataLength).load();
                rawData = mappedData.duplicate().order(ByteOrder.LITTLE_ENDIAN);
                return;
            }

            rawData = ByteBuffer.allocateDirect((int) dataLength).order(ByteOrder.LITTLE_ENDIAN);
            ch.position(dataOffset);

            // Ensure all bytes are read
            long bytesRead = 0;
            while (bytesRead < dataLength) {
                int read = ch.read(rawData);
                if (read < 0) {
                    throw new EOFException(
                        String.format("Unexpected EOF: read %d/%d bytes", bytesRead, dataLength)
                    );
                }
                bytesRead += read;
            }

            rawData.flip();
        } catch (Exception e) {
            throw new RuntimeException(
                String.format("Failed to load tensor from file: %s (offset=%d, length=%d)",
                    filePath, dataOffset, dataLength), e);
        }
    }

    private void loadRawFromFile() {
        loadFromFile();
    }

    // ==================== 数据类型转换 ====================

    private float[] convertToFloat() {
        int totalSize = size();
        long expectedBytes = (long) totalSize * dtype.size();
        if (rawData == null || rawData.remaining() < expectedBytes) {
            long actual = rawData == null ? 0 : rawData.remaining();
            throw new IllegalStateException("Tensor raw buffer too small: expected " + expectedBytes + " bytes, got " + actual);
        }
        float[] result = new float[totalSize];
        ByteBuffer buf = rawData.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        buf.rewind();

        switch (dtype) {
            case F32 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = buf.getFloat();
                }
            }
            case F64 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = (float) buf.getDouble();
                }
            }
            case F16 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = float16ToFloat(buf.getShort());
                }
            }
            case BF16 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = bfloat16ToFloat(buf.getShort());
                }
            }
            case I64 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = (float) buf.getLong();
                }
            }
            case I32 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = (float) buf.getInt();
                }
            }
            case I16 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = buf.getShort();
                }
            }
            case I8 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = buf.get();
                }
            }
            case U8 -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = buf.get() & 0xFF;
                }
            }
            case BOOL -> {
                for (int i = 0; i < totalSize; i++) {
                    result[i] = buf.get() != 0 ? 1.0f : 0.0f;
                }
            }
        }
        return result;
    }

    private static float float16ToFloat(short bits) {
        int s = (bits >> 15) & 0x1;
        int e = (bits >> 10) & 0x1F;
        int m = bits & 0x3FF;

        if (e == 0) {
            if (m == 0) return s == 0 ? 0.0f : -0.0f;
            float mantissa = m / 1024.0f;
            float value = (float) (Math.pow(2, -14) * mantissa);
            return s == 0 ? value : -value;
        } else if (e == 31) {
            return m == 0 ? (s == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY) : Float.NaN;
        }

        int newExp = e - 15 + 127;
        int newMantissa = m << 13;
        int floatBits = (s << 31) | (newExp << 23) | newMantissa;
        return Float.intBitsToFloat(floatBits);
    }

    private static float bfloat16ToFloat(short bits) {
        int floatBits = (bits & 0xFFFF) << 16;
        return Float.intBitsToFloat(floatBits);
    }

    // ==================== 原有方法保留 ====================

    public TensorData reshape(int[] newShape) {
        int newSize = 1;
        for (int dim : newShape) newSize *= dim;
        if (newSize != size()) {
            throw new IllegalArgumentException("新形状的元素数量必须与原形状相同");
        }
        return new TensorData(getData().clone(), newShape);
    }

    @Override
    public Object toArrowCompatible() {
        Map<String, Object> arrowData = new HashMap<>();
        arrowData.put("shape", shape);
        arrowData.put("dtype", dtype.getName());
        arrowData.put("size", size());
        arrowData.put("lazy", isLazy);
        if (!isLazy || floatData != null) {
            arrowData.put("data", getData());
        }
        return arrowData;
    }

    @Override
    public String getShortDesc() {
        String loadStatus = isLazy ? (isLoaded() ? "loaded" : "lazy") : "eager";
        return String.format("shape=%s, dtype=%s, size=%d, %s",
                Arrays.toString(shape), dtype.getName(), size(), loadStatus);
    }

    @Override
    public int getSize() {
        return size();
    }

    @Override
    public Map<String, Object> toMap() {
        Map<String, Object> map = new HashMap<>();
        map.put("shape", shape);
        map.put("dtype", dtype.getName());
        map.put("size", size());
        map.put("lazy", isLazy);
        return map;
    }

    @Override
    public boolean isValid() {
        return super.isValid() && shape != null && shape.length > 0;
    }

    @Override
    public String getDataType() {
        return "TENSOR";
    }

    @Override
    public String toString() {
        if (isLazy && !isLoaded()) {
            return String.format("TensorData[shape=%s, dtype=%s, lazy=true]",
                    Arrays.toString(shape), dtype.getName());
        }
        return String.format("TensorData[shape=%s, dtype=%s]",
                Arrays.toString(shape), dtype.getName());
    }

    @Override
    public Number getNumericValue() {
        return null;
    }

    // ==================== Pickle 兼容方法 ====================

    public double[] getFlatData() {
        float[] data = getData();
        double[] flatData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            flatData[i] = data[i];
        }
        return flatData;
    }

    public int[] getTensorShape() {
        return Arrays.copyOf(shape, shape.length);
    }

    // ==================== 静态工厂方法 ====================

    /**
     * 创建懒加载的 TensorData
     */
    public static TensorData lazy(String filePath, long offset, long length, int[] shape, String dtype) {
        return new TensorData(filePath, offset, length, shape, TensorDType.fromString(dtype));
    }

    /**
     * 从 ByteBuffer 创建
     */
    public static TensorData fromBuffer(ByteBuffer buffer, int[] shape, String dtype) {
        return new TensorData(buffer, shape, TensorDType.fromString(dtype));
    }

    /**
     * Build from float storage with an explicit logical {@link TensorDType}
     * (e.g. F16/BF16 metadata on an F32 working buffer).
     */
    public static TensorData fromFloatData(float[] data, int[] shape, TensorDType dtype) {
        return new TensorData(data, shape, dtype == null ? TensorDType.F32 : dtype);
    }

    /** Copy a javacpp-pytorch {@link Tensor} into a {@link TensorData}. */
    public static TensorData fromTensor(Tensor t) {
        return TensorBridge.toTensorData(t);
    }

    /** Copy a javacpp-pytorch {@link Tensor}, optionally attaching the native handle. */
    public static TensorData fromTensor(Tensor t, boolean attachNative) {
        return TensorBridge.toTensorData(t, attachNative);
    }

    /** Convert this cell to a javacpp-pytorch {@link Tensor} (prefers attached native). */
    public Tensor toTensor() {
        return TensorBridge.toTensor(this);
    }

    public static TensorData fromNDArray(org.bytedeco.pytorch.data.numpy.NDArray arr) {
        return TensorBridge.toTensorData(arr);
    }

    public org.bytedeco.pytorch.data.numpy.NDArray toNDArray() {
        return TensorBridge.toNDArray(this);
    }

    /** Rank (number of dimensions). */
    public int ndim() {
        return shape == null ? 0 : shape.length;
    }

    public boolean hasNativeTensor() {
        return nativeTensor != null;
    }

    /**
     * Return attached native tensor (may be null). The caller MUST NOT free it.
     */
    public Tensor getNativeTensor() {
        return nativeTensor;
    }

    /**
     * Attach an externally created Torch tensor for zero-copy bridge.
     */
    public void attachNativeTensor(Tensor t) {
        this.nativeTensor = t;
    }

    /**
     * Attach a native buffer pointer when zero-copy is available.
     * @param ptr pointer to underlying data
     * @param owns whether TensorData should close it on unload
     */
    public void attachNativePointer(BytePointer ptr, boolean owns) {
        this.nativePointer = ptr;
        this.ownsNativePointer = owns;
    }

    public BytePointer getNativePointer() {
        return nativePointer;
    }

    public boolean ownsNativePointer() {
        return ownsNativePointer;
    }

    /**
     * Ensure a readable direct buffer view for zero-copy tensor creation.
     */
    public ByteBuffer ensureDirectBuffer() {
        if (nativePointer != null) {
            // bridge should wrap pointer directly; do not duplicate here
        }
        if (rawData == null && isLazy) {
            loadRawFromFile();
        }
        if (rawData != null) {
            ByteBuffer dup = rawData.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            dup.rewind();
            return dup;
        }
        if (mappedData != null) {
            ByteBuffer dup = mappedData.duplicate().order(ByteOrder.LITTLE_ENDIAN);
            dup.rewind();
            return dup;
        }
        return null;
    }

    // ==================== 静态工具方法 ====================

    public static boolean isLargeTensor(long bytes) {
        return bytes >= LARGE_TENSOR_THRESHOLD_BYTES || bytes > Integer.MAX_VALUE;
    }
}
