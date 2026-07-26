package org.bytedeco.pytorch.data.gguf;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * GGUF v3 writer for LibTorch {@link Tensor}s (float/int families).
 * Quantized GGML block packing is intentionally out of scope — callers that
 * need Q4/Q8 should pre-pack and supply raw {@code uint8} tensors with an
 * explicit GGML type via {@link #addRawTensor(String, byte[], long[], int)}.
 *
 * <pre>
 *   GGUFWriter w = new GGUFWriter(new File("model.gguf"));
 *   w.addMetadata("general.name", "demo");
 *   w.addTensor("weight", tensor);
 *   w.write();
 * </pre>
 */
public final class GGUFWriter {
    private final File file;
    private final int version;
    private final Map<String, Object> metadata = new LinkedHashMap<>();
    private final Map<String, TensorEntry> tensors = new LinkedHashMap<>();

    public GGUFWriter(File file) {
        this(file, GGUFConstants.GGUF_VERSION_LATEST);
    }

    public GGUFWriter(File file, int version) {
        if (file == null) throw new IllegalArgumentException("file");
        if (!GGUFConstants.isSupportedVersion(version)) {
            throw new IllegalArgumentException("unsupported GGUF version: " + version);
        }
        this.file = file;
        this.version = version;
    }

    public GGUFWriter addMetadata(String key, Object value) {
        metadata.put(key, value);
        return this;
    }

    public GGUFWriter addAllMetadata(Map<String, ?> map) {
        if (map != null) metadata.putAll(map);
        return this;
    }

    /** Add a floating/integer tensor (auto GGML type from torch dtype). */
    public GGUFWriter addTensor(String name, Tensor tensor) {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("name");
        if (tensor == null || !tensor.defined()) throw new IllegalArgumentException("tensor");
        Tensor cpu = tensor.contiguous().cpu();
        long[] shape = shapeOf(cpu);
        int ggml = ggmlTypeOf(cpu.scalar_type());
        byte[] raw = tensorToBytes(cpu, ggml);
        tensors.put(name, new TensorEntry(name, shape, ggml, raw));
        return this;
    }

    /** Add a pre-packed (possibly quantized) payload. */
    public GGUFWriter addRawTensor(String name, byte[] data, long[] shape, int ggmlType) {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("name");
        if (data == null) throw new IllegalArgumentException("data");
        tensors.put(name, new TensorEntry(name, shape != null ? shape.clone() : new long[0], ggmlType, data));
        return this;
    }

    public void write() throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "rw");
             FileChannel ch = raf.getChannel()) {
            raf.setLength(0);

            // header
            writeU32(ch, GGUFConstants.GGUF_MAGIC);
            writeU32(ch, version);
            writeU64(ch, tensors.size());
            writeU64(ch, metadata.size());

            // metadata
            for (Map.Entry<String, Object> e : metadata.entrySet()) {
                writeString(ch, e.getKey());
                writeValue(ch, e.getValue());
            }

            // tensor infos (offsets relative to data section)
            long dataOffset = 0;
            for (TensorEntry te : tensors.values()) {
                writeString(ch, te.name);
                writeU32(ch, te.shape.length);
                // GGUF stores dims reversed relative to row-major torch
                for (int i = te.shape.length - 1; i >= 0; i--) writeU64(ch, te.shape[i]);
                writeU32(ch, te.ggmlType);
                writeU64(ch, dataOffset);
                dataOffset += te.data.length;
            }

            // pad to alignment
            long alignment = GGUFConstants.ALIGNMENT;
            Object alignMeta = metadata.get(GGUFConstants.MetadataKeys.GENERAL_ALIGNMENT);
            if (alignMeta instanceof Number) alignment = ((Number) alignMeta).longValue();
            long pos = ch.position();
            long pad = (alignment - (pos % alignment)) % alignment;
            if (pad > 0) ch.write(ByteBuffer.allocate((int) pad));

            // payloads
            for (TensorEntry te : tensors.values()) {
                ch.write(ByteBuffer.wrap(te.data));
            }
        }
    }

    // ---- helpers ------------------------------------------------------------

    private static long[] shapeOf(Tensor t) {
        long[] s = new long[(int) t.dim()];
        for (int i = 0; i < s.length; i++) s[i] = t.sizes().get(i);
        return s;
    }

    private static int ggmlTypeOf(ScalarType st) {
        if (st.equals(torch.kFloat())) return GGUFConstants.GGML_TYPE_F32;
        if (st.equals(torch.kHalf())) return GGUFConstants.GGML_TYPE_F16;
        if (st.equals(torch.kDouble())) return GGUFConstants.GGML_TYPE_F64;
        if (st.equals(torch.kLong())) return GGUFConstants.GGML_TYPE_I64;
        if (st.equals(torch.kInt())) return GGUFConstants.GGML_TYPE_I32;
        if (st.equals(torch.kShort())) return GGUFConstants.GGML_TYPE_I16;
        if (st.equals(torch.kChar())) return GGUFConstants.GGML_TYPE_I8;
        if (st.equals(torch.kByte())) return GGUFConstants.GGML_TYPE_I8; // UINT8 maps to I8 storage
        if (st.equals(torch.kBFloat16())) return GGUFConstants.GGML_TYPE_BF16;
        // default: dump as F32
        return GGUFConstants.GGML_TYPE_F32;
    }

    private static byte[] tensorToBytes(Tensor t, int ggml) {
        long n = t.numel();
        ByteBuffer buf;
        switch (ggml) {
            case GGUFConstants.GGML_TYPE_F32: {
                Tensor f = t.to(torch.kFloat()).contiguous();
                org.bytedeco.javacpp.FloatPointer p = f.data_ptr_float();
                float[] data = new float[(int) n];
                for (int i = 0; i < n; i++) data[i] = p.get(i);
                buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                buf.asFloatBuffer().put(data);
                return buf.array();
            }
            case GGUFConstants.GGML_TYPE_F64: {
                Tensor d = t.to(torch.kDouble()).contiguous();
                org.bytedeco.javacpp.DoublePointer p = d.data_ptr_double();
                double[] data = new double[(int) n];
                for (int i = 0; i < n; i++) data[i] = p.get(i);
                buf = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);
                buf.asDoubleBuffer().put(data);
                return buf.array();
            }
            case GGUFConstants.GGML_TYPE_F16: {
                Tensor f = t.to(torch.kFloat()).contiguous();
                org.bytedeco.javacpp.FloatPointer p = f.data_ptr_float();
                buf = ByteBuffer.allocate((int) n * 2).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < n; i++) buf.putShort(floatToHalf(p.get(i)));
                return buf.array();
            }
            case GGUFConstants.GGML_TYPE_I64: {
                Tensor x = t.to(torch.kLong()).contiguous();
                org.bytedeco.javacpp.LongPointer p = x.data_ptr_long();
                long[] data = new long[(int) n];
                for (int i = 0; i < n; i++) data[i] = p.get(i);
                buf = ByteBuffer.allocate(data.length * 8).order(ByteOrder.LITTLE_ENDIAN);
                buf.asLongBuffer().put(data);
                return buf.array();
            }
            case GGUFConstants.GGML_TYPE_I32: {
                Tensor x = t.to(torch.kInt()).contiguous();
                org.bytedeco.javacpp.IntPointer p = x.data_ptr_int();
                int[] data = new int[(int) n];
                for (int i = 0; i < n; i++) data[i] = p.get(i);
                buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                buf.asIntBuffer().put(data);
                return buf.array();
            }
            case GGUFConstants.GGML_TYPE_I16: {
                Tensor x = t.to(torch.kShort()).contiguous();
                org.bytedeco.javacpp.ShortPointer p = x.data_ptr_short();
                short[] data = new short[(int) n];
                for (int i = 0; i < n; i++) data[i] = p.get(i);
                buf = ByteBuffer.allocate(data.length * 2).order(ByteOrder.LITTLE_ENDIAN);
                buf.asShortBuffer().put(data);
                return buf.array();
            }
            case GGUFConstants.GGML_TYPE_I8: {
                Tensor x = t.to(torch.kChar()).contiguous();
                org.bytedeco.javacpp.BytePointer p = x.data_ptr_char();
                byte[] data = new byte[(int) n];
                for (int i = 0; i < n; i++) data[i] = p.get(i);
                return data;
            }
            default: {
                Tensor f = t.to(torch.kFloat()).contiguous();
                org.bytedeco.javacpp.FloatPointer p = f.data_ptr_float();
                float[] data = new float[(int) n];
                for (int i = 0; i < n; i++) data[i] = p.get(i);
                buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
                buf.asFloatBuffer().put(data);
                return buf.array();
            }
        }
    }

    private static void writeU32(FileChannel ch, int v) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(v).flip();
        ch.write(b);
    }

    private static void writeU64(FileChannel ch, long v) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(v).flip();
        ch.write(b);
    }

    private static void writeString(FileChannel ch, String s) throws IOException {
        byte[] raw = s.getBytes(StandardCharsets.UTF_8);
        writeU64(ch, raw.length);
        ch.write(ByteBuffer.wrap(raw));
    }

    private static void writeValue(FileChannel ch, Object value) throws IOException {
        if (value instanceof Boolean) {
            writeU32(ch, GGUFConstants.VALUE_BOOL);
            ch.write(ByteBuffer.wrap(new byte[]{(byte) ((Boolean) value ? 1 : 0)}));
        } else if (value instanceof Byte) {
            writeU32(ch, GGUFConstants.VALUE_INT8);
            ch.write(ByteBuffer.wrap(new byte[]{(Byte) value}));
        } else if (value instanceof Short) {
            writeU32(ch, GGUFConstants.VALUE_INT16);
            ByteBuffer b = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
            b.putShort((Short) value).flip();
            ch.write(b);
        } else if (value instanceof Integer) {
            writeU32(ch, GGUFConstants.VALUE_INT32);
            writeU32(ch, (Integer) value);
        } else if (value instanceof Long) {
            writeU32(ch, GGUFConstants.VALUE_INT64);
            writeU64(ch, (Long) value);
        } else if (value instanceof Float) {
            writeU32(ch, GGUFConstants.VALUE_FLOAT32);
            ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            b.putFloat((Float) value).flip();
            ch.write(b);
        } else if (value instanceof Double) {
            writeU32(ch, GGUFConstants.VALUE_FLOAT64);
            ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            b.putDouble((Double) value).flip();
            ch.write(b);
        } else if (value instanceof String) {
            writeU32(ch, GGUFConstants.VALUE_STRING);
            writeString(ch, (String) value);
        } else if (value instanceof Object[]) {
            Object[] arr = (Object[]) value;
            writeU32(ch, GGUFConstants.VALUE_ARRAY);
            // infer element type from first element (default STRING)
            int elemType = GGUFConstants.VALUE_STRING;
            if (arr.length > 0) elemType = valueTypeOf(arr[0]);
            writeU32(ch, elemType);
            writeU64(ch, arr.length);
            for (Object o : arr) writeValueBody(ch, o, elemType);
        } else if (value instanceof Iterable) {
            java.util.List<Object> list = new java.util.ArrayList<>();
            for (Object o : (Iterable<?>) value) list.add(o);
            writeValue(ch, list.toArray());
        } else {
            // fallback: stringify
            writeU32(ch, GGUFConstants.VALUE_STRING);
            writeString(ch, String.valueOf(value));
        }
    }

    private static int valueTypeOf(Object v) {
        if (v instanceof Boolean) return GGUFConstants.VALUE_BOOL;
        if (v instanceof Byte) return GGUFConstants.VALUE_INT8;
        if (v instanceof Short) return GGUFConstants.VALUE_INT16;
        if (v instanceof Integer) return GGUFConstants.VALUE_INT32;
        if (v instanceof Long) return GGUFConstants.VALUE_INT64;
        if (v instanceof Float) return GGUFConstants.VALUE_FLOAT32;
        if (v instanceof Double) return GGUFConstants.VALUE_FLOAT64;
        return GGUFConstants.VALUE_STRING;
    }

    private static void writeValueBody(FileChannel ch, Object value, int vtype) throws IOException {
        switch (vtype) {
            case GGUFConstants.VALUE_BOOL:
                ch.write(ByteBuffer.wrap(new byte[]{(byte) (Boolean.TRUE.equals(value) ? 1 : 0)}));
                break;
            case GGUFConstants.VALUE_INT8:
                ch.write(ByteBuffer.wrap(new byte[]{((Number) value).byteValue()}));
                break;
            case GGUFConstants.VALUE_INT16: {
                ByteBuffer b = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
                b.putShort(((Number) value).shortValue()).flip();
                ch.write(b);
                break;
            }
            case GGUFConstants.VALUE_INT32:
                writeU32(ch, ((Number) value).intValue());
                break;
            case GGUFConstants.VALUE_INT64:
                writeU64(ch, ((Number) value).longValue());
                break;
            case GGUFConstants.VALUE_FLOAT32: {
                ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                b.putFloat(((Number) value).floatValue()).flip();
                ch.write(b);
                break;
            }
            case GGUFConstants.VALUE_FLOAT64: {
                ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                b.putDouble(((Number) value).doubleValue()).flip();
                ch.write(b);
                break;
            }
            default:
                writeString(ch, String.valueOf(value));
        }
    }

    private static short floatToHalf(float fval) {
        int fbits = Float.floatToIntBits(fval);
        int sign = (fbits >>> 16) & 0x8000;
        int val = (fbits & 0x7fffffff) + 0x1000;
        if (val >= 0x47800000) {
            if ((fbits & 0x7fffffff) >= 0x47800000) {
                if (val < 0x7f800000) return (short) (sign | 0x7c00);
                return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) return (short) (sign | ((val - 0x38000000) >>> 13));
        if (val < 0x33000000) return (short) sign;
        val = (fbits & 0x7fffffff) >>> 23;
        return (short) (sign | ((((fbits & 0x007fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
    }

    private static final class TensorEntry {
        final String name;
        final long[] shape;
        final int ggmlType;
        final byte[] data;

        TensorEntry(String name, long[] shape, int ggmlType, byte[] data) {
            this.name = name;
            this.shape = shape;
            this.ggmlType = ggmlType;
            this.data = data;
        }
    }
}
