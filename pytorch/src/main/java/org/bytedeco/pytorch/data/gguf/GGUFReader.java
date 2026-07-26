package org.bytedeco.pytorch.data.gguf;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * GGUF v2/v3 reader that loads tensor payloads into LibTorch {@link Tensor}s.
 * Quantized GGML types are returned as raw {@code uint8} byte blobs; floating
 * / integer types are decoded to the matching torch dtype.
 *
 * <pre>
 *   try (GGUFReader r = new GGUFReader(new File("model.gguf"))) {
 *       Tensor w = r.loadTensor("blk.0.attn_q.weight");
 *       Map&lt;String, Tensor&gt; all = r.loadAll();
 *   }
 * </pre>
 *
 * @see GGUFWriter
 * @see GGUFConstants
 * @see <a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">GGUF spec</a>
 */
public final class GGUFReader implements AutoCloseable {

    private final File file;
    private RandomAccessFile raf;
    private FileChannel channel;
    private int version;
    private final Map<String, Object> metadata = new LinkedHashMap<>();
    private final Map<String, TensorInfo> tensors = new LinkedHashMap<>();
    private long tensorDataOffset;

    public GGUFReader(File file) throws IOException {
        this.file = file;
        this.raf = new RandomAccessFile(file, "r");
        this.channel = raf.getChannel();
        readHeader();
    }

    public File file() { return file; }
    public int version() { return version; }
    public Map<String, Object> metadata() { return Collections.unmodifiableMap(metadata); }
    public Map<String, TensorInfo> tensorInfos() { return Collections.unmodifiableMap(tensors); }
    public long tensorDataOffset() { return tensorDataOffset; }

    /** Load one tensor by name (decoded when possible). */
    public Tensor loadTensor(String name) throws IOException {
        TensorInfo info = tensors.get(name);
        if (info == null) throw new IllegalArgumentException("unknown tensor: " + name);
        long nbytes = info.nBytes();
        if (nbytes > Integer.MAX_VALUE) throw new IOException("tensor too large: " + name);
        ByteBuffer buf = ByteBuffer.allocate((int) nbytes).order(ByteOrder.LITTLE_ENDIAN);
        channel.position(tensorDataOffset + info.offset);
        int read = channel.read(buf);
        if (read != nbytes) throw new EOFException("short read " + name + " expected=" + nbytes + " got=" + read);
        buf.flip();
        return decode(buf, info);
    }

    /** Load all tensors (may be large — prefer {@link #loadTensor} for selective use). */
    public Map<String, Tensor> loadAll() throws IOException {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (String name : tensors.keySet()) {
            out.put(name, loadTensor(name));
        }
        return out;
    }

    @Override
    public void close() throws IOException {
        if (raf != null) {
            raf.close();
            raf = null;
            channel = null;
        }
    }

    // ---- header -------------------------------------------------------------

    private void readHeader() throws IOException {
        // Spec: magic u32 | version u32 | n_tensors u64 | n_kv u64
        ByteBuffer h = ByteBuffer.allocate(4 + 4 + 8 + 8).order(ByteOrder.LITTLE_ENDIAN);
        if (channel.read(h) != h.capacity()) throw new EOFException("gguf header");
        h.flip();
        int magic = h.getInt();
        if (magic != GGUFConstants.GGUF_MAGIC) {
            throw new IOException("not a GGUF file (bad magic 0x" + Integer.toHexString(magic) + ")");
        }
        version = h.getInt();
        if (!GGUFConstants.isSupportedVersion(version)) {
            throw new IOException("unsupported GGUF version: " + version);
        }
        long nTensors = h.getLong();
        long nKv = h.getLong();

        for (long i = 0; i < nKv; i++) {
            String key = readString();
            int vtype = readU32();
            Object val = readValue(vtype);
            metadata.put(key, val);
        }

        for (long i = 0; i < nTensors; i++) {
            String name = readString();
            int nDims = readU32();
            long[] shape = new long[nDims];
            for (int d = 0; d < nDims; d++) shape[d] = readU64();
            // GGUF stores dims reversed relative to row-major torch convention
            reverse(shape);
            int ggmlType = readU32();
            long offset = readU64();
            tensors.put(name, new TensorInfo(name, shape, ggmlType, offset));
        }

        long alignment = GGUFConstants.ALIGNMENT;
        Object alignMeta = metadata.get(GGUFConstants.MetadataKeys.GENERAL_ALIGNMENT);
        if (alignMeta instanceof Number) alignment = ((Number) alignMeta).longValue();
        long pos = channel.position();
        tensorDataOffset = (pos + alignment - 1) / alignment * alignment;
    }

    private Tensor decode(ByteBuffer buf, TensorInfo info) {
        int t = info.ggmlType;
        long[] shape = info.shape;
        long n = info.nElements();
        switch (t) {
            case GGUFConstants.GGML_TYPE_F32: {
                float[] data = new float[(int) n];
                buf.asFloatBuffer().get(data);
                Tensor ten = torch.tensor(data);
                return shape.length > 0 ? ten.reshape(shape) : ten;
            }
            case GGUFConstants.GGML_TYPE_F16: {
                float[] data = new float[(int) n];
                for (int i = 0; i < n; i++) data[i] = halfToFloat(buf.getShort());
                Tensor ten = torch.tensor(data);
                return (shape.length > 0 ? ten.reshape(shape) : ten).to(torch.kHalf());
            }
            case GGUFConstants.GGML_TYPE_BF16: {
                // BF16: top 16 bits of float32
                float[] data = new float[(int) n];
                for (int i = 0; i < n; i++) {
                    int bits = (buf.getShort() & 0xffff) << 16;
                    data[i] = Float.intBitsToFloat(bits);
                }
                Tensor ten = torch.tensor(data);
                return shape.length > 0 ? ten.reshape(shape) : ten;
            }
            case GGUFConstants.GGML_TYPE_F64: {
                double[] data = new double[(int) n];
                buf.asDoubleBuffer().get(data);
                Tensor ten = torch.tensor(data);
                return shape.length > 0 ? ten.reshape(shape) : ten;
            }
            case GGUFConstants.GGML_TYPE_I32: {
                int[] data = new int[(int) n];
                buf.asIntBuffer().get(data);
                Tensor ten = torch.tensor(data);
                return shape.length > 0 ? ten.reshape(shape) : ten;
            }
            case GGUFConstants.GGML_TYPE_I64: {
                long[] data = new long[(int) n];
                buf.asLongBuffer().get(data);
                Tensor ten = torch.tensor(data);
                return shape.length > 0 ? ten.reshape(shape) : ten;
            }
            case GGUFConstants.GGML_TYPE_I16: {
                short[] data = new short[(int) n];
                buf.asShortBuffer().get(data);
                Tensor ten = torch.tensor(data);
                return shape.length > 0 ? ten.reshape(shape) : ten;
            }
            case GGUFConstants.GGML_TYPE_I8: {
                byte[] data = new byte[(int) n];
                buf.get(data);
                Tensor ten = torch.tensor(data);
                return shape.length > 0 ? ten.reshape(shape) : ten;
            }
            default: {
                // Quantized / unknown: return raw bytes [nbytes]
                byte[] raw = new byte[buf.remaining()];
                buf.get(raw);
                return torch.tensor(raw).to(torch.kByte());
            }
        }
    }

    // ---- primitives ---------------------------------------------------------

    private int readU32() throws IOException {
        ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        if (channel.read(b) != 4) throw new EOFException();
        b.flip();
        return b.getInt();
    }

    private long readU64() throws IOException {
        ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        if (channel.read(b) != 8) throw new EOFException();
        b.flip();
        return b.getLong();
    }

    private byte readU8() throws IOException {
        ByteBuffer b = ByteBuffer.allocate(1);
        if (channel.read(b) != 1) throw new EOFException();
        return b.array()[0];
    }

    private short readU16() throws IOException {
        ByteBuffer b = ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN);
        if (channel.read(b) != 2) throw new EOFException();
        b.flip();
        return b.getShort();
    }

    private String readString() throws IOException {
        long len = readU64();
        if (len < 0 || len > 10_000_000) throw new IOException("bad string len " + len);
        byte[] raw = new byte[(int) len];
        ByteBuffer b = ByteBuffer.wrap(raw);
        if (channel.read(b) != len) throw new EOFException();
        return new String(raw, StandardCharsets.UTF_8);
    }

    /**
     * GGUF value types (ggml/gguf.h):
     * 0=UINT8 1=INT8 2=UINT16 3=INT16 4=UINT32 5=INT32 6=FLOAT32
     * 7=BOOL 8=STRING 9=ARRAY 10=UINT64 11=INT64 12=FLOAT64
     */
    private Object readValue(int vtype) throws IOException {
        switch (vtype) {
            case GGUFConstants.VALUE_UINT8:  return readU8() & 0xff;
            case GGUFConstants.VALUE_INT8:   return (int) readU8();
            case GGUFConstants.VALUE_UINT16: return readU16() & 0xffff;
            case GGUFConstants.VALUE_INT16:  return (int) readU16();
            case GGUFConstants.VALUE_UINT32: return readU32() & 0xffffffffL;
            case GGUFConstants.VALUE_INT32:  return readU32();
            case GGUFConstants.VALUE_FLOAT32: {
                ByteBuffer b = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                if (channel.read(b) != 4) throw new EOFException();
                b.flip();
                return b.getFloat();
            }
            case GGUFConstants.VALUE_BOOL:
                return readU8() != 0;
            case GGUFConstants.VALUE_STRING:
                return readString();
            case GGUFConstants.VALUE_ARRAY: {
                int elemType = readU32();
                long n = readU64();
                if (n < 0 || n > 10_000_000) throw new IOException("bad array len " + n);
                Object[] arr = new Object[(int) n];
                for (int i = 0; i < arr.length; i++) arr[i] = readValue(elemType);
                return arr;
            }
            case GGUFConstants.VALUE_UINT64:
            case GGUFConstants.VALUE_INT64:
                return readU64();
            case GGUFConstants.VALUE_FLOAT64: {
                ByteBuffer b = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                if (channel.read(b) != 8) throw new EOFException();
                b.flip();
                return b.getDouble();
            }
            default:
                throw new IOException("unknown GGUF value type: " + vtype);
        }
    }

    private static void reverse(long[] a) {
        for (int i = 0, j = a.length - 1; i < j; i++, j--) {
            long t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }

    private static float halfToFloat(short hbits) {
        int mant = hbits & 0x03ff;
        int exp = hbits & 0x7c00;
        if (exp == 0x7c00) exp = 0x3fc00;
        else if (exp != 0) exp += 0x1c000;
        else if (mant != 0) {
            exp = 0x1c400;
            do { mant <<= 1; exp -= 0x400; } while ((mant & 0x400) == 0);
            mant &= 0x3ff;
        }
        return Float.intBitsToFloat((hbits & 0x8000) << 16 | (exp | mant) << 13);
    }

    /** Tensor metadata entry. */
    public static final class TensorInfo {
        public final String name;
        public final long[] shape;
        public final int ggmlType;
        public final long offset; // relative to tensor data section

        TensorInfo(String name, long[] shape, int ggmlType, long offset) {
            this.name = name;
            this.shape = shape;
            this.ggmlType = ggmlType;
            this.offset = offset;
        }

        public long nElements() {
            long n = 1;
            for (long s : shape) n *= s;
            return n;
        }

        public long nBytes() {
            return GGUFConstants.nbytes(ggmlType, nElements());
        }

        @Override
        public String toString() {
            return "TensorInfo{name='" + name + "', shape=" + java.util.Arrays.toString(shape)
                    + ", ggmlType=" + ggmlType + ", offset=" + offset + "}";
        }
    }
}
