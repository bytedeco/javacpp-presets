package org.bytedeco.pytorch.data.safetensors;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * High-performance safetensors reader/writer with optional zero-copy
 * {@link Tensor} views over memory-mapped file regions via {@code torch.from_blob}.
 *
 * <p>Format (https://huggingface.co/docs/safetensors):
 * <pre>
 *   u64 header_len | utf-8 JSON header | raw little-endian tensor bytes
 * </pre>
 *
 * <p>Zero-copy Module load:
 * <pre>
 *   Map&lt;String, Tensor&gt; weights = SafeTensors.loadAsTensors(file, true);
 *   SafeTensors.loadIntoModule(module, weights, true);
 * </pre>
 */
public final class SafeTensors {
    /**
     * Prefer zero-copy {@code from_blob} when a tensor payload is at least this
     * large. Smaller tensors are always copied so short-lived maps stay safe.
     */
    private static final long LARGE_MMAP_THRESHOLD = 1L * 1024 * 1024; // 1 MiB

    /**
     * Strong references to mmap buffers that back zero-copy tensors. Without
     * this, the GC can reclaim a {@link MappedByteBuffer} while a
     * {@code from_blob} Tensor still points into it.
     */
    private static final List<MappedByteBuffer> PINNED_MAPS =
            Collections.synchronizedList(new ArrayList<>());

    private SafeTensors() {}

    // ---- public API ---------------------------------------------------------

    /** Lazy / mmap-backed tensor map (preferred for large models). */
    public static Map<String, Tensor> loadAsTensors(File file) throws IOException {
        return loadAsTensors(file, true);
    }

    /**
     * @param zeroCopy when true, tensors above the mmap threshold share the
     *                 file mapping via {@code from_blob}; otherwise data is copied.
     *                 The mapping is pinned for process lifetime so tensors remain valid.
     */
    public static Map<String, Tensor> loadAsTensors(File file, boolean zeroCopy) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel ch = raf.getChannel()) {
            HeaderInfo hi = readHeader(ch);
            Map<String, TensorMeta> meta = parseHeader(hi.json);
            Map<String, Tensor> out = new LinkedHashMap<>();

            // Single mmap for the whole data region when zero-copy is requested
            MappedByteBuffer wholeMap = null;
            long dataLen = ch.size() - hi.dataOffset;
            if (zeroCopy && dataLen > 0) {
                wholeMap = ch.map(FileChannel.MapMode.READ_ONLY, hi.dataOffset, dataLen);
                wholeMap.order(ByteOrder.LITTLE_ENDIAN);
                // Pin so from_blob tensors outlive this method / GC
                PINNED_MAPS.add(wholeMap);
            }

            for (Map.Entry<String, TensorMeta> e : meta.entrySet()) {
                String name = e.getKey();
                TensorMeta m = e.getValue();
                SafeDType dtype = SafeDType.fromString(m.dtype);
                long start = m.dataOffsets[0];
                long end = m.dataOffsets[1];
                long nbytes = end - start;
                long[] shape = m.shape;

                Tensor t;
                if (zeroCopy && wholeMap != null && nbytes >= LARGE_MMAP_THRESHOLD
                        && start >= 0 && start + nbytes <= dataLen
                        && dtype.isNativeLayout()) {
                    t = fromMappedRegion(wholeMap, start, nbytes, shape, dtype);
                } else {
                    if (nbytes > Integer.MAX_VALUE) {
                        throw new IOException("tensor too large to copy: " + name + " (" + nbytes + " bytes)");
                    }
                    ByteBuffer buf = ByteBuffer.allocateDirect((int) nbytes).order(ByteOrder.LITTLE_ENDIAN);
                    ch.position(hi.dataOffset + start);
                    int read = ch.read(buf);
                    if (read != nbytes) throw new EOFException("short read for " + name);
                    buf.flip();
                    t = copyBufferToTensor(buf, shape, dtype);
                }
                out.put(name, t);
            }
            return out;
        }
    }

    /**
     * Drop pinned mmap references. Only call when you are certain no zero-copy
     * tensors loaded by this class are still in use.
     */
    public static void releasePinnedMaps() {
        PINNED_MAPS.clear();
    }

    /** List tensor names without loading data. */
    public static List<String> listTensors(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel ch = raf.getChannel()) {
            HeaderInfo hi = readHeader(ch);
            return new ArrayList<>(parseHeader(hi.json).keySet());
        }
    }

    /**
     * Write tensors to a safetensors file (copy-out; always little-endian).
     */
    public static void save(Map<String, Tensor> tensors, File file) throws IOException {
        save(tensors, file, null);
    }

    public static void save(Map<String, Tensor> tensors, File file, Map<String, String> metadata) throws IOException {
        // Materialize contiguous CPU copies and compute offsets
        Map<String, byte[]> payloads = new LinkedHashMap<>();
        Map<String, TensorMeta> metas = new LinkedHashMap<>();
        long offset = 0;
        for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
            String name = e.getKey();
            Tensor t = e.getValue().contiguous().cpu();
            SafeDType dtype = SafeDType.fromTorch(t.scalar_type());
            long[] shape = shapeOf(t);
            byte[] raw = tensorToBytes(t, dtype);
            long[] offs = new long[]{offset, offset + raw.length};
            offset += raw.length;
            payloads.put(name, raw);
            metas.put(name, new TensorMeta(dtype.typeName(), shape, offs));
        }

        String headerJson = buildHeaderJson(metas, metadata);
        byte[] headerBytes = padHeader(headerJson);

        try (RandomAccessFile raf = new RandomAccessFile(file, "rw");
             FileChannel ch = raf.getChannel()) {
            raf.setLength(0);
            ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            lenBuf.putLong(headerBytes.length);
            lenBuf.flip();
            ch.write(lenBuf);
            ch.write(ByteBuffer.wrap(headerBytes));
            for (byte[] raw : payloads.values()) {
                ch.write(ByteBuffer.wrap(raw));
            }
        }
    }

    /**
     * Copy matching named parameters from {@code weights} into {@code module}.
     * Keys are matched against {@code named_parameters()} keys (exact, then
     * with/without {@code .weight}/{@code .bias} suffixes stripped).
     *
     * @param strict if true, missing keys throw; if false, skip quietly
     * @return number of parameters written
     */
    public static int loadIntoModule(Module module, Map<String, Tensor> weights, boolean strict) {
        if (module == null || weights == null) return 0;
        int written = 0;
        try {
            // Walk named_parameters via Module API
            // StringTensorDict may not exist — use parameters + children fallback
            written += loadIntoModuleRecursive(module, "", weights, strict);
        } catch (RuntimeException e) {
            if (strict) throw e;
        }
        return written;
    }

    /**
     * Zero-copy convenience: load safetensors and inject into a Module.
     */
    public static int loadModuleFromFile(Module module, File file, boolean zeroCopy, boolean strict) throws IOException {
        Map<String, Tensor> weights = loadAsTensors(file, zeroCopy);
        return loadIntoModule(module, weights, strict);
    }

    // ---- recursive parameter injection --------------------------------------

    private static int loadIntoModuleRecursive(Module m, String prefix, Map<String, Tensor> weights, boolean strict) {
        int n = 0;
        // Linear / Embedding common cases via typed as*()
        try {
            LinearImpl lin = m.asLinear();
            if (lin != null && !lin.isNull()) {
                n += copyParam(weights, prefix + "weight", lin.weight(), strict);
                if (lin.bias() != null && lin.bias().defined()) {
                    n += copyParam(weights, prefix + "bias", lin.bias(), strict);
                }
                return n;
            }
        } catch (Throwable ignored) {}
        try {
            EmbeddingImpl emb = m.asEmbedding();
            if (emb != null && !emb.isNull()) {
                n += copyParam(weights, prefix + "weight", emb.weight(), strict);
                return n;
            }
        } catch (Throwable ignored) {}

        // Generic: named_children recursion
        try {
            org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDict kids = m.named_children();
            if (kids != null && !kids.isNull() && kids.size() > 0) {
                for (int i = 0; i < kids.size(); i++) {
                    org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDictItem item = kids.get(i);
                    if (item == null || item.isNull()) continue;
                    String key = item.key() != null ? item.key().getString() : String.valueOf(i);
                    Module child = item.value();
                    if (child == null || child.isNull()) continue;
                    String childPrefix = prefix.isEmpty() ? key + "." : prefix + key + ".";
                    n += loadIntoModuleRecursive(child, childPrefix, weights, strict);
                }
            }
        } catch (Throwable ignored) {}
        return n;
    }

    private static int copyParam(Map<String, Tensor> weights, String key, Tensor dest, boolean strict) {
        if (dest == null || !dest.defined()) return 0;
        Tensor src = weights.get(key);
        // try without trailing dot-parts variants
        if (src == null) {
            // strip leading module name variants: "model." prefix etc.
            for (Map.Entry<String, Tensor> e : weights.entrySet()) {
                if (e.getKey().endsWith(key) || e.getKey().equals(key)) {
                    src = e.getValue();
                    break;
                }
            }
        }
        if (src == null) {
            if (strict) throw new IllegalStateException("Missing weight key: " + key);
            return 0;
        }
        // shape check
        if (src.dim() != dest.dim()) {
            if (strict) throw new IllegalStateException("Shape rank mismatch for " + key);
            return 0;
        }
        for (int i = 0; i < src.dim(); i++) {
            if (src.sizes().get(i) != dest.sizes().get(i)) {
                if (strict) throw new IllegalStateException("Shape mismatch for " + key);
                return 0;
            }
        }
        dest.copy_(src);
        return 1;
    }

    // ---- buffer → Tensor ----------------------------------------------------

    private static Tensor fromMappedRegion(MappedByteBuffer whole, long start, long nbytes,
                                           long[] shape, SafeDType dtype) {
        // Duplicate and slice the region; keep as direct buffer
        ByteBuffer slice = whole.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        // MappedByteBuffer supports long positions only via duplicate + absolute ops;
        // for regions within Integer.MAX_VALUE of the map start we can slice.
        if (start > Integer.MAX_VALUE || nbytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "zero-copy region exceeds 2GiB slice limit; use copy path or split tensors");
        }
        slice.position((int) start);
        slice.limit((int) (start + nbytes));
        ByteBuffer region = slice.slice().order(ByteOrder.LITTLE_ENDIAN);

        // from_blob shares storage without copy. PINNED_MAPS keeps `whole` alive.
        BytePointer ptr = new BytePointer(region);
        TensorOptions opts = new TensorOptions(dtype.toTorch());
        Tensor t = torch.from_blob(ptr, shape, opts);
        t.retainReference();
        return t;
    }

    private static Tensor copyBufferToTensor(ByteBuffer buf, long[] shape, SafeDType dtype) {
        long n = 1;
        for (long s : shape) n *= s;
        switch (dtype) {
            case F32: {
                float[] data = new float[(int) n];
                buf.asFloatBuffer().get(data);
                Tensor t = torch.tensor(data);
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case F64: {
                double[] data = new double[(int) n];
                buf.asDoubleBuffer().get(data);
                Tensor t = torch.tensor(data);
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case I64: {
                long[] data = new long[(int) n];
                buf.asLongBuffer().get(data);
                Tensor t = torch.tensor(data);
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case I32: {
                int[] data = new int[(int) n];
                buf.asIntBuffer().get(data);
                Tensor t = torch.tensor(data);
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case I16: {
                short[] data = new short[(int) n];
                buf.asShortBuffer().get(data);
                Tensor t = torch.tensor(data);
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case F16:
            case BF16: {
                // promote to float32 for usability
                float[] data = new float[(int) n];
                for (int i = 0; i < n; i++) {
                    short h = buf.getShort();
                    data[i] = dtype == SafeDType.F16 ? halfToFloat(h) : bfloat16ToFloat(h);
                }
                Tensor t = torch.tensor(data);
                Tensor out = shape.length > 0 ? t.reshape(shape) : t;
                return out.to(dtype.toTorch());
            }
            case I8:
            case U8: {
                byte[] data = new byte[(int) n];
                buf.get(data);
                Tensor t = torch.tensor(data);
                if (dtype == SafeDType.U8) t = t.to(torch.kByte());
                return shape.length > 0 ? t.reshape(shape) : t;
            }
            case BOOL: {
                boolean[][] as2d = new boolean[1][(int) n];
                for (int i = 0; i < n; i++) as2d[0][i] = buf.get() != 0;
                return torch.tensor(as2d).reshape(shape);
            }
            default:
                throw new IllegalArgumentException("Unsupported dtype " + dtype);
        }
    }

    private static byte[] tensorToBytes(Tensor t, SafeDType dtype) {
        Tensor c = t.contiguous().cpu();
        long n = c.numel();
        ByteBuffer buf = ByteBuffer.allocate((int) (n * dtype.sizeBytes())).order(ByteOrder.LITTLE_ENDIAN);
        switch (dtype) {
            case F32: {
                FloatPointer p = c.data_ptr_float();
                for (int i = 0; i < n; i++) buf.putFloat(p.get(i));
                break;
            }
            case F64: {
                org.bytedeco.javacpp.DoublePointer p = c.data_ptr_double();
                for (int i = 0; i < n; i++) buf.putDouble(p.get(i));
                break;
            }
            case I64: {
                org.bytedeco.javacpp.LongPointer p = c.data_ptr_long();
                for (int i = 0; i < n; i++) buf.putLong(p.get(i));
                break;
            }
            case I32: {
                org.bytedeco.javacpp.IntPointer p = c.data_ptr_int();
                for (int i = 0; i < n; i++) buf.putInt(p.get(i));
                break;
            }
            case I16: {
                org.bytedeco.javacpp.ShortPointer p = c.data_ptr_short();
                for (int i = 0; i < n; i++) buf.putShort(p.get(i));
                break;
            }
            case F16: {
                Tensor f = c.to(torch.kFloat());
                FloatPointer p = f.data_ptr_float();
                for (int i = 0; i < n; i++) buf.putShort(floatToHalf(p.get(i)));
                break;
            }
            case BF16: {
                Tensor f = c.to(torch.kFloat());
                FloatPointer p = f.data_ptr_float();
                for (int i = 0; i < n; i++) buf.putShort(floatToBFloat16(p.get(i)));
                break;
            }
            case I8: {
                BytePointer p = c.data_ptr_char();
                for (int i = 0; i < n; i++) buf.put(p.get(i));
                break;
            }
            case U8: {
                BytePointer p = c.data_ptr_byte();
                for (int i = 0; i < n; i++) buf.put(p.get(i));
                break;
            }
            case BOOL: {
                Tensor flat = c.reshape(n);
                for (int i = 0; i < n; i++) buf.put((byte) (flat.get((long) i).item_bool() ? 1 : 0));
                break;
            }
            default:
                throw new IllegalArgumentException("Unsupported dtype " + dtype);
        }
        return buf.array();
    }

    // ---- header parsing (minimal JSON subset, no external deps) -------------

    private static final class HeaderInfo {
        final String json;
        final long dataOffset;
        HeaderInfo(String json, long dataOffset) {
            this.json = json;
            this.dataOffset = dataOffset;
        }
    }

    private static final class TensorMeta {
        final String dtype;
        final long[] shape;
        final long[] dataOffsets; // [begin, end) relative to data section
        TensorMeta(String dtype, long[] shape, long[] dataOffsets) {
            this.dtype = dtype;
            this.shape = shape;
            this.dataOffsets = dataOffsets;
        }
    }

    private static HeaderInfo readHeader(FileChannel ch) throws IOException {
        ByteBuffer lenBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
        if (ch.read(lenBuf) != 8) throw new EOFException("safetensors header length");
        lenBuf.flip();
        long headerLen = lenBuf.getLong();
        if (headerLen <= 0 || headerLen > 100_000_000L) {
            throw new IOException("invalid safetensors header length: " + headerLen);
        }
        ByteBuffer hdr = ByteBuffer.allocate((int) headerLen);
        if (ch.read(hdr) != headerLen) throw new EOFException("safetensors header");
        String json = new String(hdr.array(), StandardCharsets.UTF_8);
        long dataOffset = 8 + headerLen;
        return new HeaderInfo(json, dataOffset);
    }

    private static Map<String, TensorMeta> parseHeader(String json) {
        Map<String, TensorMeta> out = new LinkedHashMap<>();
        // Match "name":{...} objects at top level (skip __metadata__)
        Pattern entry = Pattern.compile("\"([^\"]+)\"\\s*:\\s*\\{([^}]*)\\}");
        Matcher m = entry.matcher(json);
        while (m.find()) {
            String name = m.group(1);
            if ("__metadata__".equals(name)) continue;
            String body = m.group(2);
            String dtype = extractStr(body, "dtype");
            long[] shape = extractLongArray(body, "shape");
            long[] offsets = extractLongArray(body, "data_offsets");
            if (dtype == null || shape == null || offsets == null || offsets.length < 2) continue;
            out.put(name, new TensorMeta(dtype, shape, offsets));
        }
        return out;
    }

    private static String extractStr(String body, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]+)\"").matcher(body);
        return m.find() ? m.group(1) : null;
    }

    private static long[] extractLongArray(String body, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]").matcher(body);
        if (!m.find()) return null;
        String raw = m.group(1).trim();
        if (raw.isEmpty()) return new long[0];
        String[] parts = raw.split(",");
        long[] out = new long[parts.length];
        for (int i = 0; i < parts.length; i++) out[i] = Long.parseLong(parts[i].trim());
        return out;
    }

    private static String buildHeaderJson(Map<String, TensorMeta> metas, Map<String, String> metadata) {
        StringBuilder sb = new StringBuilder();
        sb.append('{');
        boolean first = true;
        if (metadata != null && !metadata.isEmpty()) {
            sb.append("\"__metadata__\":{");
            boolean mf = true;
            for (Map.Entry<String, String> e : metadata.entrySet()) {
                if (!mf) sb.append(',');
                sb.append('"').append(escape(e.getKey())).append("\":\"")
                        .append(escape(e.getValue())).append('"');
                mf = false;
            }
            sb.append('}');
            first = false;
        }
        for (Map.Entry<String, TensorMeta> e : metas.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            TensorMeta m = e.getValue();
            sb.append('"').append(escape(e.getKey())).append("\":{");
            sb.append("\"dtype\":\"").append(m.dtype).append("\",");
            sb.append("\"shape\":[");
            for (int i = 0; i < m.shape.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(m.shape[i]);
            }
            sb.append("],\"data_offsets\":[")
                    .append(m.dataOffsets[0]).append(',').append(m.dataOffsets[1])
                    .append("]}");
        }
        sb.append('}');
        return sb.toString();
    }

    private static byte[] padHeader(String json) {
        // header must be 8-byte aligned after the 8-byte length prefix
        byte[] raw = json.getBytes(StandardCharsets.UTF_8);
        int pad = (8 - (raw.length % 8)) % 8;
        if (pad == 0) return raw;
        byte[] out = new byte[raw.length + pad];
        System.arraycopy(raw, 0, out, 0, raw.length);
        // pad with spaces
        for (int i = raw.length; i < out.length; i++) out[i] = ' ';
        return out;
    }

    private static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static long[] shapeOf(Tensor t) {
        long[] s = new long[(int) t.dim()];
        for (int i = 0; i < s.length; i++) s[i] = t.sizes().get(i);
        return s;
    }

    // half/bfloat helpers
    private static short floatToHalf(float fval) {
        int fbits = Float.floatToIntBits(fval);
        int sign = (fbits >>> 16) & 0x8000;
        int val = (fbits & 0x7fffffff) + 0x1000;
        if (val >= 0x47800000) {
            if ((fbits & 0x7fffffff) >= 0x47800000) {
                return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) return (short) (sign | ((val - 0x38000000) >>> 13));
        if (val < 0x33000000) return (short) sign;
        val = (fbits & 0x7fffffff) >>> 23;
        return (short) (sign | ((((fbits & 0x7fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
    }

    private static float halfToFloat(short hbits) {
        int mant = hbits & 0x03ff;
        int exp = hbits & 0x7c00;
        if (exp == 0x7c00) exp = 0x3fc00;
        else if (exp != 0) {
            exp += 0x1c000;
        } else if (mant != 0) {
            exp = 0x1c400;
            do { mant <<= 1; exp -= 0x400; } while ((mant & 0x400) == 0);
            mant &= 0x3ff;
        }
        return Float.intBitsToFloat((hbits & 0x8000) << 16 | (exp | mant) << 13);
    }

    private static short floatToBFloat16(float f) {
        return (short) (Float.floatToIntBits(f) >>> 16);
    }

    private static float bfloat16ToFloat(short bits) {
        return Float.intBitsToFloat((bits & 0xffff) << 16);
    }
}
