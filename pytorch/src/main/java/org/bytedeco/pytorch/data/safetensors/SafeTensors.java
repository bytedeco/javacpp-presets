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
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.data.serialize.WeightBagModule;
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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
 * <p>Load paths:
 * <pre>
 *   // A) inject into an existing architecture-aware Module
 *   Map&lt;String, Tensor&gt; weights = SafeTensors.loadAsTensors(file, true);
 *   SafeTensors.loadIntoModule(module, weights, true);
 *
 *   // B) build a trainable typed Module from arbitrary safetensors (no Java
 *   //    architecture class required — Linear/Embedding/LayerNorm/… inferred)
 *   WeightBagModule bag = SafeTensors.toModule(file);
 *   bag.freezePrefix("embedding_layer.");
 *   Adam opt = new Adam(bag.parameters(), new AdamOptions(1e-4));
 *   bag.saveSafetensors(new File("finetuned.safetensors"));
 * </pre>
 */
public final class SafeTensors {
    /**
     * Prefer zero-copy {@code from_blob} when a tensor payload is at least this
     * large. Smaller tensors are always copied so short-lived maps stay safe.
     */
    private static final long LARGE_MMAP_THRESHOLD = 1L * 1024 * 1024; // 1 MiB

    /**
     * {@link FileChannel#map} size argument is effectively limited to
     * {@link Integer#MAX_VALUE} (~2 GiB) on HotSpot. Whole-file maps for
     * single-shard models like Llama-1B (~2.5 GiB) fail with
     * {@code Size exceeds Integer.MAX_VALUE}; map per-tensor (or chunk) instead.
     */
    private static final long MAX_MAP_BYTES = Integer.MAX_VALUE;

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

            long dataLen = ch.size() - hi.dataOffset;
            // Whole-file mmap only when the data region fits in one map (≤ ~2 GiB).
            // Larger single-shard checkpoints (e.g. Llama-1B) use per-tensor maps.
            MappedByteBuffer wholeMap = null;
            if (zeroCopy && dataLen > 0 && dataLen <= MAX_MAP_BYTES) {
                wholeMap = ch.map(FileChannel.MapMode.READ_ONLY, hi.dataOffset, dataLen);
                wholeMap.order(ByteOrder.LITTLE_ENDIAN);
                PINNED_MAPS.add(wholeMap);
            }

            for (Map.Entry<String, TensorMeta> e : meta.entrySet()) {
                String name = e.getKey();
                TensorMeta m = e.getValue();
                SafeDType dtype = SafeDType.fromString(m.dtype);
                if (dtype == null) {
                    throw new IOException("unknown safetensors dtype: " + m.dtype + " for tensor " + name);
                }
                long start = m.dataOffsets[0];
                long end = m.dataOffsets[1];
                long nbytes = end - start;
                long[] shape = m.shape;

                Tensor t;
                if (zeroCopy && nbytes >= LARGE_MMAP_THRESHOLD
                        && start >= 0 && start + nbytes <= dataLen
                        && dtype.isNativeLayout()
                        && nbytes <= MAX_MAP_BYTES) {
                    if (wholeMap != null) {
                        t = fromMappedRegion(wholeMap, start, nbytes, shape, dtype);
                    } else {
                        // Per-tensor mmap — works for files larger than Integer.MAX_VALUE.
                        MappedByteBuffer slice = ch.map(
                                FileChannel.MapMode.READ_ONLY,
                                hi.dataOffset + start,
                                nbytes);
                        slice.order(ByteOrder.LITTLE_ENDIAN);
                        PINNED_MAPS.add(slice);
                        t = fromMappedRegion(slice, 0, nbytes, shape, dtype);
                    }
                } else {
                    if (nbytes > Integer.MAX_VALUE) {
                        throw new IOException("tensor too large to copy: " + name + " (" + nbytes + " bytes)");
                    }
                    // FileChannel.read may return partial; loop until full or EOF.
                    ByteBuffer buf = ByteBuffer.allocateDirect((int) nbytes).order(ByteOrder.LITTLE_ENDIAN);
                    ch.position(hi.dataOffset + start);
                    int got = 0;
                    while (buf.hasRemaining()) {
                        int n = ch.read(buf);
                        if (n < 0) {
                            throw new EOFException("short read for " + name
                                    + " got=" + got + " need=" + nbytes
                                    + " (file may be truncated)");
                        }
                        got += n;
                    }
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
        // Materialize contiguous CPU copies and compute offsets.
        // Defensive: skip null / undefined / dangling @ByRef handles so a single
        // bad leaf (e.g. BatchNorm num_batches_tracked after retain failed) does
        // not SIGSEGV the whole save.
        Map<String, byte[]> payloads = new LinkedHashMap<>();
        Map<String, TensorMeta> metas = new LinkedHashMap<>();
        long offset = 0;
        for (Map.Entry<String, Tensor> e : tensors.entrySet()) {
            String name = e.getKey();
            Tensor src = e.getValue();
            if (src == null || src.isNull()) continue;
            Tensor t;
            try {
                if (!src.defined()) continue;
                // Retain first so contiguous()/cpu() can't race a temporary ByRef.
                t = new Tensor(src).contiguous().cpu();
            } catch (Throwable ex) {
                System.err.println("SafeTensors.save: skip '" + name + "': " + ex);
                continue;
            }
            if (t == null || t.isNull() || !t.defined()) continue;
            SafeDType dtype = SafeDType.fromTorch(t.scalar_type());
            long[] shape = shapeOf(t);
            byte[] raw = tensorToBytes(t, dtype);
            long[] offs = new long[]{offset, offset + raw.length};
            offset += raw.length;
            payloads.put(name, raw);
            metas.put(name, new TensorMeta(dtype.typeName(), shape, offs));
        }
        if (payloads.isEmpty()) {
            throw new IOException("SafeTensors.save: no valid tensors to write for " + file);
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

    // ---- state-dict → trainable Module (arbitrary safetensors) ------------------

    /**
     * Build a trainable typed {@link WeightBagModule} from a safetensors file.
     * Reads {@code module_structure} metadata when present so ReLU/Dropout are exact.
     */
    public static WeightBagModule toModule(File file) throws IOException {
        return WeightBagModule.fromSafetensors(file, true);
    }

    /**
     * @param requiresGrad set requires_grad on every parameter
     * @param zeroCopy     when loading the file, prefer mmap/from_blob for large
     *                     tensors (still cloned into the bag when requiresGrad)
     */
    public static WeightBagModule toModule(File file, boolean requiresGrad, boolean zeroCopy)
            throws IOException {
        // zeroCopy is honored inside fromSafetensors via loadAsTensors(true) by default;
        // for explicit control, load then build:
        Map<String, Tensor> weights = loadAsTensors(file, zeroCopy);
        Map<String, String> meta = readMetadata(file);
        Map<String, String> structure = null;
        if (meta != null) {
            String enc = meta.get("module_structure");
            if (enc == null) enc = meta.get("structure");
            if (enc != null && !enc.isEmpty()) {
                structure = org.bytedeco.pytorch.data.serialize.StateDictModuleBuilder
                        .decodeStructureMeta(enc);
            }
        }
        return new WeightBagModule(weights, requiresGrad, true, true, structure);
    }

    public static WeightBagModule toModule(String path) throws IOException {
        return toModule(new File(path));
    }

    /**
     * Build a trainable typed {@link WeightBagModule} from an in-memory state-dict.
     * Clones tensors so the bag owns storage; infers Linear/Embedding/… leaves
     * and Sequential gap-fill for ReLU/Dropout.
     */
    public static WeightBagModule toModule(Map<String, Tensor> weights) {
        return toModule(weights, true);
    }

    public static WeightBagModule toModule(Map<String, Tensor> weights, boolean requiresGrad) {
        return WeightBagModule.fromTyped(weights, requiresGrad);
    }

    /**
     * Save a live Module's {@code named_parameters(true)} to safetensors.
     *
     * @return number of tensors written
     */
    public static int saveModule(Module module, File file) throws IOException {
        return saveModule(module, file, null);
    }

    public static int saveModule(Module module, File file, Map<String, String> metadata)
            throws IOException {
        if (module == null) throw new IllegalArgumentException("module required");
        Map<String, Tensor> sd = collectNamedParameters(module);
        if (sd.isEmpty()) {
            throw new IOException("module has no named_parameters to save: " + module);
        }
        save(sd, file, metadata);
        return sd.size();
    }

    // ---- inject into existing Module ----------------------------------------

    /**
     * Copy matching named parameters from {@code weights} into {@code module}.
     *
     * <p>Primary path: {@code module.named_parameters(true)} exact key match,
     * then loose suffix / prefix variants. Falls back to Linear/Embedding
     * typed walk when named_parameters is empty (rare).
     *
     * @param strict if true, missing module keys or shape mismatches throw
     * @return number of parameters written
     */
    public static int loadIntoModule(Module module, Map<String, Tensor> weights, boolean strict) {
        if (module == null || weights == null || weights.isEmpty()) return 0;

        Map<String, Tensor> params = collectNamedParameters(module);
        if (!params.isEmpty()) {
            return loadIntoNamedParameters(params, weights, strict);
        }

        // Fallback for modules that don't surface named_parameters yet
        int written = 0;
        try {
            written += loadIntoModuleRecursive(module, "", weights, strict);
        } catch (RuntimeException e) {
            if (strict) throw e;
        }
        return written;
    }

    /**
     * Zero-copy convenience: load safetensors and inject into a Module.
     */
    public static int loadModuleFromFile(Module module, File file, boolean zeroCopy, boolean strict)
            throws IOException {
        Map<String, Tensor> weights = loadAsTensors(file, zeroCopy);
        return loadIntoModule(module, weights, strict);
    }

    /**
     * Collect {@code module.named_parameters(recurse=true)} as a Java map.
     */
    public static Map<String, Tensor> collectNamedParameters(Module module) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        if (module == null) return out;
        try {
            StringTensorDict dict = module.named_parameters(/*recurse=*/true);
            if (dict == null || dict.isNull()) return out;
            long n = dict.size();
            for (long i = 0; i < n; i++) {
                StringTensorDictItem item = dict.get(i);
                if (item == null || item.isNull()) continue;
                String key = item.key() != null ? item.key().getString() : null;
                Tensor val = item.value();
                if (key == null || val == null) continue;
                out.put(key, val);
            }
        } catch (Throwable ignored) {
            // return whatever we have
        }
        return out;
    }

    // ---- named_parameters injection -----------------------------------------

    private static int loadIntoNamedParameters(Map<String, Tensor> params,
                                               Map<String, Tensor> weights,
                                               boolean strict) {
        int written = 0;
        List<String> missing = new ArrayList<>();
        List<String> shapeMismatch = new ArrayList<>();
        Set<String> used = new LinkedHashSet<>();

        for (Map.Entry<String, Tensor> pe : params.entrySet()) {
            String key = pe.getKey();
            Tensor dest = pe.getValue();
            if (dest == null || !dest.defined()) {
                missing.add(key);
                continue;
            }
            Tensor src = weights.get(key);
            if (src == null) src = findLoose(weights, key);
            if (src == null || !src.defined()) {
                missing.add(key);
                continue;
            }
            if (!shapesEqual(src, dest)) {
                shapeMismatch.add(key + " src=" + shapeStr(src) + " dest=" + shapeStr(dest));
                continue;
            }
            try (org.bytedeco.pytorch.NoGradGuard guard = new org.bytedeco.pytorch.NoGradGuard()) {
                dest.copy_(src);
            }
            written++;
            used.add(key);
        }

        if (strict) {
            if (!missing.isEmpty() || !shapeMismatch.isEmpty()) {
                throw new IllegalStateException(
                        "loadIntoModule strict failure: missing=" + missing
                                + " shapeMismatch=" + shapeMismatch
                                + " written=" + written);
            }
        }
        return written;
    }

    private static Tensor findLoose(Map<String, Tensor> weights, String key) {
        if (weights.containsKey(key)) return weights.get(key);
        // strip common wrappers
        String[] strips = {"module.", "model.", "net.", "state_dict."};
        for (String s : strips) {
            if (key.startsWith(s) && weights.containsKey(key.substring(s.length()))) {
                return weights.get(key.substring(s.length()));
            }
            String with = s + key;
            if (weights.containsKey(with)) return weights.get(with);
        }
        for (Map.Entry<String, Tensor> e : weights.entrySet()) {
            String k = e.getKey();
            if (k.equals(key) || k.endsWith("." + key) || key.endsWith("." + k)) {
                return e.getValue();
            }
        }
        return null;
    }

    private static boolean shapesEqual(Tensor a, Tensor b) {
        if (a.dim() != b.dim()) return false;
        for (int i = 0; i < a.dim(); i++) {
            if (a.sizes().get(i) != b.sizes().get(i)) return false;
        }
        return true;
    }

    private static String shapeStr(Tensor t) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < t.dim(); i++) {
            if (i > 0) sb.append(',');
            sb.append(t.sizes().get(i));
        }
        return sb.append(']').toString();
    }

    // ---- recursive Linear/Embedding fallback --------------------------------

    private static int loadIntoModuleRecursive(Module m, String prefix,
                                               Map<String, Tensor> weights, boolean strict) {
        int n = 0;
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
        if (src == null) src = findLoose(weights, key);
        if (src == null) {
            if (strict) throw new IllegalStateException("Missing weight key: " + key);
            return 0;
        }
        if (!shapesEqual(src, dest)) {
            if (strict) throw new IllegalStateException("Shape mismatch for " + key
                    + " src=" + shapeStr(src) + " dest=" + shapeStr(dest));
            return 0;
        }
        try (org.bytedeco.pytorch.NoGradGuard guard = new org.bytedeco.pytorch.NoGradGuard()) {
            dest.copy_(src);
        }
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
            case F8_E4M3:
            case F8_E5M2: {
                // raw bytes → UInt8 view → cast to Float8 torch dtype
                byte[] data = new byte[(int) n];
                buf.get(data);
                Tensor t = torch.tensor(data).to(torch.kByte());
                if (shape.length > 0) t = t.reshape(shape);
                return t.view(dtype.toTorch());
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

    /**
     * Read {@code __metadata__} string map from a safetensors header
     * (no tensor payloads loaded). Used for {@code module_structure} etc.
     */
    public static Map<String, String> readMetadata(File file) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(file, "r");
             FileChannel ch = raf.getChannel()) {
            HeaderInfo hi = readHeader(ch);
            return parseMetadata(hi.json);
        }
    }

    /**
     * Parse {@code "__metadata__":{...}} from a safetensors header JSON.
     * Values may contain {@code ;} / {@code =} (structure meta encoding).
     */
    static Map<String, String> parseMetadata(String json) {
        Map<String, String> out = new LinkedHashMap<>();
        if (json == null) return out;
        int idx = json.indexOf("\"__metadata__\"");
        if (idx < 0) return out;
        int brace = json.indexOf('{', idx);
        if (brace < 0) return out;
        // scan matching brace (no nested objects expected in our metadata values)
        int depth = 0;
        int end = -1;
        for (int i = brace; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0) { end = i; break; }
            }
        }
        if (end < 0) return out;
        String body = json.substring(brace + 1, end);
        // Match "key":"value" with escaped quotes
        Pattern p = Pattern.compile("\"((?:\\\\.|[^\"\\\\])*)\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
        Matcher m = p.matcher(body);
        while (m.find()) {
            out.put(unescapeJson(m.group(1)), unescapeJson(m.group(2)));
        }
        return out;
    }

    private static String unescapeJson(String s) {
        if (s == null) return null;
        return s.replace("\\\"", "\"").replace("\\\\", "\\");
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
