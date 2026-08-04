package org.bytedeco.pytorch.data.safetensors;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;

import java.io.Closeable;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Streaming / lazy view over a single {@code .safetensors} file — pure-Java
 * counterpart of Python {@code safetensors.safe_open}.
 *
 * <p>Header is parsed once; tensor payloads are materialised on demand via
 * {@link #getTensor(String)} (mmap/{@code from_blob} when possible). Useful for
 * models larger than host RAM where only a subset of weights is needed at a time
 * (LoRA merge, layer-wise offload, selective fine-tune).
 *
 * <pre>{@code
 * try (SafeOpen so = SafeOpen.open(new File("model.safetensors"))) {
 *     System.out.println(so.keys());
 *     System.out.println(so.metadata());
 *     Tensor w = so.getTensor("model.layers.0.self_attn.q_proj.weight");
 * }
 * }</pre>
 *
 * <p>The underlying {@link FileChannel} and any mmap buffers stay pinned for the
 * lifetime of this object. Closing releases the channel; zero-copy tensors that
 * were returned remain valid only while the process pins their buffers (same
 * contract as {@link SafeTensors#loadAsTensors}).
 */
public final class SafeOpen implements Closeable {

    private static final long LARGE_MMAP_THRESHOLD = 1L * 1024 * 1024;
    private static final long MAX_MAP_BYTES = Integer.MAX_VALUE;

    private final File file;
    private final RandomAccessFile raf;
    private final FileChannel channel;
    private final long dataOffset;
    private final long dataLen;
    private final Map<String, TensorInfo> infos;
    private final Map<String, String> metadata;
    private final List<MappedByteBuffer> pinned = new ArrayList<>();
    private final boolean zeroCopy;
    private MappedByteBuffer wholeMap;
    private boolean closed;

    private SafeOpen(File file, RandomAccessFile raf, FileChannel channel,
                     long dataOffset, long dataLen,
                     Map<String, TensorInfo> infos, Map<String, String> metadata,
                     boolean zeroCopy) throws IOException {
        this.file = file;
        this.raf = raf;
        this.channel = channel;
        this.dataOffset = dataOffset;
        this.dataLen = dataLen;
        this.infos = infos;
        this.metadata = metadata;
        this.zeroCopy = zeroCopy;
        if (zeroCopy && dataLen > 0 && dataLen <= MAX_MAP_BYTES) {
            this.wholeMap = channel.map(FileChannel.MapMode.READ_ONLY, dataOffset, dataLen);
            this.wholeMap.order(ByteOrder.LITTLE_ENDIAN);
            this.pinned.add(this.wholeMap);
        }
    }

    // ---- factories ----------------------------------------------------------

    public static SafeOpen open(File file) throws IOException {
        return open(file, true);
    }

    public static SafeOpen open(Path path) throws IOException {
        return open(path.toFile(), true);
    }

    public static SafeOpen open(String path) throws IOException {
        return open(new File(path), true);
    }

    /**
     * @param zeroCopy prefer mmap/from_blob for large tensors (default true)
     */
    public static SafeOpen open(File file, boolean zeroCopy) throws IOException {
        Objects.requireNonNull(file, "file");
        if (!file.isFile()) throw new IOException("not a file: " + file);
        RandomAccessFile raf = new RandomAccessFile(file, "r");
        FileChannel ch = raf.getChannel();
        try {
            // header: u64 len + utf-8 JSON
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
            long dataLen = ch.size() - dataOffset;
            Map<String, TensorInfo> infos = parseInfos(json);
            Map<String, String> meta = SafeTensors.parseMetadata(json);
            return new SafeOpen(file, raf, ch, dataOffset, dataLen, infos, meta, zeroCopy);
        } catch (IOException | RuntimeException e) {
            try { ch.close(); } catch (IOException ignored) {}
            try { raf.close(); } catch (IOException ignored) {}
            throw e;
        }
    }

    // ---- queries ------------------------------------------------------------

    public File file() { return file; }

    /** Ordered tensor names (insertion order from header). */
    public List<String> keys() {
        return new ArrayList<>(infos.keySet());
    }

    public boolean contains(String name) {
        return infos.containsKey(name);
    }

    public int size() {
        return infos.size();
    }

    /** Header {@code __metadata__} map (may be empty). */
    public Map<String, String> metadata() {
        return Collections.unmodifiableMap(metadata);
    }

    /** Shape of a named tensor without loading payload. */
    public long[] shape(String name) {
        TensorInfo ti = require(name);
        return ti.shape.clone();
    }

    /** On-disk dtype name (e.g. {@code F16}, {@code BF16}, {@code F8_E4M3}). */
    public String dtype(String name) {
        return require(name).dtype;
    }

    /** Byte length of the on-disk payload. */
    public long nbytes(String name) {
        TensorInfo ti = require(name);
        return ti.end - ti.start;
    }

    /** Structured info for every tensor (no payloads). */
    public Map<String, TensorInfo> tensorInfos() {
        return Collections.unmodifiableMap(infos);
    }

    // ---- materialize --------------------------------------------------------

    /**
     * Load one tensor by name. Large native-layout tensors use zero-copy when
     * {@code zeroCopy=true} was requested at open.
     */
    public Tensor getTensor(String name) throws IOException {
        ensureOpen();
        TensorInfo ti = require(name);
        SafeDType dtype = SafeDType.fromString(ti.dtype);
        if (dtype == null) {
            throw new IOException("unknown safetensors dtype: " + ti.dtype + " for " + name);
        }
        long nbytes = ti.end - ti.start;
        if (zeroCopy && nbytes >= LARGE_MMAP_THRESHOLD
                && ti.start >= 0 && ti.start + nbytes <= dataLen
                && dtype.isNativeLayout()
                && nbytes <= MAX_MAP_BYTES) {
            if (wholeMap != null) {
                return fromMappedRegion(wholeMap, ti.start, nbytes, ti.shape, dtype);
            }
            MappedByteBuffer slice = channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    dataOffset + ti.start,
                    nbytes);
            slice.order(ByteOrder.LITTLE_ENDIAN);
            pinned.add(slice);
            return fromMappedRegion(slice, 0, nbytes, ti.shape, dtype);
        }
        // copy path
        if (nbytes > Integer.MAX_VALUE) {
            throw new IOException("tensor too large to copy: " + name + " (" + nbytes + " bytes)");
        }
        ByteBuffer buf = ByteBuffer.allocateDirect((int) nbytes).order(ByteOrder.LITTLE_ENDIAN);
        channel.position(dataOffset + ti.start);
        int got = 0;
        while (buf.hasRemaining()) {
            int n = channel.read(buf);
            if (n < 0) {
                throw new EOFException("short read for " + name + " got=" + got + " need=" + nbytes);
            }
            got += n;
        }
        buf.flip();
        return SafeTensors.copyBufferToTensorPublic(buf, ti.shape, dtype);
    }

    /**
     * Materialise every tensor (equivalent to {@link SafeTensors#loadAsTensors}
     * but reuses this open handle / maps).
     */
    public Map<String, Tensor> loadAll() throws IOException {
        Map<String, Tensor> out = new LinkedHashMap<>(infos.size());
        for (String k : infos.keySet()) {
            out.put(k, getTensor(k));
        }
        return out;
    }

    /**
     * Load a subset of tensors by name. Missing names are skipped (or throw when
     * {@code strict}).
     */
    public Map<String, Tensor> loadSlice(Iterable<String> names, boolean strict) throws IOException {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (String n : names) {
            if (!infos.containsKey(n)) {
                if (strict) throw new IOException("tensor not in file: " + n);
                continue;
            }
            out.put(n, getTensor(n));
        }
        return out;
    }

    // ---- close --------------------------------------------------------------

    @Override
    public void close() throws IOException {
        if (closed) return;
        closed = true;
        // Keep pinned maps alive via SafeTensors global pin list so already-returned
        // zero-copy tensors remain valid after close (Python safe_open keeps frames
        // only while open; we opt for process-lifetime pins for training friendliness).
        for (MappedByteBuffer m : pinned) {
            SafeTensors.pinMappedBuffer(m);
        }
        pinned.clear();
        wholeMap = null;
        try { channel.close(); } finally { raf.close(); }
    }

    // ---- internals ----------------------------------------------------------

    private void ensureOpen() throws IOException {
        if (closed) throw new IOException("SafeOpen already closed: " + file);
    }

    private TensorInfo require(String name) {
        TensorInfo ti = infos.get(name);
        if (ti == null) throw new IllegalArgumentException("unknown tensor: " + name);
        return ti;
    }

    private static Tensor fromMappedRegion(MappedByteBuffer whole, long start, long nbytes,
                                           long[] shape, SafeDType dtype) {
        if (start > Integer.MAX_VALUE || nbytes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "zero-copy region exceeds 2GiB slice limit; use copy path or split tensors");
        }
        ByteBuffer slice = whole.duplicate().order(ByteOrder.LITTLE_ENDIAN);
        slice.position((int) start);
        slice.limit((int) (start + nbytes));
        ByteBuffer region = slice.slice().order(ByteOrder.LITTLE_ENDIAN);
        BytePointer ptr = new BytePointer(region);
        TensorOptions opts = new TensorOptions(dtype.toTorch());
        Tensor t = torch.from_blob(ptr, shape, opts);
        t.retainReference();
        return t;
    }

    private static Map<String, TensorInfo> parseInfos(String json) {
        Map<String, TensorInfo> out = new LinkedHashMap<>();
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
            out.put(name, new TensorInfo(name, dtype, shape, offsets[0], offsets[1]));
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

    /** Public tensor descriptor (header only). */
    public static final class TensorInfo {
        public final String name;
        public final String dtype;
        public final long[] shape;
        public final long start;
        public final long end;

        TensorInfo(String name, String dtype, long[] shape, long start, long end) {
            this.name = name;
            this.dtype = dtype;
            this.shape = shape;
            this.start = start;
            this.end = end;
        }

        public long nbytes() { return end - start; }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder(name).append(':').append(dtype).append('[');
            for (int i = 0; i < shape.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(shape[i]);
            }
            return sb.append("] bytes=").append(nbytes()).toString();
        }
    }
}
