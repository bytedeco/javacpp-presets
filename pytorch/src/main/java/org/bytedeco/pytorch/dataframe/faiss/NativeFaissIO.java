package org.bytedeco.pytorch.dataframe.faiss;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Official FAISS binary index format — wire-compatible with Python
 * {@code faiss.write_index} / {@code faiss.read_index}.
 *
 * <p>Layout follows Meta FAISS {@code faiss/impl/index_write.cpp} /
 * {@code index_read.cpp} (little-endian, host float32/int64/size_t=8 on LP64).
 *
 * <p>Supported fourccs:
 * <ul>
 *   <li>{@code IxF2} / {@code IxFI} / {@code IxFl} — IndexFlat</li>
 *   <li>{@code IHNf} — IndexHNSWFlat (+ nested Flat storage)</li>
 *   <li>{@code IxMp} / {@code IxM2} — IndexIDMap</li>
 *   <li>{@code IwPQ} — IndexIVFPQ</li>
 *   <li>{@code null} — empty nested index</li>
 * </ul>
 *
 * <p>Also auto-detects our custom {@code JDF1} Java-serialization format
 * (legacy) and delegates to {@link IndexIO}.
 */
public final class NativeFaissIO {
    private NativeFaissIO() {}

    // ---- fourcc helpers ----

    public static int fourcc(String s) {
        if (s == null || s.length() != 4)
            throw new IllegalArgumentException("fourcc must be 4 chars: " + s);
        byte[] b = s.getBytes(StandardCharsets.US_ASCII);
        return (b[0] & 0xFF)
            | ((b[1] & 0xFF) << 8)
            | ((b[2] & 0xFF) << 16)
            | ((b[3] & 0xFF) << 24);
    }

    public static String fourccString(int h) {
        char[] c = new char[4];
        c[0] = (char) (h & 0xFF);
        c[1] = (char) ((h >>> 8) & 0xFF);
        c[2] = (char) ((h >>> 16) & 0xFF);
        c[3] = (char) ((h >>> 24) & 0xFF);
        return new String(c);
    }

    public static final int FCC_IxF2 = fourcc("IxF2");
    public static final int FCC_IxFI = fourcc("IxFI");
    public static final int FCC_IxFl = fourcc("IxFl");
    public static final int FCC_IHNf = fourcc("IHNf");
    public static final int FCC_IxMp = fourcc("IxMp");
    public static final int FCC_IxM2 = fourcc("IxM2");
    public static final int FCC_IwPQ = fourcc("IwPQ");
    public static final int FCC_null = fourcc("null");
    public static final int FCC_ilar = fourcc("ilar");
    public static final int FCC_il00 = fourcc("il00");
    public static final int FCC_full = fourcc("full");
    public static final int FCC_sprs = fourcc("sprs");
    /** Our legacy Java-serialization magic 'JDF1'. */
    public static final int FCC_JDF1 = IndexIO.MAGIC;

    // ---- public API ----

    public static void write(Index index, Path path) throws IOException {
        if (index == null) throw new IllegalArgumentException("null index");
        if (index.is_gpu()) index.to_cpu_storage();
        try (OutputStream raw = Files.newOutputStream(path);
             BufferedOutputStream bos = new BufferedOutputStream(raw, 1 << 20);
             FaissWriter w = new FaissWriter(bos)) {
            writeIndex(index, w);
            w.flush();
        }
    }

    public static void write(Index index, String path) throws IOException {
        write(index, Path.of(path));
    }

    public static Index read(Path path) throws IOException {
        // Peek magic / fourcc to decide format
        byte[] head = new byte[4];
        try (InputStream in = Files.newInputStream(path)) {
            int n = in.read(head);
            if (n < 4) throw new IOException("file too small: " + path);
        }
        int magic = (head[0] & 0xFF)
            | ((head[1] & 0xFF) << 8)
            | ((head[2] & 0xFF) << 16)
            | ((head[3] & 0xFF) << 24);
        // JDF1 is big-endian int written by DataOutputStream → on LE host bytes are reversed
        // DataOutputStream.writeInt writes big-endian: JDF1 = 0x4A444631 → bytes 4A 44 46 31
        int be = ((head[0] & 0xFF) << 24)
            | ((head[1] & 0xFF) << 16)
            | ((head[2] & 0xFF) << 8)
            | (head[3] & 0xFF);
        if (be == IndexIO.MAGIC || magic == IndexIO.MAGIC) {
            try {
                return IndexIO.read(path);
            } catch (ClassNotFoundException e) {
                throw new IOException("JDF1 payload class missing", e);
            }
        }
        try (InputStream raw = Files.newInputStream(path);
             BufferedInputStream bis = new BufferedInputStream(raw, 1 << 20);
             FaissReader r = new FaissReader(bis)) {
            return readIndex(r);
        }
    }

    public static Index read(String path) throws IOException {
        return read(Path.of(path));
    }

    /** True if path looks like native FAISS (not JDF1). */
    public static boolean isNativeFaissFile(Path path) throws IOException {
        byte[] head = new byte[4];
        try (InputStream in = Files.newInputStream(path)) {
            if (in.read(head) < 4) return false;
        }
        int be = ((head[0] & 0xFF) << 24)
            | ((head[1] & 0xFF) << 16)
            | ((head[2] & 0xFF) << 8)
            | (head[3] & 0xFF);
        if (be == IndexIO.MAGIC) return false;
        // FAISS fourccs are ASCII letters
        for (byte b : head) {
            if (b < 0x20 || b > 0x7e) return false;
        }
        return true;
    }

    // =====================================================================
    // Writer
    // =====================================================================

    static final class FaissWriter implements Closeable {
        private final OutputStream out;
        private final byte[] buf8 = new byte[8];

        FaissWriter(OutputStream out) { this.out = out; }

        void writeBytes(byte[] b, int off, int len) throws IOException {
            out.write(b, off, len);
        }

        void writeInt(int v) throws IOException {
            buf8[0] = (byte) v;
            buf8[1] = (byte) (v >>> 8);
            buf8[2] = (byte) (v >>> 16);
            buf8[3] = (byte) (v >>> 24);
            out.write(buf8, 0, 4);
        }

        void writeLong(long v) throws IOException {
            for (int i = 0; i < 8; i++) buf8[i] = (byte) (v >>> (8 * i));
            out.write(buf8, 0, 8);
        }

        void writeFloat(float v) throws IOException {
            writeInt(Float.floatToIntBits(v));
        }

        void writeDouble(double v) throws IOException {
            writeLong(Double.doubleToLongBits(v));
        }

        void writeBool(boolean v) throws IOException {
            // FAISS bool is typically 1 byte
            out.write(v ? 1 : 0);
        }

        void writeRawByte(int b) throws IOException {
            out.write(b & 0xFF);
        }

        int readRawByte() throws IOException {
            throw new UnsupportedOperationException("use FaissReader");
        }

        /** size_t on LP64 FAISS builds = uint64. */
        void writeSize(long v) throws IOException {
            writeLong(v);
        }

        void writeIntArray(int[] a, int n) throws IOException {
            ByteBuffer bb = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) bb.putInt(a[i]);
            out.write(bb.array());
        }

        void writeLongArray(long[] a, int n) throws IOException {
            ByteBuffer bb = ByteBuffer.allocate(n * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) bb.putLong(a[i]);
            out.write(bb.array());
        }

        void writeFloatArray(float[] a, int off, int n) throws IOException {
            ByteBuffer bb = ByteBuffer.allocate(n * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) bb.putFloat(a[off + i]);
            out.write(bb.array());
        }

        void writeDoubleArray(double[] a, int n) throws IOException {
            ByteBuffer bb = ByteBuffer.allocate(n * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < n; i++) bb.putDouble(a[i]);
            out.write(bb.array());
        }

        /** WRITEVECTOR: size_t count + elements. */
        void writeVectorInt(int[] v) throws IOException {
            writeSize(v == null ? 0 : v.length);
            if (v != null && v.length > 0) writeIntArray(v, v.length);
        }

        void writeVectorLong(long[] v) throws IOException {
            writeSize(v == null ? 0 : v.length);
            if (v != null && v.length > 0) writeLongArray(v, v.length);
        }

        void writeVectorFloat(float[] v) throws IOException {
            writeSize(v == null ? 0 : v.length);
            if (v != null && v.length > 0) writeFloatArray(v, 0, v.length);
        }

        void writeVectorDouble(double[] v) throws IOException {
            writeSize(v == null ? 0 : v.length);
            if (v != null && v.length > 0) writeDoubleArray(v, v.length);
        }

        void writeVectorSize(long[] v) throws IOException {
            writeVectorLong(v);
        }

        /**
         * WRITEXBVECTOR: codes stored as uint8 vector but size written as
         * {@code codes.size()/4} (legacy float-count), then raw bytes.
         */
        void writeXbVector(byte[] codes) throws IOException {
            if (codes == null) codes = new byte[0];
            if ((codes.length & 3) != 0)
                throw new IOException("XB vector length not multiple of 4: " + codes.length);
            writeSize(codes.length / 4L);
            if (codes.length > 0) out.write(codes);
        }

        void flush() throws IOException { out.flush(); }

        @Override public void close() throws IOException { out.close(); }
    }

    // =====================================================================
    // Reader
    // =====================================================================

    static final class FaissReader implements Closeable {
        private final InputStream in;
        private final byte[] buf8 = new byte[8];

        FaissReader(InputStream in) { this.in = in; }

        void readFully(byte[] b, int off, int len) throws IOException {
            int got = 0;
            while (got < len) {
                int n = in.read(b, off + got, len - got);
                if (n < 0) throw new EOFException("unexpected EOF need=" + len + " got=" + got);
                got += n;
            }
        }

        int readInt() throws IOException {
            readFully(buf8, 0, 4);
            return (buf8[0] & 0xFF)
                | ((buf8[1] & 0xFF) << 8)
                | ((buf8[2] & 0xFF) << 16)
                | ((buf8[3] & 0xFF) << 24);
        }

        long readLong() throws IOException {
            readFully(buf8, 0, 8);
            long v = 0;
            for (int i = 0; i < 8; i++) v |= ((long) (buf8[i] & 0xFF)) << (8 * i);
            return v;
        }

        float readFloat() throws IOException {
            return Float.intBitsToFloat(readInt());
        }

        double readDouble() throws IOException {
            return Double.longBitsToDouble(readLong());
        }

        boolean readBool() throws IOException {
            int b = in.read();
            if (b < 0) throw new EOFException();
            return b != 0;
        }

        /** Single unsigned byte (e.g. DirectMap type char). */
        int readRawByte() throws IOException {
            int b = in.read();
            if (b < 0) throw new EOFException();
            return b;
        }

        long readSize() throws IOException {
            return readLong();
        }

        int[] readIntArray(int n) throws IOException {
            byte[] raw = new byte[n * 4];
            readFully(raw, 0, raw.length);
            ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            int[] a = new int[n];
            for (int i = 0; i < n; i++) a[i] = bb.getInt();
            return a;
        }

        long[] readLongArray(int n) throws IOException {
            byte[] raw = new byte[Math.multiplyExact(n, 8)];
            readFully(raw, 0, raw.length);
            ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            long[] a = new long[n];
            for (int i = 0; i < n; i++) a[i] = bb.getLong();
            return a;
        }

        float[] readFloatArray(int n) throws IOException {
            byte[] raw = new byte[Math.multiplyExact(n, 4)];
            readFully(raw, 0, raw.length);
            ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            float[] a = new float[n];
            for (int i = 0; i < n; i++) a[i] = bb.getFloat();
            return a;
        }

        double[] readDoubleArray(int n) throws IOException {
            byte[] raw = new byte[Math.multiplyExact(n, 8)];
            readFully(raw, 0, raw.length);
            ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            double[] a = new double[n];
            for (int i = 0; i < n; i++) a[i] = bb.getDouble();
            return a;
        }

        int[] readVectorInt() throws IOException {
            long sz = readSize();
            checkSize(sz);
            return readIntArray((int) sz);
        }

        long[] readVectorLong() throws IOException {
            long sz = readSize();
            checkSize(sz);
            return readLongArray((int) sz);
        }

        float[] readVectorFloat() throws IOException {
            long sz = readSize();
            checkSize(sz);
            return readFloatArray((int) sz);
        }

        double[] readVectorDouble() throws IOException {
            long sz = readSize();
            checkSize(sz);
            return readDoubleArray((int) sz);
        }

        long[] readVectorSize() throws IOException {
            return readVectorLong();
        }

        /** READXBVECTOR: size is float-count; bytes = size*4. */
        byte[] readXbVector() throws IOException {
            long sz = readSize();
            checkSize(sz);
            long bytes = Math.multiplyExact(sz, 4L);
            if (bytes > Integer.MAX_VALUE) throw new IOException("XB too large: " + bytes);
            byte[] raw = new byte[(int) bytes];
            if (raw.length > 0) readFully(raw, 0, raw.length);
            return raw;
        }

        byte[] readBytes(int n) throws IOException {
            byte[] b = new byte[n];
            if (n > 0) readFully(b, 0, n);
            return b;
        }

        private static void checkSize(long sz) throws IOException {
            if (sz < 0 || sz >= (1L << 40))
                throw new IOException("invalid vector size: " + sz);
        }

        @Override public void close() throws IOException { in.close(); }
    }

    // =====================================================================
    // write_index dispatch
    // =====================================================================

    static void writeIndex(Index idx, FaissWriter w) throws IOException {
        if (idx == null) {
            w.writeInt(FCC_null);
            return;
        }
        if (idx instanceof IndexIDMap idm) {
            writeIDMap(idm, w);
        } else if (idx instanceof IndexHNSWFlat hnsw) {
            writeHNSWFlat(hnsw, w);
        } else if (idx instanceof IndexIVFPQ ivf) {
            writeIVFPQ(ivf, w);
        } else if (idx instanceof IndexFlat flat) {
            writeFlat(flat, w);
        } else {
            throw new IOException("Native FAISS write unsupported for " + idx.indexType()
                + " (" + idx.getClass().getName() + "). Use IndexIO JDF1 or convert to Flat/HNSW/IVFPQ/IDMap.");
        }
    }

    static void writeIndexHeader(Index idx, FaissWriter w) throws IOException {
        // d: int, ntotal: int64, dummy, dummy, is_trained: bool, metric_type: int
        // (+ metric_arg: float if metric_type > 1)
        w.writeInt(idx.d);
        w.writeLong(idx.ntotal());
        long dummy = 1L << 20;
        w.writeLong(dummy);
        w.writeLong(dummy);
        w.writeBool(idx.is_trained());
        int mt = idx.metric_type.code();
        w.writeInt(mt);
        if (mt > 1) {
            w.writeFloat(0f); // metric_arg
        }
    }

    static void writeFlat(IndexFlat flat, FaissWriter w) throws IOException {
        int h = flat.metric_type == MetricType.METRIC_INNER_PRODUCT ? FCC_IxFI
            : flat.metric_type == MetricType.METRIC_L2 ? FCC_IxF2
            : FCC_IxFl;
        w.writeInt(h);
        writeIndexHeader(flat, w);
        // codes as uint8 bytes of float32 xb[0 .. ntotal*d)
        int n = (int) flat.ntotal();
        float[] xb = flat.getXb();
        byte[] codes = new byte[n * flat.d * 4];
        ByteBuffer bb = ByteBuffer.wrap(codes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n * flat.d; i++) bb.putFloat(xb[i]);
        w.writeXbVector(codes);
    }

    static void writeIDMap(IndexIDMap idm, FaissWriter w) throws IOException {
        w.writeInt(FCC_IxMp);
        writeIndexHeader(idm, w);
        writeIndex(idm.index, w);
        long[] ids = idm.id_map();
        w.writeVectorLong(ids);
    }

    static void writeHNSWFlat(IndexHNSWFlat idx, FaissWriter w) throws IOException {
        w.writeInt(FCC_IHNf);
        writeIndexHeader(idx, w);
        writeHnswGraph(idx, w);
        // storage = IndexFlat with same metric + vectors
        IndexFlat storage = new IndexFlat(idx.d, idx.metric_type);
        float[] data = idx.storageData();
        int n = (int) idx.ntotal();
        if (n > 0) storage.add(data, n);
        writeFlat(storage, w);
    }

    /**
     * Serialize HNSW link structure in FAISS layout:
     * assign_probas, cum_nneighbor_per_level, levels(+1), offsets, neighbors,
     * entry_point, max_level, efConstruction, efSearch, upper_beam(=1).
     */
    static void writeHnswGraph(IndexHNSWFlat idx, FaissWriter w) throws IOException {
        int M = idx.M();
        int n = (int) idx.ntotal();
        double levelMult = 1.0 / Math.log(M);

        // assign_probas + cum_nneighbor_per_level (set_default_probas)
        List<Double> assignProbas = new ArrayList<>();
        List<Integer> cumNN = new ArrayList<>();
        cumNN.add(0);
        int nn = 0;
        for (int level = 0; ; level++) {
            double proba = Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult));
            if (proba < 1e-9) break;
            assignProbas.add(proba);
            nn += (level == 0) ? M * 2 : M;
            cumNN.add(nn);
        }
        double[] ap = new double[assignProbas.size()];
        for (int i = 0; i < ap.length; i++) ap[i] = assignProbas.get(i);
        w.writeVectorDouble(ap);
        int[] cum = new int[cumNN.size()];
        for (int i = 0; i < cum.length; i++) cum[i] = cumNN.get(i);
        w.writeVectorInt(cum);

        // levels: FAISS stores level+1
        int[] levelsOut = new int[n];
        for (int i = 0; i < n; i++) levelsOut[i] = idx.levelOf(i) + 1;
        w.writeVectorInt(levelsOut);

        // offsets + neighbors (fixed slots per level, -1 pad)
        long[] offsets = new long[n + 1];
        offsets[0] = 0;
        for (int i = 0; i < n; i++) {
            int ptLevel = levelsOut[i] - 1; // 0-based max level of node
            // cum_nb_neighbors(pt_level+1) = cum[ptLevel+1]
            int slots = cumNbNeighbors(cum, ptLevel + 1);
            offsets[i + 1] = offsets[i] + slots;
        }
        int totalNeighbors = (int) offsets[n];
        int[] neighbors = new int[totalNeighbors];
        Arrays.fill(neighbors, -1);
        for (int i = 0; i < n; i++) {
            int ptLevel = levelsOut[i] - 1;
            long base = offsets[i];
            for (int lc = 0; lc <= ptLevel; lc++) {
                int[] nbs = idx.neighborsOf(i, lc);
                int begin = (int) (base + cumNbNeighbors(cum, lc));
                int maxM = nbNeighbors(cum, lc);
                int copy = nbs == null ? 0 : Math.min(nbs.length, maxM);
                for (int j = 0; j < copy; j++) neighbors[begin + j] = nbs[j];
                // rest stay -1
            }
        }
        w.writeVectorSize(offsets);
        w.writeVectorInt(neighbors);

        w.writeInt(idx.entryPoint());
        w.writeInt(idx.maxLevel());
        w.writeInt(idx.hnsw.efConstruction);
        w.writeInt(idx.hnsw.efSearch);
        w.writeInt(1); // deprecated upper_beam
    }

    private static int cumNbNeighbors(int[] cum, int layerNo) {
        if (layerNo < 0) return 0;
        if (layerNo >= cum.length) return cum[cum.length - 1];
        return cum[layerNo];
    }

    private static int nbNeighbors(int[] cum, int layerNo) {
        return cumNbNeighbors(cum, layerNo + 1) - cumNbNeighbors(cum, layerNo);
    }

    static void writeIVFPQ(IndexIVFPQ ivf, FaissWriter w) throws IOException {
        w.writeInt(FCC_IwPQ);
        writeIndexHeader(ivf, w);
        w.writeLong(ivf.nlist);                    // size_t nlist
        w.writeLong(Math.max(1, ivf.nprobe));      // size_t nprobe
        writeIndex(ivf.quantizer, w);
        // direct_map: type char (0 = NoMap) + empty array vector
        w.writeRawByte(0);
        w.writeSize(0);
        // by_residual — our IVFPQ always encodes residuals
        w.writeBool(true);
        int codeSize = ivf.pqCodeSize();
        w.writeLong(codeSize);                     // size_t code_size
        writeProductQuantizer(ivf, w);
        writeInvertedLists(ivf, w);
    }

    static void writeProductQuantizer(IndexIVFPQ ivf, FaissWriter w) throws IOException {
        // FAISS ProductQuantizer: size_t d, M, nbits + float centroids vector
        w.writeLong(ivf.d);
        w.writeLong(ivf.m);
        w.writeLong(ivf.nbits);
        float[] flat = ivf.flatCodebooks();
        w.writeVectorFloat(flat);
    }

    static void writeInvertedLists(IndexIVFPQ ivf, FaissWriter w) throws IOException {
        w.writeInt(FCC_ilar);
        w.writeLong(ivf.nlist);
        int codeSize = ivf.pqCodeSize();
        w.writeLong(codeSize);
        // decide full vs sparse
        int nNon0 = 0;
        for (int i = 0; i < ivf.nlist; i++) {
            if (ivf.listSize(i) > 0) nNon0++;
        }
        if (nNon0 > ivf.nlist / 2) {
            w.writeInt(FCC_full);
            long[] sizes = new long[ivf.nlist];
            for (int i = 0; i < ivf.nlist; i++) sizes[i] = ivf.listSize(i);
            w.writeVectorLong(sizes);
        } else {
            w.writeInt(FCC_sprs);
            // pairs (list_id, size) for non-empty
            int pairs = nNon0 * 2;
            long[] sizes = new long[pairs];
            int p = 0;
            for (int i = 0; i < ivf.nlist; i++) {
                int n = ivf.listSize(i);
                if (n > 0) {
                    sizes[p++] = i;
                    sizes[p++] = n;
                }
            }
            w.writeVectorLong(sizes);
        }
        // contiguous codes + ids per non-empty list
        for (int i = 0; i < ivf.nlist; i++) {
            int n = ivf.listSize(i);
            if (n <= 0) continue;
            byte[] codes = ivf.listCodes(i); // n * codeSize
            long[] ids = ivf.listIds(i);
            w.writeBytes(codes, 0, codes.length);
            w.writeLongArray(ids, ids.length);
        }
    }

    // =====================================================================
    // read_index dispatch
    // =====================================================================

    static Index readIndex(FaissReader r) throws IOException {
        int h = r.readInt();
        return readIndexWithFourcc(h, r);
    }

    static Index readIndexWithFourcc(int h, FaissReader r) throws IOException {
        if (h == FCC_null) {
            return null;
        } else if (h == FCC_IxF2 || h == FCC_IxFI || h == FCC_IxFl) {
            return readFlat(h, r);
        } else if (h == FCC_IHNf) {
            return readHNSWFlat(r);
        } else if (h == FCC_IxMp || h == FCC_IxM2) {
            return readIDMap(h, r);
        } else if (h == FCC_IwPQ) {
            return readIVFPQ(r);
        } else {
            throw new IOException("Unsupported FAISS fourcc '" + fourccString(h)
                + "' (0x" + Integer.toHexString(h) + "). "
                + "Supported: IxF2/IxFI/IHNf/IxMp/IwPQ");
        }
    }

    static final class Header {
        int d;
        long ntotal;
        boolean is_trained;
        MetricType metric;
    }

    static Header readIndexHeader(FaissReader r) throws IOException {
        Header h = new Header();
        h.d = r.readInt();
        h.ntotal = r.readLong();
        r.readLong(); // dummy
        r.readLong(); // dummy
        h.is_trained = r.readBool();
        int mt = r.readInt();
        h.metric = MetricType.fromCode(mt);
        if (mt > 1) {
            r.readFloat(); // metric_arg
        }
        return h;
    }

    static IndexFlat readFlat(int fourcc, FaissReader r) throws IOException {
        Header h = readIndexHeader(r);
        MetricType metric = (fourcc == FCC_IxFI) ? MetricType.METRIC_INNER_PRODUCT
            : (fourcc == FCC_IxF2) ? MetricType.METRIC_L2
            : h.metric;
        IndexFlat flat = metric == MetricType.METRIC_INNER_PRODUCT
            ? new IndexFlatIP(h.d) : new IndexFlatL2(h.d);
        // override if generic
        flat.metric_type = metric;
        byte[] codes = r.readXbVector();
        int n = (int) h.ntotal;
        if (n > 0) {
            if (codes.length < n * h.d * 4)
                throw new IOException("Flat codes too small: " + codes.length
                    + " need " + (n * h.d * 4));
            float[] xb = new float[n * h.d];
            ByteBuffer bb = ByteBuffer.wrap(codes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < xb.length; i++) xb[i] = bb.getFloat();
            flat.add(xb, n);
        }
        // trust header ntotal
        if (flat.ntotal() != h.ntotal) {
            // add() sets ntotal; OK
        }
        return flat;
    }

    static IndexIDMap readIDMap(int fourcc, FaissReader r) throws IOException {
        Header h = readIndexHeader(r);
        Index inner = readIndex(r);
        if (inner == null) throw new IOException("IndexIDMap with null inner");
        long[] idMap = r.readVectorLong();
        IndexIDMap map = IndexIDMap.wrapExisting(inner, idMap);
        // header ntotal should match
        return map;
    }

    static IndexHNSWFlat readHNSWFlat(FaissReader r) throws IOException {
        Header h = readIndexHeader(r);
        HnswGraphData g = readHnswGraph(r);
        Index storage = readIndex(r);
        if (!(storage instanceof IndexFlat flatStorage)) {
            throw new IOException("IHNf storage must be IndexFlat, got "
                + (storage == null ? "null" : storage.indexType()));
        }
        int M = inferM(g);
        IndexHNSWFlat idx = new IndexHNSWFlat(h.d, Math.max(2, M), h.metric);
        idx.hnsw.efConstruction = g.efConstruction;
        idx.hnsw.efSearch = g.efSearch;
        // load vectors + graph
        float[] xb = Arrays.copyOf(flatStorage.getXb(), (int) flatStorage.ntotal() * h.d);
        idx.loadFromFaiss(xb, (int) flatStorage.ntotal(), g);
        return idx;
    }

    static final class HnswGraphData {
        double[] assignProbas;
        int[] cumNNeighbor;
        int[] levels;       // FAISS levels (already +1)
        long[] offsets;
        int[] neighbors;
        int entryPoint;
        int maxLevel;
        int efConstruction;
        int efSearch;
    }

    static HnswGraphData readHnswGraph(FaissReader r) throws IOException {
        HnswGraphData g = new HnswGraphData();
        g.assignProbas = r.readVectorDouble();
        g.cumNNeighbor = r.readVectorInt();
        g.levels = r.readVectorInt();
        g.offsets = r.readVectorSize();
        g.neighbors = r.readVectorInt();
        g.entryPoint = r.readInt();
        g.maxLevel = r.readInt();
        g.efConstruction = r.readInt();
        g.efSearch = r.readInt();
        r.readInt(); // upper_beam discarded
        return g;
    }

    static int inferM(HnswGraphData g) {
        // level 0 has 2*M, level>0 has M
        if (g.cumNNeighbor != null && g.cumNNeighbor.length >= 2) {
            int m0 = g.cumNNeighbor[1] - g.cumNNeighbor[0];
            return Math.max(2, m0 / 2);
        }
        return 32;
    }

    static IndexIVFPQ readIVFPQ(FaissReader r) throws IOException {
        Header h = readIndexHeader(r);
        long nlist = r.readLong();
        long nprobe = r.readLong();
        Index quantizer = readIndex(r);
        if (quantizer == null) {
            // create empty flat quantizer; will be filled from centroids via lists?
            quantizer = h.metric == MetricType.METRIC_INNER_PRODUCT
                ? new IndexFlatIP(h.d) : new IndexFlatL2(h.d);
        }
        // direct_map
        int dmType = r.readRawByte(); // char
        long[] dmArray = r.readVectorLong();
        if (dmType == 2) { // Hashtable
            // vector of pair<idx_t,idx_t>
            long npairs = r.readSize();
            if (npairs < 0 || npairs > (1L << 30)) throw new IOException("bad dm hashtable");
            // each pair = 16 bytes
            r.readBytes(Math.toIntExact(npairs * 16));
        }
        boolean byResidual = r.readBool();
        long codeSize = r.readLong();
        // ProductQuantizer: size_t d, M, nbits
        int pqD = (int) r.readLong();
        int pqM = (int) r.readLong();
        int pqNbits = (int) r.readLong();
        float[] centroidsFlat = r.readVectorFloat();
        if (pqNbits > 8)
            throw new IOException("IVFPQ nbits>8 not supported in pure-Java reader");
        IndexIVFPQ ivf = new IndexIVFPQ(quantizer, h.d, (int) nlist, pqM, pqNbits);
        ivf.metric_type = h.metric;
        ivf.nprobe = (int) Math.max(1, nprobe);
        ivf.loadPqCodebooks(centroidsFlat, pqD, pqM, pqNbits);
        // inverted lists
        readInvertedListsInto(ivf, r, (int) codeSize);
        ivf.setTrained(true);
        // ntotal from lists
        ivf.recomputeNtotalFromLists();
        return ivf;
    }

    static void readInvertedListsInto(IndexIVFPQ ivf, FaissReader r, int codeSize) throws IOException {
        int h = r.readInt();
        if (h == FCC_il00) return;
        if (h != FCC_ilar)
            throw new IOException("Expected ilar inverted lists, got " + fourccString(h));
        long nlist = r.readLong();
        long cs = r.readLong();
        if ((int) cs != codeSize && codeSize > 0) {
            // prefer file
            codeSize = (int) cs;
        }
        int listType = r.readInt();
        long[] sizes;
        boolean full = listType == FCC_full;
        if (full) {
            sizes = r.readVectorLong();
            if (sizes.length != nlist)
                throw new IOException("full sizes len mismatch");
        } else if (listType == FCC_sprs) {
            long[] pairs = r.readVectorLong();
            sizes = new long[(int) nlist];
            for (int i = 0; i + 1 < pairs.length; i += 2) {
                int li = (int) pairs[i];
                long n = pairs[i + 1];
                if (li >= 0 && li < sizes.length) sizes[li] = n;
            }
        } else {
            throw new IOException("Unknown invlist type " + fourccString(listType));
        }
        for (int i = 0; i < (int) nlist; i++) {
            int n = (int) sizes[i];
            if (n <= 0) continue;
            byte[] codes = r.readBytes(n * codeSize);
            long[] ids = r.readLongArray(n);
            ivf.loadList(i, ids, codes, codeSize);
        }
    }
}
