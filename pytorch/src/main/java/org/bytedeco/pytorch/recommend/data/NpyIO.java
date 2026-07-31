/*
 * Minimal NumPy .npy reader (v1.0 / v2.0) for float32 / int64 / int32 arrays.
 * Used by generative SID pipelines (MicroLens item_emb_d128.npy, item_seq.npy, ...).
 *
 * No numpy/JNI dependency — pure Java little-endian parser.
 */
package org.bytedeco.pytorch.recommend.data;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class NpyIO {

    private NpyIO() {}

    public static final class Array {
        public final long[] shape;
        public final String dtype; // e.g. "<f4", "<i8"
        public final ByteBuffer data; // little-endian, positioned at payload

        public Array(long[] shape, String dtype, ByteBuffer data) {
            this.shape = shape;
            this.dtype = dtype;
            this.data = data;
        }

        public int rank() { return shape.length; }
        public long size() {
            long n = 1;
            for (long s : shape) n *= s;
            return n;
        }
    }

    public static Array load(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = ch.size();
            if (fileSize > Integer.MAX_VALUE) {
                throw new IOException("npy too large to map: " + fileSize);
            }
            ByteBuffer buf = ch.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            buf.order(ByteOrder.LITTLE_ENDIAN);

            // magic \x93NUMPY
            if (buf.remaining() < 10) throw new IOException("truncated npy");
            byte m0 = buf.get(), m1 = buf.get(), m2 = buf.get(), m3 = buf.get();
            byte m4 = buf.get(), m5 = buf.get();
            if (m0 != (byte) 0x93 || m1 != 'N' || m2 != 'U' || m3 != 'M' || m4 != 'P' || m5 != 'Y') {
                throw new IOException("not a .npy file: " + path);
            }
            int major = buf.get() & 0xff;
            int minor = buf.get() & 0xff;
            int headerLen;
            if (major == 1) {
                headerLen = buf.getShort() & 0xffff;
            } else if (major == 2 || major == 3) {
                headerLen = buf.getInt();
            } else {
                throw new IOException("unsupported npy version " + major + "." + minor);
            }
            byte[] headerBytes = new byte[headerLen];
            buf.get(headerBytes);
            String header = new String(headerBytes, java.nio.charset.StandardCharsets.US_ASCII);

            String descr = match(header, "'descr'\\s*:\\s*'([^']+)'");
            String shapeStr = match(header, "'shape'\\s*:\\s*\\(([^)]*)\\)");
            boolean fortran = header.contains("'fortran_order': True")
                    || header.contains("'fortran_order':True");
            if (fortran) {
                throw new IOException("fortran_order npy not supported: " + path);
            }
            long[] shape = parseShape(shapeStr);
            ByteBuffer data = buf.slice().order(ByteOrder.LITTLE_ENDIAN);
            return new Array(shape, descr, data);
        }
    }

    public static float[] loadFloat32Flat(Path path) throws IOException {
        Array a = load(path);
        if (!a.dtype.contains("f4") && !a.dtype.contains("float32")) {
            // allow f8 downcast
            if (a.dtype.contains("f8")) {
                long n = a.size();
                float[] out = new float[(int) n];
                DoubleBuffer db = a.data.asDoubleBuffer();
                for (int i = 0; i < n; i++) out[i] = (float) db.get(i);
                return out;
            }
            throw new IOException("expected float32, got " + a.dtype + " @ " + path);
        }
        long n = a.size();
        float[] out = new float[(int) n];
        a.data.asFloatBuffer().get(out);
        return out;
    }

    public static float[][] loadFloat32Matrix(Path path) throws IOException {
        Array a = load(path);
        if (a.shape.length != 2) throw new IOException("expected rank-2, got " + a.shape.length);
        int rows = (int) a.shape[0];
        int cols = (int) a.shape[1];
        float[] flat = loadFloat32Flat(path);
        float[][] mat = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, mat[i], 0, cols);
        }
        return mat;
    }

    public static long[] loadInt64Flat(Path path) throws IOException {
        Array a = load(path);
        long n = a.size();
        long[] out = new long[(int) n];
        if (a.dtype.contains("i8") || a.dtype.contains("int64")) {
            a.data.asLongBuffer().get(out);
        } else if (a.dtype.contains("i4") || a.dtype.contains("int32")) {
            IntBuffer ib = a.data.asIntBuffer();
            for (int i = 0; i < n; i++) out[i] = ib.get(i);
        } else if (a.dtype.contains("f4")) {
            FloatBuffer fb = a.data.asFloatBuffer();
            for (int i = 0; i < n; i++) out[i] = (long) fb.get(i);
        } else {
            throw new IOException("unsupported int dtype " + a.dtype + " @ " + path);
        }
        return out;
    }

    public static long[][] loadInt64Matrix(Path path) throws IOException {
        Array a = load(path);
        if (a.shape.length == 1) {
            long[] flat = loadInt64Flat(path);
            long[][] m = new long[flat.length][1];
            for (int i = 0; i < flat.length; i++) m[i][0] = flat[i];
            return m;
        }
        if (a.shape.length != 2) throw new IOException("expected rank-1/2 int array");
        int rows = (int) a.shape[0];
        int cols = (int) a.shape[1];
        long[] flat = loadInt64Flat(path);
        long[][] mat = new long[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, mat[i], 0, cols);
        }
        return mat;
    }

    private static String match(String header, String regex) throws IOException {
        Matcher m = Pattern.compile(regex).matcher(header);
        if (!m.find()) throw new IOException("header field missing: " + regex + " in " + header);
        return m.group(1);
    }

    private static long[] parseShape(String shapeStr) {
        shapeStr = shapeStr.trim();
        if (shapeStr.isEmpty()) return new long[0];
        // "91718, 128" or "91718," or "64,"
        String[] parts = shapeStr.split(",");
        java.util.List<Long> dims = new java.util.ArrayList<>();
        for (String p : parts) {
            p = p.trim();
            if (p.isEmpty()) continue;
            dims.add(Long.parseLong(p));
        }
        long[] shape = new long[dims.size()];
        for (int i = 0; i < dims.size(); i++) shape[i] = dims.get(i);
        return shape;
    }
}
