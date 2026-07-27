package org.bytedeco.pytorch.data.dataframe.hdf5.internal;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal pure-Java HDF5-family container used by DataFrame I/O.
 *
 * <p>File starts with the standard HDF5 signature {@code \\x89HDF\\r\\n\\x1a\\n} so
 * format sniffers recognize it. Body uses a compact LE layout owned by this library
 * (columnar / matrix DataFrame round-trip). Not a full HDF5 stack: no filters,
 * chunks, soft links, or external storage.
 *
 * <pre>
 * magic(8) | ver(u32)=1 | flags(u32)=0 | root_offset(u64)
 * Group  'GRP\\0' | attr_count(u32) | child_count(u32) | attrs... | children...
 *   attr: name_len(u16) name utf8 | type(u8) | payload
 *     type 1 string: len(u32) bytes
 *     type 2 string[]: n(u32) {len(u32) bytes}*
 *     type 3 int64: value(u64)
 *   child: name_len(u16) name | kind(u8 0=group 1=dataset) | offset(u64)
 * Dataset 'DSET' | dtype(u8) | rank(u8) | dim0(u64) | dim1(u64) | nbytes(u64) | raw
 *   dtype: 1=i32 2=i64 3=f32 4=f64 5=bool 6=fixed utf8 strings
 * </pre>
 */
public final class Hdf5WriterCore {
    public static final byte[] SIGNATURE = new byte[]{(byte) 0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'};
    public static final int VERSION = 1;
    private static final int HEADER_SIZE = 8 + 4 + 4 + 8;

    private Hdf5WriterCore() {}

    public static void write(Path path, String key, Map<String, Object> attrs,
                             LinkedHashMap<String, EncodedData> columns) throws IOException {
        if (Files.exists(path)) Files.delete(path);
        Path parent = path.getParent();
        if (parent != null) Files.createDirectories(parent);

        Grow body = new Grow();
        body.write(new byte[HEADER_SIZE]);

        LinkedHashMap<String, Long> childOff = new LinkedHashMap<>();
        for (Map.Entry<String, EncodedData> e : columns.entrySet()) {
            long off = body.size();
            writeDatasetRecord(body, e.getValue());
            childOff.put(e.getKey(), off);
        }

        String[] parts = splitKey(key);
        List<ChildRef> children = new ArrayList<>();
        for (Map.Entry<String, Long> e : childOff.entrySet()) {
            children.add(new ChildRef(e.getKey(), (byte) 1, e.getValue()));
        }

        long off;
        if (parts.length == 0) {
            off = writeGroupRecord(body, attrs, children);
        } else {
            // innermost group holds datasets + attrs
            off = writeGroupRecord(body, attrs, children);
            for (int i = parts.length - 1; i >= 0; i--) {
                List<ChildRef> ch = new ArrayList<>();
                ch.add(new ChildRef(parts[i], (byte) 0, off));
                off = writeGroupRecord(body, Map.of(), ch);
            }
        }

        byte[] file = body.bytes();
        ByteBuffer hdr = ByteBuffer.wrap(file, 0, HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
        hdr.put(SIGNATURE);
        hdr.putInt(VERSION);
        hdr.putInt(0);
        hdr.putLong(off);
        Files.write(path, java.util.Arrays.copyOf(file, body.size()));
    }

    private static String[] splitKey(String key) {
        if (key == null || key.isEmpty() || "/".equals(key)) return new String[0];
        String p = key.startsWith("/") ? key.substring(1) : key;
        return java.util.Arrays.stream(p.split("/")).filter(s -> !s.isEmpty()).toArray(String[]::new);
    }

    private static long writeGroupRecord(Grow body, Map<String, Object> attrs, List<ChildRef> children) {
        long off = body.size();
        Grow g = new Grow();
        g.write(new byte[]{'G', 'R', 'P', 0});
        List<Map.Entry<String, Object>> al = new ArrayList<>(attrs.entrySet());
        g.writeInt(al.size());
        g.writeInt(children.size());
        for (Map.Entry<String, Object> a : al) writeAttr(g, a.getKey(), a.getValue());
        for (ChildRef c : children) {
            byte[] nb = c.name.getBytes(StandardCharsets.UTF_8);
            g.writeShort(nb.length);
            g.write(nb);
            g.write(new byte[]{c.kind});
            g.writeLong(c.offset);
        }
        body.write(g.bytes(), 0, g.size());
        return off;
    }

    private static void writeDatasetRecord(Grow body, EncodedData enc) {
        Grow g = new Grow();
        g.write(new byte[]{'D', 'S', 'E', 'T'});
        g.write(new byte[]{enc.dtypeCode});
        g.write(new byte[]{(byte) enc.rank});
        g.writeLong(enc.dim0);
        g.writeLong(enc.dim1);
        g.writeLong(enc.raw.length);
        g.write(enc.raw);
        body.write(g.bytes(), 0, g.size());
    }

    private static void writeAttr(Grow g, String name, Object value) {
        byte[] nb = name.getBytes(StandardCharsets.UTF_8);
        g.writeShort(nb.length);
        g.write(nb);
        if (value instanceof String) {
            g.write(new byte[]{1});
            byte[] vb = ((String) value).getBytes(StandardCharsets.UTF_8);
            g.writeInt(vb.length);
            g.write(vb);
        } else if (value instanceof String[]) {
            g.write(new byte[]{2});
            String[] arr = (String[]) value;
            g.writeInt(arr.length);
            for (String s : arr) {
                byte[] vb = (s == null ? "" : s).getBytes(StandardCharsets.UTF_8);
                g.writeInt(vb.length);
                g.write(vb);
            }
        } else if (value instanceof Number) {
            g.write(new byte[]{3});
            g.writeLong(((Number) value).longValue());
        } else {
            g.write(new byte[]{1});
            byte[] vb = String.valueOf(value).getBytes(StandardCharsets.UTF_8);
            g.writeInt(vb.length);
            g.write(vb);
        }
    }

    public static EncodedData encodeData(Object data) {
        if (data instanceof int[]) {
            int[] a = (int[]) data;
            ByteBuffer bb = ByteBuffer.allocate(a.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (int v : a) bb.putInt(v);
            return new EncodedData((byte) 1, 1, a.length, 0, bb.array());
        }
        if (data instanceof long[]) {
            long[] a = (long[]) data;
            ByteBuffer bb = ByteBuffer.allocate(a.length * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (long v : a) bb.putLong(v);
            return new EncodedData((byte) 2, 1, a.length, 0, bb.array());
        }
        if (data instanceof float[]) {
            float[] a = (float[]) data;
            ByteBuffer bb = ByteBuffer.allocate(a.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            for (float v : a) bb.putFloat(v);
            return new EncodedData((byte) 3, 1, a.length, 0, bb.array());
        }
        if (data instanceof double[]) {
            double[] a = (double[]) data;
            ByteBuffer bb = ByteBuffer.allocate(a.length * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (double v : a) bb.putDouble(v);
            return new EncodedData((byte) 4, 1, a.length, 0, bb.array());
        }
        if (data instanceof boolean[]) {
            boolean[] a = (boolean[]) data;
            byte[] raw = new byte[a.length];
            for (int i = 0; i < a.length; i++) raw[i] = (byte) (a[i] ? 1 : 0);
            return new EncodedData((byte) 5, 1, a.length, 0, raw);
        }
        if (data instanceof double[][]) {
            double[][] m = (double[][]) data;
            int rows = m.length;
            int cols = rows == 0 ? 0 : m[0].length;
            ByteBuffer bb = ByteBuffer.allocate(Math.max(0, rows * cols) * 8).order(ByteOrder.LITTLE_ENDIAN);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++) bb.putDouble(m[r][c]);
            return new EncodedData((byte) 4, 2, rows, cols, bb.array());
        }
        if (data instanceof String[]) {
            String[] a = (String[]) data;
            int max = 1;
            byte[][] encoded = new byte[a.length][];
            for (int i = 0; i < a.length; i++) {
                encoded[i] = (a[i] == null ? "" : a[i]).getBytes(StandardCharsets.UTF_8);
                max = Math.max(max, encoded[i].length);
            }
            ByteBuffer bb = ByteBuffer.allocate(4 + a.length * max).order(ByteOrder.LITTLE_ENDIAN);
            bb.putInt(max);
            for (byte[] e : encoded) {
                bb.put(e);
                for (int p = e.length; p < max; p++) bb.put((byte) 0);
            }
            return new EncodedData((byte) 6, 1, a.length, max, bb.array());
        }
        return new EncodedData((byte) 2, 1, 0, 0, new byte[0]);
    }

    public static String sanitize(String name) {
        if (name == null || name.isEmpty()) return "col";
        return name.replace('/', '_').replace('\0', '_');
    }

    public static final class EncodedData {
        public final byte dtypeCode;
        public final int rank;
        public final long dim0, dim1;
        public final byte[] raw;

        public EncodedData(byte dtypeCode, int rank, long dim0, long dim1, byte[] raw) {
            this.dtypeCode = dtypeCode;
            this.rank = rank;
            this.dim0 = dim0;
            this.dim1 = dim1;
            this.raw = raw == null ? new byte[0] : raw;
        }
    }

    private static final class ChildRef {
        final String name;
        final byte kind;
        final long offset;
        ChildRef(String name, byte kind, long offset) {
            this.name = name;
            this.kind = kind;
            this.offset = offset;
        }
    }

    public static final class Grow {
        private byte[] buf = new byte[256];
        private int size;

        public int size() { return size; }
        public byte[] bytes() { return buf; }

        public void write(byte[] b) { write(b, 0, b.length); }

        public void write(byte[] b, int off, int len) {
            ensure(size + len);
            System.arraycopy(b, off, buf, size, len);
            size += len;
        }

        public void writeInt(int v) {
            ensure(size + 4);
            ByteBuffer.wrap(buf, size, 4).order(ByteOrder.LITTLE_ENDIAN).putInt(v);
            size += 4;
        }

        public void writeShort(int v) {
            ensure(size + 2);
            ByteBuffer.wrap(buf, size, 2).order(ByteOrder.LITTLE_ENDIAN).putShort((short) v);
            size += 2;
        }

        public void writeLong(long v) {
            ensure(size + 8);
            ByteBuffer.wrap(buf, size, 8).order(ByteOrder.LITTLE_ENDIAN).putLong(v);
            size += 8;
        }

        private void ensure(int n) {
            if (n <= buf.length) return;
            int c = buf.length;
            while (c < n) c *= 2;
            buf = java.util.Arrays.copyOf(buf, c);
        }
    }
}
