package org.bytedeco.pytorch.dataframe.hdf5.internal;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Reader for the minimal HDF5-family layout produced by {@link Hdf5WriterCore}.
 */
public final class Hdf5ReaderCore {
    private Hdf5ReaderCore() {}

    public static final class Node {
        public final boolean group;
        public final Map<String, Object> attrs;
        public final Map<String, Node> children; // groups only
        public final Hdf5WriterCore.EncodedData dataset; // datasets only
        public final String name;

        private Node(String name, boolean group, Map<String, Object> attrs,
                     Map<String, Node> children, Hdf5WriterCore.EncodedData dataset) {
            this.name = name;
            this.group = group;
            this.attrs = attrs;
            this.children = children;
            this.dataset = dataset;
        }

        static Node group(String name, Map<String, Object> attrs, Map<String, Node> children) {
            return new Node(name, true, attrs, children, null);
        }

        static Node dataset(String name, Hdf5WriterCore.EncodedData data) {
            return new Node(name, false, Map.of(), Map.of(), data);
        }
    }

    public static Node open(Path path) throws IOException {
        byte[] file = Files.readAllBytes(path);
        if (file.length < 24) throw new IOException("File too small for HDF5");
        for (int i = 0; i < 8; i++) {
            if (file[i] != Hdf5WriterCore.SIGNATURE[i]) {
                throw new IOException("Not an HDF5 file (bad signature)");
            }
        }
        ByteBuffer bb = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        bb.position(8);
        int ver = bb.getInt();
        bb.getInt(); // flags
        long rootOff = bb.getLong();
        if (ver != Hdf5WriterCore.VERSION) {
            throw new IOException("Unsupported HDF5-family version: " + ver
                + " (only library columnar v" + Hdf5WriterCore.VERSION + ")");
        }
        if (rootOff < 0 || rootOff >= file.length) {
            throw new IOException("Invalid root offset: " + rootOff);
        }
        return readNode(file, rootOff, "/");
    }

    public static Node resolve(Node root, String key) {
        if (key == null || key.isEmpty() || "/".equals(key)) return root;
        String p = key.startsWith("/") ? key.substring(1) : key;
        Node cur = root;
        for (String part : p.split("/")) {
            if (part.isEmpty()) continue;
            if (!cur.group) return null;
            Node next = cur.children.get(part);
            if (next == null) return null;
            cur = next;
        }
        return cur;
    }

    private static Node readNode(byte[] file, long offset, String name) throws IOException {
        if (offset < 0 || offset + 4 > file.length) {
            throw new IOException("Bad node offset: " + offset);
        }
        int o = (int) offset;
        if (file[o] == 'G' && file[o + 1] == 'R' && file[o + 2] == 'P') {
            return readGroup(file, o, name);
        }
        if (file[o] == 'D' && file[o + 1] == 'S' && file[o + 2] == 'E' && file[o + 3] == 'T') {
            return Node.dataset(name, readDataset(file, o));
        }
        throw new IOException("Unknown node magic at " + offset);
    }

    private static Node readGroup(byte[] file, int o, String name) throws IOException {
        ByteBuffer bb = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        bb.position(o + 4);
        int attrCount = bb.getInt();
        int childCount = bb.getInt();
        Map<String, Object> attrs = new LinkedHashMap<>();
        for (int i = 0; i < attrCount; i++) {
            String an = readString(bb);
            int type = bb.get() & 0xFF;
            if (type == 1) {
                int len = bb.getInt();
                byte[] raw = new byte[len];
                bb.get(raw);
                attrs.put(an, new String(raw, StandardCharsets.UTF_8));
            } else if (type == 2) {
                int n = bb.getInt();
                String[] arr = new String[n];
                for (int j = 0; j < n; j++) {
                    int len = bb.getInt();
                    byte[] raw = new byte[len];
                    bb.get(raw);
                    arr[j] = new String(raw, StandardCharsets.UTF_8);
                }
                attrs.put(an, arr);
            } else if (type == 3) {
                attrs.put(an, bb.getLong());
            } else {
                throw new IOException("Unknown attr type " + type + " for " + an);
            }
        }
        Map<String, Node> children = new LinkedHashMap<>();
        for (int i = 0; i < childCount; i++) {
            String cn = readString(bb);
            int kind = bb.get() & 0xFF;
            long off = bb.getLong();
            children.put(cn, readNode(file, off, cn));
            // kind 0/1 informational; structure is recursive
            if (kind != 0 && kind != 1) {
                // ignore
            }
        }
        return Node.group(name, attrs, children);
    }

    private static Hdf5WriterCore.EncodedData readDataset(byte[] file, int o) {
        ByteBuffer bb = ByteBuffer.wrap(file).order(ByteOrder.LITTLE_ENDIAN);
        bb.position(o + 4);
        byte dtype = bb.get();
        byte rank = bb.get();
        long dim0 = bb.getLong();
        long dim1 = bb.getLong();
        long nbytes = bb.getLong();
        byte[] raw = new byte[(int) nbytes];
        bb.get(raw);
        return new Hdf5WriterCore.EncodedData(dtype, rank, dim0, dim1, raw);
    }

    private static String readString(ByteBuffer bb) {
        int len = bb.getShort() & 0xFFFF;
        byte[] raw = new byte[len];
        bb.get(raw);
        return new String(raw, StandardCharsets.UTF_8);
    }

    /** Decode dataset raw payload to Object[] (1-D) or leave matrix as double[][]. */
    public static Object decodeToJava(Hdf5WriterCore.EncodedData enc) {
        ByteBuffer bb = ByteBuffer.wrap(enc.raw).order(ByteOrder.LITTLE_ENDIAN);
        int n = (int) enc.dim0;
        switch (enc.dtypeCode) {
            case 1: {
                int[] a = new int[n];
                for (int i = 0; i < n; i++) a[i] = bb.getInt();
                return a;
            }
            case 2: {
                long[] a = new long[n];
                for (int i = 0; i < n; i++) a[i] = bb.getLong();
                return a;
            }
            case 3: {
                float[] a = new float[n];
                for (int i = 0; i < n; i++) a[i] = bb.getFloat();
                return a;
            }
            case 4: {
                if (enc.rank == 2) {
                    int rows = (int) enc.dim0;
                    int cols = (int) enc.dim1;
                    double[][] m = new double[rows][cols];
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++) m[r][c] = bb.getDouble();
                    return m;
                }
                double[] a = new double[n];
                for (int i = 0; i < n; i++) a[i] = bb.getDouble();
                return a;
            }
            case 5: {
                boolean[] a = new boolean[n];
                for (int i = 0; i < n; i++) a[i] = bb.get() != 0;
                return a;
            }
            case 6: {
                int max = bb.getInt();
                String[] a = new String[n];
                byte[] tmp = new byte[max];
                for (int i = 0; i < n; i++) {
                    bb.get(tmp);
                    int end = max;
                    while (end > 0 && tmp[end - 1] == 0) end--;
                    a[i] = new String(tmp, 0, end, StandardCharsets.UTF_8);
                }
                return a;
            }
            default:
                return new Object[0];
        }
    }

    public static Object[] toObjectArray(Object data) {
        if (data == null) return new Object[0];
        if (data instanceof Object[]) return (Object[]) data;
        if (data instanceof int[]) {
            int[] a = (int[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = (long) a[i];
            return o;
        }
        if (data instanceof long[]) {
            long[] a = (long[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof float[]) {
            float[] a = (float[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof double[]) {
            double[] a = (double[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof boolean[]) {
            boolean[] a = (boolean[]) data;
            Object[] o = new Object[a.length];
            for (int i = 0; i < a.length; i++) o[i] = a[i];
            return o;
        }
        if (data instanceof String[]) return (String[]) data;
        return new Object[]{data};
    }
}
