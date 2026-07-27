package org.bytedeco.pytorch.data.pickle;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.DType;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.global.torch;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Lightweight pickle protocol 2/3/4 reader/writer focused on the subset used
 * by scientific Python (dict/list/tuple/str/bytes/int/float/bool/None) plus
 * a custom opcode path for LibTorch tensors (stored as numpy-compatible
 * contiguous payloads).
 *
 * <p>This is <em>not</em> a full CPython pickle clone — it intentionally
 * rejects arbitrary {@code GLOBAL}/{@code REDUCE} callables except a small
 * allow-list so untrusted files cannot execute code.
 */
public final class Pickle {
    // Protocol opcodes (subset)
    private static final int MARK = '(';
    private static final int STOP = '.';
    private static final int POP = '0';
    private static final int POP_MARK = '1';
    private static final int DUP = '2';
    private static final int FLOAT = 'F';
    private static final int INT = 'I';
    private static final int BININT = 'J';
    private static final int BININT1 = 'K';
    private static final int BININT2 = 'M';
    private static final int NONE = 'N';
    private static final int BINFLOAT = 'G';
    private static final int SHORT_BINSTRING = 'U';
    private static final int BINSTRING = 'T';
    private static final int BINUNICODE = 'X';
    private static final int EMPTY_LIST = ']';
    private static final int APPEND = 'a';
    private static final int APPENDS = 'e';
    private static final int LIST = 'l';
    private static final int EMPTY_DICT = '}';
    private static final int DICT = 'd';
    private static final int SETITEM = 's';
    private static final int SETITEMS = 'u';
    private static final int EMPTY_TUPLE = ')';
    private static final int TUPLE = 't';
    private static final int TUPLE1 = 0x85;
    private static final int TUPLE2 = 0x86;
    private static final int TUPLE3 = 0x87;
    private static final int PROTO = 0x80;
    private static final int FRAME = 0x95;
    private static final int SHORT_BINUNICODE = 0x8c;
    private static final int BINUNICODE8 = 0x8d;
    private static final int BINBYTES = 'B';
    private static final int SHORT_BINBYTES = 'C';
    private static final int BINBYTES8 = 0x8e;
    private static final int NEWTRUE = 0x88;
    private static final int NEWFALSE = 0x89;
    private static final int LONG1 = 0x8a;
    private static final int BINGET = 'h';
    private static final int LONG_BINGET = 'j';
    private static final int BINPUT = 'q';
    private static final int LONG_BINPUT = 'r';
    private static final int MEMOIZE = 0x94;

    // Custom extension for torch tensors: GLOBAL 'org.bytedeco.pytorch\nTensor\n' + REDUCE
    // We instead use a dedicated marker BINUNICODE key "__torch_tensor__" dict form
    // written by our dumpTensor for safe round-trip without code execution.

    private Pickle() {}

    // ---- public API ---------------------------------------------------------

    public static Object load(File file) throws IOException {
        try (FileInputStream in = new FileInputStream(file)) {
            return load(in);
        }
    }

    public static Object load(InputStream in) throws IOException {
        return new Unpickler(in).load();
    }

    public static Object loads(byte[] data) throws IOException {
        return load(new ByteArrayInputStream(data));
    }

    public static void dump(Object obj, File file) throws IOException {
        try (FileOutputStream out = new FileOutputStream(file)) {
            dump(obj, out);
        }
    }

    public static void dump(Object obj, OutputStream out) throws IOException {
        new Pickler(out).dump(obj);
    }

    public static byte[] dumps(Object obj) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        dump(obj, bos);
        return bos.toByteArray();
    }

    /** Convenience: dump a Tensor as a safe self-describing dict. */
    public static void dumpTensor(Tensor t, File file) throws IOException {
        dump(tensorToMap(t), file);
    }

    public static Tensor loadTensor(File file) throws IOException {
        Object o = load(file);
        return mapToTensor(o);
    }

    // ---- tensor map codec ---------------------------------------------------

    @SuppressWarnings("unchecked")
    public static Map<String, Object> tensorToMap(Tensor t) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("__torch_tensor__", true);
        long[] shape = new long[(int) t.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = t.sizes().get(i);
        m.put("shape", shape);
        m.put("dtype", t.scalar_type().name());
        NDArray a = NP.fromTensor(t);
        m.put("data", a.asDoubleArray()); // portable
        m.put("nd_dtype", a.dtype.getDescriptor());
        return m;
    }

    @SuppressWarnings("unchecked")
    public static Tensor mapToTensor(Object o) {
        if (o instanceof Tensor) return (Tensor) o;
        if (!(o instanceof Map)) {
            throw new IllegalArgumentException("not a tensor map: " + (o == null ? null : o.getClass()));
        }
        Map<?, ?> m = (Map<?, ?>) o;
        if (!Boolean.TRUE.equals(m.get("__torch_tensor__"))) {
            throw new IllegalArgumentException("missing __torch_tensor__ marker");
        }
        Object dataObj = m.get("data");
        Object shapeObj = m.get("shape");
        String ndDesc = String.valueOf(m.get("nd_dtype"));
        DType dtype = DType.fromDescriptor(ndDesc);
        long[] shape;
        if (shapeObj instanceof long[]) shape = (long[]) shapeObj;
        else if (shapeObj instanceof List) {
            List<?> L = (List<?>) shapeObj;
            shape = new long[L.size()];
            for (int i = 0; i < L.size(); i++) shape[i] = ((Number) L.get(i)).longValue();
        } else shape = new long[]{((double[]) dataObj).length};

        double[] data;
        if (dataObj instanceof double[]) data = (double[]) dataObj;
        else if (dataObj instanceof List) {
            List<?> L = (List<?>) dataObj;
            data = new double[L.size()];
            for (int i = 0; i < L.size(); i++) data[i] = ((Number) L.get(i)).doubleValue();
        } else throw new IllegalArgumentException("bad data field");

        NDArray a = new NDArray(dtype, shape);
        for (int i = 0; i < data.length; i++) a.setDouble(i, data[i]);
        return NP.toTensor(a);
    }

    // ---- unpickler ----------------------------------------------------------

    private static final class Unpickler {
        private final DataInputStream in;
        private final List<Object> stack = new ArrayList<>();
        private final List<Integer> marks = new ArrayList<>();
        private final Map<Integer, Object> memo = new HashMap<>();

        Unpickler(InputStream in) {
            this.in = new DataInputStream(in);
        }

        Object load() throws IOException {
            while (true) {
                int op = in.read();
                if (op < 0) throw new EOFException("unexpected EOF in pickle");
                switch (op) {
                    case PROTO: {
                        int ver = in.readUnsignedByte();
                        if (ver > 5) throw new IOException("unsupported pickle protocol " + ver);
                        break;
                    }
                    case FRAME: {
                        // skip 8-byte frame size
                        in.readLong();
                        break;
                    }
                    case STOP:
                        if (stack.isEmpty()) return null;
                        return stack.get(stack.size() - 1);
                    case MARK:
                        marks.add(stack.size());
                        break;
                    case POP:
                        if (!stack.isEmpty()) stack.remove(stack.size() - 1);
                        break;
                    case POP_MARK:
                        popMark();
                        break;
                    case DUP:
                        stack.add(stack.get(stack.size() - 1));
                        break;
                    case NONE:
                        stack.add(null);
                        break;
                    case NEWTRUE:
                        stack.add(Boolean.TRUE);
                        break;
                    case NEWFALSE:
                        stack.add(Boolean.FALSE);
                        break;
                    case INT: {
                        String s = readLine();
                        if ("01".equals(s)) stack.add(Boolean.TRUE);
                        else if ("00".equals(s)) stack.add(Boolean.FALSE);
                        else {
                            // Prefer Integer when value fits int32 (Java fidelity for dumps(Integer)).
                            long v = Long.parseLong(s);
                            stack.add(boxIntOrLong(v));
                        }
                        break;
                    }
                    case BININT: {
                        // pickle BININT is little-endian; DataInputStream.readInt is big-endian
                        int v = Integer.reverseBytes(in.readInt());
                        stack.add(Integer.valueOf(v));
                        break;
                    }
                    case BININT1:
                        stack.add(Integer.valueOf(in.readUnsignedByte()));
                        break;
                    case BININT2: {
                        int lo = in.readUnsignedByte();
                        int hi = in.readUnsignedByte();
                        stack.add(Integer.valueOf(lo | (hi << 8)));
                        break;
                    }
                    case LONG1: {
                        int n = in.readUnsignedByte();
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(bytesToLongLE(raw));
                        break;
                    }
                    case FLOAT: {
                        stack.add(Double.parseDouble(readLine()));
                        break;
                    }
                    case BINFLOAT: {
                        // pickle BINFLOAT is big-endian IEEE-754; DataInputStream.readLong is big-endian
                        stack.add(Double.longBitsToDouble(in.readLong()));
                        break;
                    }
                    case SHORT_BINSTRING: {
                        int n = in.readUnsignedByte();
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(new String(raw, StandardCharsets.ISO_8859_1));
                        break;
                    }
                    case BINSTRING: {
                        int n = Integer.reverseBytes(in.readInt());
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(new String(raw, StandardCharsets.ISO_8859_1));
                        break;
                    }
                    case SHORT_BINUNICODE: {
                        int n = in.readUnsignedByte();
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(new String(raw, StandardCharsets.UTF_8));
                        break;
                    }
                    case BINUNICODE: {
                        int n = Integer.reverseBytes(in.readInt());
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(new String(raw, StandardCharsets.UTF_8));
                        break;
                    }
                    case BINUNICODE8: {
                        long n = Long.reverseBytes(in.readLong());
                        byte[] raw = new byte[(int) n];
                        in.readFully(raw);
                        stack.add(new String(raw, StandardCharsets.UTF_8));
                        break;
                    }
                    case SHORT_BINBYTES: {
                        int n = in.readUnsignedByte();
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(raw);
                        break;
                    }
                    case BINBYTES: {
                        int n = Integer.reverseBytes(in.readInt());
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(raw);
                        break;
                    }
                    case BINBYTES8: {
                        long n = Long.reverseBytes(in.readLong());
                        byte[] raw = new byte[(int) n];
                        in.readFully(raw);
                        stack.add(raw);
                        break;
                    }
                    case EMPTY_LIST:
                        stack.add(new ArrayList<>());
                        break;
                    case EMPTY_DICT:
                        stack.add(new LinkedHashMap<>());
                        break;
                    case EMPTY_TUPLE:
                        stack.add(new Object[0]);
                        break;
                    case LIST: {
                        int start = popMark();
                        List<Object> list = new ArrayList<>(stack.subList(start, stack.size()));
                        stack.subList(start, stack.size()).clear();
                        stack.add(list);
                        break;
                    }
                    case DICT: {
                        int start = popMark();
                        Map<Object, Object> dict = new LinkedHashMap<>();
                        for (int i = start; i + 1 < stack.size(); i += 2) {
                            dict.put(stack.get(i), stack.get(i + 1));
                        }
                        stack.subList(start, stack.size()).clear();
                        stack.add(unwrapJavaMarker(dict));
                        break;
                    }
                    case TUPLE: {
                        int start = popMark();
                        Object[] tup = stack.subList(start, stack.size()).toArray();
                        stack.subList(start, stack.size()).clear();
                        stack.add(tup);
                        break;
                    }
                    case TUPLE1: {
                        Object a = stack.remove(stack.size() - 1);
                        stack.add(new Object[]{a});
                        break;
                    }
                    case TUPLE2: {
                        Object b = stack.remove(stack.size() - 1);
                        Object a = stack.remove(stack.size() - 1);
                        stack.add(new Object[]{a, b});
                        break;
                    }
                    case TUPLE3: {
                        Object c = stack.remove(stack.size() - 1);
                        Object b = stack.remove(stack.size() - 1);
                        Object a = stack.remove(stack.size() - 1);
                        stack.add(new Object[]{a, b, c});
                        break;
                    }
                    case APPEND: {
                        Object v = stack.remove(stack.size() - 1);
                        @SuppressWarnings("unchecked")
                        List<Object> list = (List<Object>) stack.get(stack.size() - 1);
                        list.add(v);
                        break;
                    }
                    case APPENDS: {
                        int start = popMark();
                        @SuppressWarnings("unchecked")
                        List<Object> list = (List<Object>) stack.get(start - 1);
                        list.addAll(stack.subList(start, stack.size()));
                        stack.subList(start, stack.size()).clear();
                        break;
                    }
                    case SETITEM: {
                        Object v = stack.remove(stack.size() - 1);
                        Object k = stack.remove(stack.size() - 1);
                        @SuppressWarnings("unchecked")
                        Map<Object, Object> dict = (Map<Object, Object>) stack.get(stack.size() - 1);
                        dict.put(k, v);
                        // unwrap Java-typed markers when the dict is complete
                        Object unwrapped = unwrapJavaMarker(dict);
                        if (unwrapped != dict) {
                            stack.set(stack.size() - 1, unwrapped);
                        }
                        break;
                    }
                    case SETITEMS: {
                        int start = popMark();
                        @SuppressWarnings("unchecked")
                        Map<Object, Object> dict = (Map<Object, Object>) stack.get(start - 1);
                        for (int i = start; i + 1 < stack.size(); i += 2) {
                            dict.put(stack.get(i), stack.get(i + 1));
                        }
                        stack.subList(start, stack.size()).clear();
                        Object unwrapped = unwrapJavaMarker(dict);
                        if (unwrapped != dict) {
                            stack.set(start - 1, unwrapped);
                        }
                        break;
                    }
                    case BINPUT: {
                        int idx = in.readUnsignedByte();
                        memo.put(idx, stack.get(stack.size() - 1));
                        break;
                    }
                    case LONG_BINPUT: {
                        int idx = Integer.reverseBytes(in.readInt());
                        memo.put(idx, stack.get(stack.size() - 1));
                        break;
                    }
                    case MEMOIZE: {
                        memo.put(memo.size(), stack.get(stack.size() - 1));
                        break;
                    }
                    case BINGET: {
                        int idx = in.readUnsignedByte();
                        stack.add(memo.get(idx));
                        break;
                    }
                    case LONG_BINGET: {
                        int idx = Integer.reverseBytes(in.readInt());
                        stack.add(memo.get(idx));
                        break;
                    }
                    default:
                        throw new IOException(String.format(
                                "unsupported / unsafe pickle opcode 0x%02x — refused (no arbitrary GLOBAL/REDUCE)", op));
                }
            }
        }

        private int popMark() {
            if (marks.isEmpty()) throw new IllegalStateException("no mark");
            return marks.remove(marks.size() - 1);
        }

        private String readLine() throws IOException {
            StringBuilder sb = new StringBuilder();
            while (true) {
                int c = in.read();
                if (c < 0 || c == '\n') break;
                if (c != '\r') sb.append((char) c);
            }
            return sb.toString();
        }

        private static long bytesToLongLE(byte[] raw) {
            long v = 0;
            for (int i = 0; i < Math.min(8, raw.length); i++) {
                v |= ((long) (raw[i] & 0xff)) << (8 * i);
            }
            return v;
        }
    }

    /** Prefer Integer when value fits int32 (Java fidelity for dumps(Integer)). */
    private static Number boxIntOrLong(long v) {
        if (v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE) return Integer.valueOf((int) v);
        return Long.valueOf(v);
    }

    /**
     * Unwrap Java-typed pickle markers written by {@link Pickler}:
     * <ul>
     *   <li>{@code {__jlong__: v}} → Long</li>
     *   <li>{@code {__jfloat__: v}} → Float</li>
     *   <li>{@code {__jdouble_array__: list}} → double[]</li>
     *   <li>{@code {__jlong_array__: list}} → long[]</li>
     * </ul>
     */
    @SuppressWarnings("unchecked")
    private static Object unwrapJavaMarker(Map<Object, Object> dict) {
        if (dict == null || dict.size() != 1) return dict;
        if (dict.containsKey("__jlong__")) {
            Object v = dict.get("__jlong__");
            if (v instanceof Number) return Long.valueOf(((Number) v).longValue());
        }
        if (dict.containsKey("__jfloat__")) {
            Object v = dict.get("__jfloat__");
            if (v instanceof Number) return Float.valueOf(((Number) v).floatValue());
        }
        if (dict.containsKey("__jdouble_array__")) {
            Object v = dict.get("__jdouble_array__");
            if (v instanceof List) {
                List<?> list = (List<?>) v;
                double[] a = new double[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    Object e = list.get(i);
                    a[i] = e instanceof Number ? ((Number) e).doubleValue() : Double.NaN;
                }
                return a;
            }
        }
        if (dict.containsKey("__jlong_array__")) {
            Object v = dict.get("__jlong_array__");
            if (v instanceof List) {
                List<?> list = (List<?>) v;
                long[] a = new long[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    Object e = list.get(i);
                    a[i] = e instanceof Number ? ((Number) e).longValue() : 0L;
                }
                return a;
            }
        }
        return dict;
    }

    // ---- pickler ------------------------------------------------------------

    private static final class Pickler {
        private final DataOutputStream out;
        private int memoIdx = 0;

        Pickler(OutputStream out) {
            this.out = new DataOutputStream(out);
        }

        void dump(Object obj) throws IOException {
            out.writeByte(PROTO);
            out.writeByte(4);
            write(obj);
            out.writeByte(STOP);
            out.flush();
        }

        @SuppressWarnings("unchecked")
        private void write(Object obj) throws IOException {
            if (obj == null) {
                out.writeByte(NONE);
            } else if (obj instanceof Boolean) {
                out.writeByte(((Boolean) obj) ? 0x88 : 0x89);
            } else if (obj instanceof Integer) {
                writeLong(((Integer) obj).longValue());
            } else if (obj instanceof Long) {
                // Always mark Long so unpickle restores Long (not Integer for small values).
                writeMarkedRawLong("__jlong__", (Long) obj);
            } else if (obj instanceof Float) {
                // Mark Float; pickle only has BINFLOAT (double).
                writeMarkedRawDouble("__jfloat__", ((Float) obj).doubleValue());
            } else if (obj instanceof Double) {
                writeDouble((Double) obj);
            } else if (obj instanceof String) {
                writeUnicode((String) obj);
            } else if (obj instanceof byte[]) {
                writeBytes((byte[]) obj);
            } else if (obj instanceof double[]) {
                // Mark so unpickle restores double[] (not List).
                writeMarkedDoubleArray((double[]) obj);
            } else if (obj instanceof long[]) {
                writeMarkedLongArray((long[]) obj);
            } else if (obj instanceof List) {
                out.writeByte(EMPTY_LIST);
                memoize();
                out.writeByte(MARK);
                for (Object v : (List<?>) obj) write(v);
                out.writeByte(APPENDS);
            } else if (obj instanceof Object[]) {
                Object[] arr = (Object[]) obj;
                out.writeByte(MARK);
                for (Object v : arr) write(v);
                out.writeByte(TUPLE);
            } else if (obj instanceof Map) {
                out.writeByte(EMPTY_DICT);
                memoize();
                out.writeByte(MARK);
                for (Map.Entry<?, ?> e : ((Map<?, ?>) obj).entrySet()) {
                    write(e.getKey());
                    write(e.getValue());
                }
                out.writeByte(SETITEMS);
            } else if (obj instanceof Tensor) {
                write(tensorToMap((Tensor) obj));
            } else if (obj instanceof NDArray) {
                NDArray a = (NDArray) obj;
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("__ndarray__", true);
                m.put("dtype", a.dtype.getDescriptor());
                m.put("shape", a.shape);
                m.put("data", a.asDoubleArray());
                write(m);
            } else {
                throw new IOException("cannot pickle type: " + obj.getClass().getName());
            }
        }

        /** Write {@code {key: longValue}} without re-entering Long marking. */
        private void writeMarkedRawLong(String key, long value) throws IOException {
            out.writeByte(EMPTY_DICT);
            memoize();
            out.writeByte(MARK);
            writeUnicode(key);
            writeLong(value); // raw int/long opcodes — no marker
            out.writeByte(SETITEMS);
        }

        private void writeMarkedRawDouble(String key, double value) throws IOException {
            out.writeByte(EMPTY_DICT);
            memoize();
            out.writeByte(MARK);
            writeUnicode(key);
            writeDouble(value);
            out.writeByte(SETITEMS);
        }

        private void writeMarkedDoubleArray(double[] arr) throws IOException {
            out.writeByte(EMPTY_DICT);
            memoize();
            out.writeByte(MARK);
            writeUnicode("__jdouble_array__");
            out.writeByte(EMPTY_LIST);
            memoize();
            out.writeByte(MARK);
            for (double v : arr) writeDouble(v);
            out.writeByte(APPENDS);
            out.writeByte(SETITEMS);
        }

        private void writeMarkedLongArray(long[] arr) throws IOException {
            out.writeByte(EMPTY_DICT);
            memoize();
            out.writeByte(MARK);
            writeUnicode("__jlong_array__");
            out.writeByte(EMPTY_LIST);
            memoize();
            out.writeByte(MARK);
            for (long v : arr) writeLong(v);
            out.writeByte(APPENDS);
            out.writeByte(SETITEMS);
        }

        private void writeLong(long v) throws IOException {
            if (v >= 0 && v <= 0xff) {
                out.writeByte(BININT1);
                out.writeByte((int) v);
            } else if (v >= 0 && v <= 0xffff) {
                out.writeByte(BININT2);
                out.writeByte((int) (v & 0xff));
                out.writeByte((int) ((v >> 8) & 0xff));
            } else if (v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE) {
                out.writeByte(BININT);
                int i = (int) v;
                out.writeByte(i & 0xff);
                out.writeByte((i >> 8) & 0xff);
                out.writeByte((i >> 16) & 0xff);
                out.writeByte((i >> 24) & 0xff);
            } else {
                out.writeByte(LONG1);
                ByteBuffer bb = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(v);
                out.writeByte(8);
                out.write(bb.array());
            }
        }

        private void writeDouble(double v) throws IOException {
            out.writeByte(BINFLOAT);
            long bits = Double.doubleToLongBits(v);
            // big-endian for BINFLOAT
            out.writeLong(bits);
        }

        private void writeUnicode(String s) throws IOException {
            byte[] raw = s.getBytes(StandardCharsets.UTF_8);
            if (raw.length < 256) {
                out.writeByte(SHORT_BINUNICODE);
                out.writeByte(raw.length);
            } else {
                out.writeByte(BINUNICODE);
                int n = raw.length;
                out.writeByte(n & 0xff);
                out.writeByte((n >> 8) & 0xff);
                out.writeByte((n >> 16) & 0xff);
                out.writeByte((n >> 24) & 0xff);
            }
            out.write(raw);
            memoize();
        }

        private void writeBytes(byte[] raw) throws IOException {
            if (raw.length < 256) {
                out.writeByte(SHORT_BINBYTES);
                out.writeByte(raw.length);
            } else {
                out.writeByte(BINBYTES);
                int n = raw.length;
                out.writeByte(n & 0xff);
                out.writeByte((n >> 8) & 0xff);
                out.writeByte((n >> 16) & 0xff);
                out.writeByte((n >> 24) & 0xff);
            }
            out.write(raw);
            memoize();
        }

        private void memoize() throws IOException {
            if (memoIdx < 256) {
                out.writeByte(BINPUT);
                out.writeByte(memoIdx);
            } else {
                out.writeByte(LONG_BINPUT);
                int i = memoIdx;
                out.writeByte(i & 0xff);
                out.writeByte((i >> 8) & 0xff);
                out.writeByte((i >> 16) & 0xff);
                out.writeByte((i >> 24) & 0xff);
            }
            memoIdx++;
        }
    }
}
