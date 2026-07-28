package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.nn.*;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.data.safetensors.SafeDType;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;

/**
 * Pure-Java reader for Python {@code torch.save(...)} checkpoints ({@code .pth}/{@code .pt}).
 *
 * <p>LibTorch / JavaCPP cannot load CPython pickles. Modern PyTorch writes a ZIP archive:
 * <pre>
 *   &lt;prefix&gt;/data.pkl          # pickle protocol 2+ graph
 *   &lt;prefix&gt;/data/&lt;N&gt;          # raw storage payloads
 *   &lt;prefix&gt;/byteorder         # little / big
 *   &lt;prefix&gt;/version
 * </pre>
 *
 * <p>This reader implements a <em>restricted</em> unpickler focused on tensors:
 * <ul>
 *   <li>{@code collections.OrderedDict}</li>
 *   <li>{@code torch._utils._rebuild_tensor_v2}</li>
 *   <li>{@code torch.*Storage} via {@code BINPERSID} → ZIP data files</li>
 * </ul>
 * Unknown {@code GLOBAL}/{@code REDUCE} (optimizer, custom classes, …) become
 * inert stubs so full training checkpoints still yield a usable state_dict —
 * no Python code is executed.
 *
 * <p>Typical usage — extract a flat state-dict of tensors:
 * <pre>
 *   Map&lt;String, Tensor&gt; sd = TorchPthReader.loadStateDict(new File("model.pth"));
 *   ModelStructure.printStateDict("model.pth", sd);
 * </pre>
 */
public final class TorchPthReader {

    private TorchPthReader() {}

    // ---- public API ---------------------------------------------------------

    /**
     * Load tensors from a Python torch ZIP checkpoint.
     * <ul>
     *   <li>If root is a tensor map ({@code state_dict}), return it.</li>
     *   <li>If root is a training checkpoint with {@code model_state_dict} /
     *       {@code state_dict} / {@code model}, unwrap that entry.</li>
     *   <li>Otherwise collect every {@link Tensor} found under string keys
     *       (best-effort flatten).</li>
     * </ul>
     */
    public static Map<String, Tensor> loadStateDict(File file) throws IOException {
        Object root = loadRaw(file);
        return extractStateDict(root);
    }

    /**
     * Load state-dict and print a structure report to {@link System#out}.
     *
     * @return the extracted tensor map
     */
    public static Map<String, Tensor> loadStateDictAndPrint(File file) throws IOException {
        Map<String, Tensor> sd = loadStateDict(file);
        ModelStructure.printStateDict(file.getName(), sd);
        return sd;
    }

    public static Map<String, Tensor> loadStateDictAndPrint(String path) throws IOException {
        return loadStateDictAndPrint(new File(path));
    }

    public static Map<String, Tensor> loadStateDict(Path path) throws IOException {
        return loadStateDict(path.toFile());
    }

    public static Map<String, Tensor> loadStateDict(String path) throws IOException {
        return loadStateDict(new File(path));
    }

    /** Load the raw unpickled object graph (maps / lists / tensors / scalars). */
    public static Object loadRaw(File file) throws IOException {
        if (file == null || !file.isFile()) {
            throw new IOException("not a file: " + file);
        }
        if (!isZipTorch(file)) {
            throw new IOException("Not a modern torch ZIP .pth/.pt (expected PK\\x03\\x04). "
                + "Legacy non-zip pickles and torchscript archives are not supported.");
        }
        try (ZipFile zip = new ZipFile(file)) {
            String prefix = detectPrefix(zip);
            byte[] pkl = readEntry(zip, prefix + "data.pkl");
            if (pkl == null) {
                throw new IOException("torch ZIP missing " + prefix + "data.pkl");
            }
            ByteOrder order = ByteOrder.LITTLE_ENDIAN;
            byte[] bo = readEntry(zip, prefix + "byteorder");
            if (bo != null) {
                String s = new String(bo, StandardCharsets.US_ASCII).trim().toLowerCase(Locale.ROOT);
                if (s.startsWith("big")) order = ByteOrder.BIG_ENDIAN;
            }
            Map<String, byte[]> storages = loadStorages(zip, prefix);
            return new TorchUnpickler(pkl, storages, order).load();
        }
    }

    /** True if file looks like a ZIP-based torch.save archive. */
    public static boolean isZipTorch(File file) throws IOException {
        if (file == null || !file.isFile() || file.length() < 4) return false;
        byte[] mag = Files.readAllBytes(file.toPath().toAbsolutePath());
        // only need first 4 — avoid reading whole file for large models
        try (InputStream in = Files.newInputStream(file.toPath())) {
            byte[] m = in.readNBytes(4);
            return m.length == 4 && m[0] == 'P' && m[1] == 'K' && m[2] == 3 && m[3] == 4;
        }
    }

    public static boolean isZipTorch(Path path) throws IOException {
        return isZipTorch(path.toFile());
    }

    /**
     * Best-effort extraction of a string→Tensor map from a torch checkpoint object.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Tensor> extractStateDict(Object root) {
        if (root == null) return Map.of();
        if (root instanceof Map) {
            Map<?, ?> m = (Map<?, ?>) root;
            // Prefer common checkpoint wrappers
            for (String key : new String[]{"model_state_dict", "state_dict", "model", "module", "net"}) {
                Object v = m.get(key);
                if (v instanceof Map && looksLikeStateDict((Map<?, ?>) v)) {
                    return toTensorMap((Map<?, ?>) v);
                }
            }
            if (looksLikeStateDict(m)) {
                return toTensorMap(m);
            }
            // Flatten nested: collect all tensors under string keys
            Map<String, Tensor> flat = new LinkedHashMap<>();
            flattenTensors("", m, flat);
            if (!flat.isEmpty()) return flat;
        }
        if (root instanceof Tensor) {
            Map<String, Tensor> one = new LinkedHashMap<>();
            one.put("tensor", (Tensor) root);
            return one;
        }
        throw new IllegalArgumentException(
            "Cannot extract state_dict from " + (root == null ? "null" : root.getClass().getName()));
    }

    // ---- helpers ------------------------------------------------------------

    private static boolean looksLikeStateDict(Map<?, ?> m) {
        if (m.isEmpty()) return false;
        int tensors = 0, keys = 0;
        for (Map.Entry<?, ?> e : m.entrySet()) {
            if (!(e.getKey() instanceof String)) return false;
            keys++;
            if (e.getValue() instanceof Tensor) tensors++;
        }
        return keys > 0 && tensors * 2 >= keys; // majority tensors
    }

    private static Map<String, Tensor> toTensorMap(Map<?, ?> m) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<?, ?> e : m.entrySet()) {
            if (e.getKey() instanceof String && e.getValue() instanceof Tensor) {
                out.put((String) e.getKey(), (Tensor) e.getValue());
            }
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static void flattenTensors(String prefix, Map<?, ?> m, Map<String, Tensor> out) {
        for (Map.Entry<?, ?> e : m.entrySet()) {
            if (!(e.getKey() instanceof String)) continue;
            String k = (String) e.getKey();
            String name = prefix.isEmpty() ? k : prefix + "." + k;
            Object v = e.getValue();
            if (v instanceof Tensor) {
                out.put(name, (Tensor) v);
            } else if (v instanceof Map) {
                flattenTensors(name, (Map<?, ?>) v, out);
            }
        }
    }

    private static String detectPrefix(ZipFile zip) {
        // Prefer entry ending with data.pkl
        String found = null;
        var en = zip.entries();
        while (en.hasMoreElements()) {
            ZipEntry e = en.nextElement();
            String n = e.getName().replace('\\', '/');
            if (n.endsWith("data.pkl")) {
                int slash = n.lastIndexOf('/');
                found = slash >= 0 ? n.substring(0, slash + 1) : "";
                break;
            }
        }
        if (found != null) return found;
        // fallback: first directory
        en = zip.entries();
        while (en.hasMoreElements()) {
            ZipEntry e = en.nextElement();
            String n = e.getName().replace('\\', '/');
            int slash = n.indexOf('/');
            if (slash > 0) return n.substring(0, slash + 1);
        }
        return "";
    }

    private static byte[] readEntry(ZipFile zip, String name) throws IOException {
        ZipEntry e = zip.getEntry(name);
        if (e == null) {
            // try without leading ./
            e = zip.getEntry(name.startsWith("/") ? name.substring(1) : name);
        }
        if (e == null) return null;
        try (InputStream in = zip.getInputStream(e)) {
            return in.readAllBytes();
        }
    }

    private static Map<String, byte[]> loadStorages(ZipFile zip, String prefix) throws IOException {
        Map<String, byte[]> map = new HashMap<>();
        String dataDir = prefix + "data/";
        var en = zip.entries();
        while (en.hasMoreElements()) {
            ZipEntry e = en.nextElement();
            String n = e.getName().replace('\\', '/');
            if (!n.startsWith(dataDir) || n.endsWith("/")) continue;
            String key = n.substring(dataDir.length()); // "0", "1", ...
            try (InputStream in = zip.getInputStream(e)) {
                map.put(key, in.readAllBytes());
            }
        }
        return map;
    }

    // ---- restricted unpickler ----------------------------------------------

    private static final class TorchUnpickler {
        // opcodes
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
        private static final int PERSID = 'P';
        private static final int BINPERSID = 'Q';
        private static final int REDUCE = 'R';
        private static final int STRING = 'S';
        private static final int BINSTRING = 'T';
        private static final int SHORT_BINSTRING = 'U';
        private static final int UNICODE = 'V';
        private static final int BINUNICODE = 'X';
        private static final int APPEND = 'a';
        private static final int BUILD = 'b';
        private static final int GLOBAL = 'c';
        private static final int DICT = 'd';
        private static final int EMPTY_DICT = '}';
        private static final int APPENDS = 'e';
        private static final int GET = 'g';
        private static final int BINGET = 'h';
        private static final int INST = 'i';
        private static final int LONG_BINGET = 'j';
        private static final int LIST = 'l';
        private static final int EMPTY_LIST = ']';
        private static final int OBJ = 'o';
        private static final int PUT = 'p';
        private static final int BINPUT = 'q';
        private static final int LONG_BINPUT = 'r';
        private static final int SETITEM = 's';
        private static final int TUPLE = 't';
        private static final int EMPTY_TUPLE = ')';
        private static final int SETITEMS = 'u';
        private static final int BINFLOAT = 'G';
        private static final int PROTO = 0x80;
        private static final int NEWOBJ = 0x81;
        private static final int TUPLE1 = 0x85;
        private static final int TUPLE2 = 0x86;
        private static final int TUPLE3 = 0x87;
        private static final int NEWTRUE = 0x88;
        private static final int NEWFALSE = 0x89;
        private static final int LONG1 = 0x8a;
        private static final int LONG4 = 0x8b;
        private static final int SHORT_BINUNICODE = 0x8c;
        private static final int BINUNICODE8 = 0x8d;
        private static final int BINBYTES = 'B';
        private static final int SHORT_BINBYTES = 'C';
        private static final int BINBYTES8 = 0x8e;
        private static final int EMPTY_SET = 0x8f;
        private static final int FROZENSET = 0x91;
        private static final int NEWOBJ_EX = 0x92;
        private static final int STACK_GLOBAL = 0x93;
        private static final int MEMOIZE = 0x94;
        private static final int FRAME = 0x95;
        private static final int BYTEARRAY8 = 0x96;
        private static final int NEXT_BUFFER = 0x97;
        private static final int READONLY_BUFFER = 0x98;

        private final DataInputStream in;
        private final Map<String, byte[]> storages;
        private final ByteOrder byteOrder;
        private final List<Object> stack = new ArrayList<>();
        private final List<Integer> marks = new ArrayList<>();
        private final Map<Integer, Object> memo = new HashMap<>();

        TorchUnpickler(byte[] pkl, Map<String, byte[]> storages, ByteOrder order) {
            this.in = new DataInputStream(new ByteArrayInputStream(pkl));
            this.storages = storages;
            this.byteOrder = order == null ? ByteOrder.LITTLE_ENDIAN : order;
        }

        Object load() throws IOException {
            while (true) {
                int op = in.read();
                if (op < 0) throw new EOFException("unexpected EOF in torch data.pkl");
                switch (op) {
                    case PROTO: {
                        int ver = in.readUnsignedByte();
                        if (ver > 5) throw new IOException("unsupported pickle protocol " + ver);
                        break;
                    }
                    case FRAME:
                        in.readLong(); // skip frame length
                        break;
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
                        else stack.add(boxLong(Long.parseLong(s)));
                        break;
                    }
                    case BININT:
                        stack.add(Integer.valueOf(Integer.reverseBytes(in.readInt())));
                        break;
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
                    case LONG4: {
                        int n = Integer.reverseBytes(in.readInt());
                        byte[] raw = new byte[n];
                        in.readFully(raw);
                        stack.add(bytesToLongLE(raw));
                        break;
                    }
                    case FLOAT:
                        stack.add(Double.parseDouble(readLine()));
                        break;
                    case BINFLOAT:
                        stack.add(Double.longBitsToDouble(in.readLong()));
                        break;
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
                    case EMPTY_SET:
                        stack.add(new java.util.LinkedHashSet<>());
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
                        stack.add(dict);
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
                        break;
                    }
                    case BINPUT: {
                        int i = in.readUnsignedByte();
                        memo.put(i, stack.get(stack.size() - 1));
                        break;
                    }
                    case LONG_BINPUT: {
                        int i = Integer.reverseBytes(in.readInt());
                        memo.put(i, stack.get(stack.size() - 1));
                        break;
                    }
                    case MEMOIZE: {
                        memo.put(memo.size(), stack.get(stack.size() - 1));
                        break;
                    }
                    case BINGET: {
                        int i = in.readUnsignedByte();
                        stack.add(memo.get(i));
                        break;
                    }
                    case LONG_BINGET: {
                        int i = Integer.reverseBytes(in.readInt());
                        stack.add(memo.get(i));
                        break;
                    }
                    case GLOBAL: {
                        String mod = readLine();
                        String name = readLine();
                        stack.add(resolveGlobal(mod, name));
                        break;
                    }
                    case STACK_GLOBAL: {
                        Object nameObj = stack.remove(stack.size() - 1);
                        Object modObj = stack.remove(stack.size() - 1);
                        stack.add(resolveGlobal(String.valueOf(modObj), String.valueOf(nameObj)));
                        break;
                    }
                    case REDUCE: {
                        Object args = stack.remove(stack.size() - 1);
                        Object callable = stack.remove(stack.size() - 1);
                        stack.add(applyReduce(callable, args));
                        break;
                    }
                    case NEWOBJ: {
                        Object args = stack.remove(stack.size() - 1);
                        Object cls = stack.remove(stack.size() - 1);
                        stack.add(applyReduce(cls, args));
                        break;
                    }
                    case BUILD: {
                        // state for objects — OrderedDict uses empty state; ignore dict update
                        Object state = stack.remove(stack.size() - 1);
                        Object obj = stack.get(stack.size() - 1);
                        if (obj instanceof Map && state instanceof Map) {
                            @SuppressWarnings("unchecked")
                            Map<Object, Object> m = (Map<Object, Object>) obj;
                            for (Map.Entry<?, ?> e : ((Map<?, ?>) state).entrySet()) {
                                m.put(e.getKey(), e.getValue());
                            }
                        }
                        // else: no-op for our allow-listed types
                        break;
                    }
                    case BINPERSID: {
                        Object pid = stack.remove(stack.size() - 1);
                        stack.add(persistentLoad(pid));
                        break;
                    }
                    case PERSID: {
                        String pid = readLine();
                        stack.add(persistentLoad(pid));
                        break;
                    }
                    default:
                        throw new IOException(String.format(
                            "unsupported pickle opcode 0x%02x in torch data.pkl", op));
                }
            }
        }

        private Object resolveGlobal(String mod, String name) throws IOException {
            String m = mod == null ? "" : mod.trim();
            String n = name == null ? "" : name.trim();
            if (("collections".equals(m) || "collections.abc".equals(m))
                && ("OrderedDict".equals(n) || "defaultdict".equals(n))) {
                return new GlobalRef("collections", "OrderedDict");
            }
            if ("torch._utils".equals(m) && ("_rebuild_tensor_v2".equals(n)
                || "_rebuild_tensor".equals(n)
                || "_rebuild_parameter".equals(n)
                || "_rebuild_qtensor".equals(n))) {
                return new GlobalRef("torch._utils", n);
            }
            if ("torch".equals(m) && n.endsWith("Storage")) {
                return new GlobalRef("torch", n);
            }
            // torch.nn.parameter.Parameter / storages under torch.*
            if (m.startsWith("torch") && ("Parameter".equals(n) || n.endsWith("Storage"))) {
                return new GlobalRef(m, n);
            }
            // Inert stub for optimizer state, custom classes, etc. — no code exec.
            // Allows full training checkpoints (state_dict + optimizer + epoch) to parse.
            return new GlobalRef(m, n, /*stub=*/true);
        }

        private Object applyReduce(Object callable, Object args) throws IOException {
            if (!(callable instanceof GlobalRef)) {
                // Unknown callable → inert stub (keep parse going)
                return new StubObject(String.valueOf(callable), args);
            }
            GlobalRef g = (GlobalRef) callable;
            Object[] a = asArray(args);

            if (g.stub) {
                return new StubObject(g.mod + "." + g.name, args);
            }

            if ("collections".equals(g.mod) && "OrderedDict".equals(g.name)) {
                // OrderedDict() or OrderedDict(iterable of pairs)
                Map<Object, Object> od = new LinkedHashMap<>();
                if (a.length == 1 && a[0] instanceof List) {
                    for (Object item : (List<?>) a[0]) {
                        Object[] pair = asArray(item);
                        if (pair.length >= 2) od.put(pair[0], pair[1]);
                    }
                } else if (a.length == 1 && a[0] instanceof Object[]) {
                    for (Object item : (Object[]) a[0]) {
                        Object[] pair = asArray(item);
                        if (pair.length >= 2) od.put(pair[0], pair[1]);
                    }
                }
                return od;
            }

            if ("torch._utils".equals(g.mod) && g.name.startsWith("_rebuild_tensor")) {
                return rebuildTensor(a);
            }
            if ("torch._utils".equals(g.mod) && "_rebuild_parameter".equals(g.name)) {
                // (tensor, requires_grad, ...) → just tensor
                if (a.length >= 1 && a[0] instanceof Tensor) return a[0];
                return rebuildTensor(a);
            }
            if ("torch".equals(g.mod) && g.name.endsWith("Storage")) {
                // Storage constructor via REDUCE is unusual in ZIP format (uses PERSID);
                // treat as type marker.
                return g;
            }
            return new StubObject(g.mod + "." + g.name, args);
        }

        /**
         * _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)
         */
        private Tensor rebuildTensor(Object[] a) throws IOException {
            if (a.length < 4) {
                throw new IOException("_rebuild_tensor_v2 expects ≥4 args, got " + a.length);
            }
            Object storageObj = a[0];
            long storageOffset = toLong(a[1]);
            long[] size = toLongArray(a[2]);
            long[] stride = toLongArray(a[3]);
            // requires_grad at a[4] ignored

            StorageView storage;
            if (storageObj instanceof StorageView) {
                storage = (StorageView) storageObj;
            } else {
                throw new IOException("tensor storage is not a StorageView: "
                    + (storageObj == null ? null : storageObj.getClass()));
            }

            return storageToTensor(storage, storageOffset, size, stride);
        }

        /**
         * persistent_load for ZIP: pid is typically
         * ("storage", storage_type, key, location, numel) as tuple/list.
         */
        private Object persistentLoad(Object pid) throws IOException {
            Object[] a = asArray(pid);
            if (a.length < 5) {
                // sometimes just key string
                if (pid instanceof String) {
                    byte[] raw = storages.get(pid);
                    if (raw == null) throw new IOException("missing storage key " + pid);
                    return new StorageView("ByteStorage", raw);
                }
                throw new IOException("unexpected PERSID: " + pid);
            }
            String tag = String.valueOf(a[0]);
            if (!"storage".equals(tag)) {
                throw new IOException("unsupported PERSID tag: " + tag);
            }
            String storageType;
            if (a[1] instanceof GlobalRef) {
                storageType = ((GlobalRef) a[1]).name;
            } else {
                storageType = String.valueOf(a[1]);
                // "torch.FloatStorage" or "FloatStorage"
                int dot = storageType.lastIndexOf('.');
                if (dot >= 0) storageType = storageType.substring(dot + 1);
            }
            String key = String.valueOf(a[2]);
            // a[3] location, a[4] numel
            byte[] raw = storages.get(key);
            if (raw == null) {
                throw new IOException("torch ZIP missing storage data/" + key
                    + " for type " + storageType);
            }
            return new StorageView(storageType, raw);
        }

        private Tensor storageToTensor(StorageView storage, long offset, long[] size, long[] stride)
                throws IOException {
            ScalarType st = storageTypeToScalar(storage.typeName);
            SafeDType sdt = SafeDType.fromTorch(st);
            int elemSize = sdt.sizeBytes();
            long numel = 1;
            for (long d : size) numel *= Math.max(d, 0);
            long start = offset * (long) elemSize;
            long nbytes = numel * (long) elemSize;
            if (start < 0 || start + nbytes > storage.bytes.length) {
                // Some storages are over-allocated; allow reading min available
                if (start < 0 || start >= storage.bytes.length) {
                    throw new IOException("storage slice out of bounds: offset=" + offset
                        + " numel=" + numel + " storageBytes=" + storage.bytes.length
                        + " type=" + storage.typeName);
                }
                nbytes = Math.min(nbytes, storage.bytes.length - start);
            }

            // Build contiguous tensor from raw bytes (handle endianness + bool)
            byte[] slice = new byte[(int) nbytes];
            System.arraycopy(storage.bytes, (int) start, slice, 0, (int) nbytes);
            if (byteOrder == ByteOrder.BIG_ENDIAN && elemSize > 1) {
                swapEndianInPlace(slice, elemSize);
            }

            // Non-contiguous source: rebuild via strided indexing into a contiguous buffer
            if (!isContiguous(size, stride)) {
                return materializeStrided(slice, elemSize, st, size, stride, 0);
            }

            return blobToTensor(slice, size, st);
        }

        private static boolean isContiguous(long[] size, long[] stride) {
            if (size == null || size.length == 0) return true;
            if (stride == null || stride.length != size.length) return false;
            long expect = 1;
            for (int i = size.length - 1; i >= 0; i--) {
                if (size[i] == 0) return true;
                if (stride[i] != expect) return false;
                expect *= size[i];
            }
            return true;
        }

        private Tensor materializeStrided(byte[] storage, int elemSize, ScalarType st,
                                          long[] size, long[] stride, long storageOffsetElems)
                throws IOException {
            long numel = 1;
            for (long d : size) numel *= Math.max(d, 0);
            byte[] contig = new byte[(int) (numel * elemSize)];
            long[] idx = new long[size.length];
            for (long linear = 0; linear < numel; linear++) {
                // compute multi-index from linear (row-major)
                long rem = linear;
                for (int d = size.length - 1; d >= 0; d--) {
                    long dim = Math.max(size[d], 1);
                    idx[d] = rem % dim;
                    rem /= dim;
                }
                long srcElem = storageOffsetElems;
                for (int d = 0; d < size.length; d++) srcElem += idx[d] * stride[d];
                int src = (int) (srcElem * elemSize);
                int dst = (int) (linear * elemSize);
                if (src + elemSize <= storage.length) {
                    System.arraycopy(storage, src, contig, dst, elemSize);
                }
            }
            return blobToTensor(contig, size, st);
        }

        private Tensor blobToTensor(byte[] raw, long[] shape, ScalarType st) {
            TensorOptions opts = new TensorOptions(st);
            // Empty tensors: no payload (3-arg form avoids long... varargs clash)
            if (raw.length == 0) {
                return torch.empty(shape, opts, null);
            }
            BytePointer ptr = new BytePointer(raw.length);
            ptr.put(raw);
            // from_blob shares memory; clone so the Tensor owns a durable copy
            Tensor view = torch.from_blob(ptr, shape, opts);
            Tensor owned = view.clone();
            view.close();
            return owned;
        }

        private static ScalarType storageTypeToScalar(String typeName) throws IOException {
            if (typeName == null) throw new IOException("null storage type");
            // Torch ZIP GLOBAL is typically "torch" + "FloatStorage"
            String n = typeName;
            int dot = n.lastIndexOf('.');
            if (dot >= 0) n = n.substring(dot + 1);
            if (n.endsWith("Storage")) n = n.substring(0, n.length() - "Storage".length());
            switch (n) {
                case "Float": return ScalarType.Float;
                case "Double": return ScalarType.Double;
                case "Half": return ScalarType.Half;
                case "BFloat16": return ScalarType.BFloat16;
                case "Long": return ScalarType.Long;
                case "Int": return ScalarType.Int;
                case "Short": return ScalarType.Short;
                case "Char": return ScalarType.Char;   // int8
                case "Byte": return ScalarType.Byte;   // uint8
                case "Bool": return ScalarType.Bool;
                case "ComplexFloat": return ScalarType.ComplexFloat;
                case "ComplexDouble": return ScalarType.ComplexDouble;
                default:
                    // case-insensitive fallback
                    String low = n.toLowerCase(Locale.ROOT);
                    switch (low) {
                        case "float": case "float32": return ScalarType.Float;
                        case "double": case "float64": return ScalarType.Double;
                        case "half": case "float16": return ScalarType.Half;
                        case "bfloat16": return ScalarType.BFloat16;
                        case "long": case "int64": return ScalarType.Long;
                        case "int": case "int32": return ScalarType.Int;
                        case "short": case "int16": return ScalarType.Short;
                        case "char": case "int8": return ScalarType.Char;
                        case "byte": case "uint8": return ScalarType.Byte;
                        case "bool": case "boolean": return ScalarType.Bool;
                        default:
                            throw new IOException("unknown torch storage type: " + typeName);
                    }
            }
        }

        private static void swapEndianInPlace(byte[] a, int elemSize) {
            for (int i = 0; i + elemSize <= a.length; i += elemSize) {
                for (int j = 0; j < elemSize / 2; j++) {
                    byte tmp = a[i + j];
                    a[i + j] = a[i + elemSize - 1 - j];
                    a[i + elemSize - 1 - j] = tmp;
                }
            }
        }

        private int popMark() throws IOException {
            if (marks.isEmpty()) throw new IOException("MARK stack underflow");
            return marks.remove(marks.size() - 1);
        }

        private String readLine() throws IOException {
            StringBuilder sb = new StringBuilder();
            while (true) {
                int c = in.read();
                if (c < 0) throw new EOFException("EOF in pickle line");
                if (c == '\n') break;
                if (c != '\r') sb.append((char) c);
            }
            return sb.toString();
        }

        private static Object[] asArray(Object o) {
            if (o == null) return new Object[0];
            if (o instanceof Object[]) return (Object[]) o;
            if (o instanceof List) return ((List<?>) o).toArray();
            return new Object[]{o};
        }

        private static long toLong(Object o) {
            if (o instanceof Number) return ((Number) o).longValue();
            if (o instanceof Boolean) return ((Boolean) o) ? 1L : 0L;
            return Long.parseLong(String.valueOf(o));
        }

        private static long[] toLongArray(Object o) {
            if (o == null) return new long[0];
            if (o instanceof long[]) return (long[]) o;
            if (o instanceof int[]) {
                int[] a = (int[]) o;
                long[] out = new long[a.length];
                for (int i = 0; i < a.length; i++) out[i] = a[i];
                return out;
            }
            if (o instanceof Object[]) {
                Object[] a = (Object[]) o;
                long[] out = new long[a.length];
                for (int i = 0; i < a.length; i++) out[i] = toLong(a[i]);
                return out;
            }
            if (o instanceof List) {
                List<?> L = (List<?>) o;
                long[] out = new long[L.size()];
                for (int i = 0; i < L.size(); i++) out[i] = toLong(L.get(i));
                return out;
            }
            return new long[]{toLong(o)};
        }

        private static Object boxLong(long v) {
            if (v >= Integer.MIN_VALUE && v <= Integer.MAX_VALUE) return Integer.valueOf((int) v);
            return Long.valueOf(v);
        }

        private static long bytesToLongLE(byte[] raw) {
            long v = 0;
            int n = Math.min(raw.length, 8);
            for (int i = 0; i < n; i++) v |= ((long) (raw[i] & 0xFF)) << (8 * i);
            // sign-extend if smaller and high bit set
            if (raw.length > 0 && raw.length < 8 && (raw[raw.length - 1] & 0x80) != 0) {
                for (int i = raw.length; i < 8; i++) v |= 0xFFL << (8 * i);
            }
            return v;
        }
    }

    private static final class GlobalRef {
        final String mod;
        final String name;
        final boolean stub;
        GlobalRef(String mod, String name) { this(mod, name, false); }
        GlobalRef(String mod, String name, boolean stub) {
            this.mod = mod;
            this.name = name;
            this.stub = stub;
        }
        @Override public String toString() {
            return (stub ? "stub:" : "") + mod + "." + name;
        }
    }

    /** Placeholder for non-tensor pickle objects (optimizer, custom classes). */
    private static final class StubObject {
        final String type;
        final Object args;
        StubObject(String type, Object args) {
            this.type = type;
            this.args = args;
        }
        @Override public String toString() { return "Stub(" + type + ")"; }
    }

    private static final class StorageView {
        final String typeName;
        final byte[] bytes;
        StorageView(String typeName, byte[] bytes) {
            this.typeName = typeName;
            this.bytes = bytes == null ? new byte[0] : bytes;
        }
    }
}
