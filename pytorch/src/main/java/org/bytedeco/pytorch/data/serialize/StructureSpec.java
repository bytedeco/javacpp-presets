package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.autograd.*;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import org.bytedeco.javacpp.BoolPointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.DoubleOptional;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDict;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDictItem;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModuleAsHelper;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm3dImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.GroupNormImpl;
import org.bytedeco.pytorch.nn.modules.IdentityImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LeakyReLUImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLU6Impl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.GELUImpl;
import org.bytedeco.pytorch.nn.modules.SiLUImpl;
import org.bytedeco.pytorch.nn.modules.SigmoidImpl;
import org.bytedeco.pytorch.nn.modules.SoftmaxImpl;
import org.bytedeco.pytorch.nn.modules.TanhImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.nn.options.DropoutOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;
import org.bytedeco.pytorch.nn.options.SoftmaxOptions;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Schema-v2 precise module structure (from Python {@code dump_module_structure.py}).
 *
 * <p>Unlike the compact {@code path=KIND;…} safetensors meta, v2 carries the
 * full tree: container kinds (Sequential / ModuleList / ModuleDict), ordered
 * children, and leaf hyperparameters (Dropout p, Softmax dim, Linear bias, …).
 * This is the source of truth for precise rebuild — pure state_dict is not enough.
 *
 * <pre>{@code
 *   StructureSpec spec = StructureSpec.load(Path.of("model.structure.json"));
 *   WeightBagModule bag = StructureModuleBuilder.build(spec, stateDict, true);
 * }</pre>
 */
public final class StructureSpec {
    public static final int VERSION = 2;

    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    /** One node in the module tree. */
    public static final class Node {
        public final String kind;
        public final String className;
        public final List<String> children;
        public final Map<String, Object> hyper;
        public final List<String> ownParameters;
        public final List<String> ownBuffers;

        public Node(String kind, String className, List<String> children,
                    Map<String, Object> hyper, List<String> ownParameters,
                    List<String> ownBuffers) {
            this.kind = kind == null ? "COMPOSITE" : kind;
            this.className = className;
            this.children = children == null
                    ? Collections.emptyList()
                    : Collections.unmodifiableList(new ArrayList<>(children));
            this.hyper = hyper == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(new LinkedHashMap<>(hyper));
            this.ownParameters = ownParameters == null
                    ? Collections.emptyList()
                    : Collections.unmodifiableList(new ArrayList<>(ownParameters));
            this.ownBuffers = ownBuffers == null
                    ? Collections.emptyList()
                    : Collections.unmodifiableList(new ArrayList<>(ownBuffers));
        }

        public boolean isContainer() {
            String k = kind.toUpperCase(Locale.ROOT);
            return "SEQUENTIAL".equals(k) || "MODULE_LIST".equals(k)
                    || "MODULE_DICT".equals(k) || "CONTAINER".equals(k)
                    || "COMPOSITE".equals(k) || k.startsWith("COMPOSITE:");
        }

        public boolean isSequential() {
            return "SEQUENTIAL".equalsIgnoreCase(kind);
        }

        public boolean isModuleList() {
            return "MODULE_LIST".equalsIgnoreCase(kind);
        }

        public double hyperDouble(String key, double def) {
            Object v = hyper.get(key);
            if (v instanceof Number) return ((Number) v).doubleValue();
            if (v instanceof String) {
                try { return Double.parseDouble((String) v); } catch (Exception ignored) {}
            }
            return def;
        }

        public long hyperLong(String key, long def) {
            Object v = hyper.get(key);
            if (v instanceof Number) return ((Number) v).longValue();
            if (v instanceof String) {
                try { return Long.parseLong((String) v); } catch (Exception ignored) {}
            }
            return def;
        }

        public boolean hyperBool(String key, boolean def) {
            Object v = hyper.get(key);
            if (v instanceof Boolean) return (Boolean) v;
            if (v instanceof String) return Boolean.parseBoolean((String) v);
            if (v instanceof Number) return ((Number) v).intValue() != 0;
            return def;
        }

        public long[] hyperLongArray(String key) {
            Object v = hyper.get(key);
            if (v instanceof List) {
                List<?> list = (List<?>) v;
                long[] out = new long[list.size()];
                for (int i = 0; i < list.size(); i++) {
                    Object e = list.get(i);
                    out[i] = e instanceof Number ? ((Number) e).longValue() : Long.parseLong(String.valueOf(e));
                }
                return out;
            }
            if (v instanceof Number) return new long[]{((Number) v).longValue()};
            return null;
        }
    }

    public final int version;
    public final String root;
    /** path → node; empty string is root. */
    public final Map<String, Node> nodes;
    public final List<String> parameters;
    public final List<String> buffers;

    public StructureSpec(int version, String root, Map<String, Node> nodes,
                         List<String> parameters, List<String> buffers) {
        this.version = version;
        this.root = root;
        this.nodes = Collections.unmodifiableMap(new LinkedHashMap<>(nodes));
        this.parameters = parameters == null
                ? Collections.emptyList()
                : Collections.unmodifiableList(new ArrayList<>(parameters));
        this.buffers = buffers == null
                ? Collections.emptyList()
                : Collections.unmodifiableList(new ArrayList<>(buffers));
    }

    public Node rootNode() {
        return nodes.get("");
    }

    public Node node(String path) {
        return nodes.get(path);
    }

    public boolean hasNode(String path) {
        return nodes.containsKey(path);
    }

    /** Compact path→kind tokens for legacy encodeStructureMeta interop. */
    public Map<String, String> toCompactMeta() {
        Map<String, String> out = new LinkedHashMap<>();
        for (Map.Entry<String, Node> e : nodes.entrySet()) {
            if (e.getKey() == null || e.getKey().isEmpty()) continue;
            Node n = e.getValue();
            String token = n.kind;
            if ("DROPOUT".equalsIgnoreCase(n.kind) && n.hyper.containsKey("p")) {
                token = "DROPOUT:" + n.hyperDouble("p", 0.5);
            } else if ("SOFTMAX".equalsIgnoreCase(n.kind) && n.hyper.containsKey("dim")) {
                token = "SOFTMAX:" + n.hyperLong("dim", -1);
            } else if ("LEAKY_RELU".equalsIgnoreCase(n.kind) && n.hyper.containsKey("negative_slope")) {
                token = "LEAKY_RELU:" + n.hyperDouble("negative_slope", 0.01);
            } else if (n.kind != null && n.kind.toUpperCase(Locale.ROOT).startsWith("COMPOSITE")) {
                token = "COMPOSITE:" + (n.className != null ? n.className : "Unknown");
            }
            out.put(e.getKey(), token);
        }
        return out;
    }

    // ---- pure-Java dump from a live Module tree ---------------------------------

    /**
     * Walk a live {@link Module} tree (e.g. {@link WeightBagModule}) and emit a
     * schema-v2 {@link StructureSpec} entirely in Java — no Python required.
     *
     * <p>Uses {@link ModuleAsHelper#recover} + {@code instanceof} for typed leaves,
     * and reads hypers from {@code options()} / weight shapes when available.
     * Composite containers keep their Java simple name as {@code class_name}.
     *
     * <pre>{@code
     *   WeightBagModule bag = WeightBagModule.fromPythonPthPrecise(pth, structure);
     *   StructureSpec.dump(bag, Path.of("roundtrip.structure.json"));
     *   // or:
     *   StructureSpec spec = StructureSpec.fromModule(bag);
     *   spec.save(Path.of("model.structure.json"));
     * }</pre>
     */
    public static StructureSpec fromModule(Module module) {
        return fromModule(module, null);
    }

    /**
     * @param rootName optional override for root class name (default: Java simple name)
     */
    public static StructureSpec fromModule(Module module, String rootName) {
        Objects.requireNonNull(module, "module");
        Map<String, Node> nodes = new LinkedHashMap<>();
        String root = rootName;
        if (root == null || root.isEmpty()) {
            root = simpleName(module);
            if (root == null || root.isEmpty() || "Module".equals(root)) {
                root = "WeightBagModule";
            }
        }
        walkModule(module, "", nodes);
        // Ensure root node exists
        if (!nodes.containsKey("")) {
            List<String> kids = childNames(module);
            nodes.put("", new Node("CONTAINER", root, kids,
                    Collections.emptyMap(), Collections.emptyList(), Collections.emptyList()));
        } else {
            // overwrite root class_name if we have a better name
            Node rn = nodes.get("");
            if (rn != null && (rn.className == null || "Module".equals(rn.className)
                    || "WeightBagModule".equals(rn.className))) {
                nodes.put("", new Node(rn.kind, root, rn.children, rn.hyper,
                        rn.ownParameters, rn.ownBuffers));
            }
        }
        List<String> parameters = new ArrayList<>();
        List<String> buffers = new ArrayList<>();
        collectParamBufferKeys(module, parameters, buffers);
        return new StructureSpec(VERSION, root, nodes, parameters, buffers);
    }

    /** Dump structure JSON next to a path (creates parent dirs). */
    public static StructureSpec dump(Module module, Path out) throws IOException {
        StructureSpec spec = fromModule(module);
        if (out.getParent() != null) Files.createDirectories(out.getParent());
        spec.save(out);
        return spec;
    }

    public static StructureSpec dump(Module module, File out) throws IOException {
        return dump(module, out.toPath());
    }

    public static StructureSpec dump(Module module, String out) throws IOException {
        return dump(module, Path.of(out));
    }

    /**
     * Suggest sibling structure path for a .pth:
     * {@code model_state_dict.pth} → {@code model.structure.json},
     * {@code model.pth} → {@code model.structure.json}.
     */
    public static Path defaultStructurePath(Path pth) {
        if (pth == null) return Path.of("module.structure.json");
        String name = pth.getFileName().toString();
        String stem = name;
        int dot = name.lastIndexOf('.');
        if (dot > 0) stem = name.substring(0, dot);
        if (stem.endsWith("_state_dict")) {
            stem = stem.substring(0, stem.length() - "_state_dict".length());
        }
        Path dir = pth.getParent() != null ? pth.getParent() : Path.of(".");
        return dir.resolve(stem + ".structure.json");
    }

    // ---- locked entry: ONLY Python torch.save ZIP .pth --------------------------

    /**
     * <b>Locked API:</b> export schema-v2 structure from an original
     * <b>Python</b> {@code torch.save} {@code .pth}/{@code .pt} (ZIP pickle).
     *
     * <p>Hard constraints (enforced):
     * <ul>
     *   <li>file must exist and pass {@link TorchPthReader#isZipTorch}</li>
     *   <li>format must be {@link ModelWeights.Format#TORCH_PTH_ZIP}
     *       (rejects safetensors / unknown / non-ZIP)</li>
     *   <li>weights are read <b>only</b> via {@link TorchPthReader#loadStateDict}
     *       — never {@link NativeModuleIO}, never {@code Module.load}</li>
     *   <li>does not accept a pre-built JavaCPP Module as input</li>
     * </ul>
     *
     * <p>{@code structureOrNull}:
     * <ul>
     *   <li>non-null existing schema-v2 JSON → precise rebuild then dump</li>
     *   <li>{@code null} → auto-discover sibling {@code *.structure.json};
     *       if still missing, heuristic rebuild (degraded) then dump</li>
     * </ul>
     *
     * <pre>{@code
     *   // precise (recommended when a prior structure exists)
     *   StructureSpec s = StructureSpec.fromPythonPth(
     *       new File("dssm_1pct_state_dict.pth"),
     *       new File("dssm_1pct.structure.json"));
     *   s.save(new File("from_java.structure.json"));
     *
     *   // or dump in one call:
     *   StructureSpec.dumpFromPythonPth(pth, structureOrNull, outJson);
     * }</pre>
     *
     * @param pth             Python {@code torch.save} ZIP checkpoint
     * @param structureOrNull optional schema-v2 JSON for precise topology
     * @return structure snapshot of the Module rebuilt from that Python pth
     * @throws IOException if file is not a Python ZIP torch checkpoint, or empty
     */
    public static StructureSpec fromPythonPth(File pth, File structureOrNull) throws IOException {
        Objects.requireNonNull(pth, "pth");
        requirePythonTorchPth(pth);

        Map<String, org.bytedeco.pytorch.Tensor> sd = TorchPthReader.loadStateDict(pth);
        if (sd == null || sd.isEmpty()) {
            throw new IOException("fromPythonPth: no tensors in Python pth: " + pth);
        }

        File structure = structureOrNull;
        if (structure == null || !structure.isFile()) {
            Path sibling = findSibling(pth.toPath());
            if (sibling != null) structure = sibling.toFile();
        }

        WeightBagModule bag;
        if (structure != null && structure.isFile()) {
            // Precise: structure + TorchPthReader weights only
            StructureSpec seed = load(structure);
            bag = StructureModuleBuilder.build(seed, sd, /*requiresGrad=*/false, /*strict=*/false);
        } else {
            // Degraded: pure state_dict rebuild (warn — param-free may be imperfect)
            System.err.println("[StructureSpec.fromPythonPth] WARNING: no structure.json for "
                    + pth.getName()
                    + " — heuristic rebuild; pass structureOrNull for precise export.");
            bag = new WeightBagModule(sd, /*requiresGrad=*/false, /*clone=*/true, /*typed=*/true, null);
        }

        // Mark provenance so dumps are auditable
        StructureSpec out = fromModule(bag);
        // Rebuild with root name preference from seed if any
        String rootName = bag.getClass().getSimpleName();
        if (structure != null && structure.isFile()) {
            try {
                StructureSpec seed = load(structure);
                if (seed.root != null && !seed.root.isEmpty()) rootName = seed.root;
            } catch (Exception ignored) {}
        }
        return fromModule(bag, rootName);
    }

    public static StructureSpec fromPythonPth(Path pth, Path structureOrNull) throws IOException {
        File s = structureOrNull == null ? null : structureOrNull.toFile();
        return fromPythonPth(pth.toFile(), s);
    }

    public static StructureSpec fromPythonPth(String pth, String structureOrNull) throws IOException {
        File s = structureOrNull == null ? null : new File(structureOrNull);
        return fromPythonPth(new File(pth), s);
    }

    /** {@link #fromPythonPth(File, File)} with auto-discovered sibling structure only. */
    public static StructureSpec fromPythonPth(File pth) throws IOException {
        return fromPythonPth(pth, null);
    }

    public static StructureSpec fromPythonPth(Path pth) throws IOException {
        return fromPythonPth(pth.toFile(), null);
    }

    public static StructureSpec fromPythonPth(String pth) throws IOException {
        return fromPythonPth(new File(pth), null);
    }

    /**
     * Locked one-shot: read Python pth → rebuild → write {@code outStructureJson}.
     * Same constraints as {@link #fromPythonPth(File, File)}.
     */
    public static StructureSpec dumpFromPythonPth(File pth, File structureOrNull, File outStructureJson)
            throws IOException {
        Objects.requireNonNull(outStructureJson, "outStructureJson");
        StructureSpec spec = fromPythonPth(pth, structureOrNull);
        Path out = outStructureJson.toPath();
        if (out.getParent() != null) Files.createDirectories(out.getParent());
        spec.save(out);
        return spec;
    }

    public static StructureSpec dumpFromPythonPth(Path pth, Path structureOrNull, Path outStructureJson)
            throws IOException {
        File s = structureOrNull == null ? null : structureOrNull.toFile();
        return dumpFromPythonPth(pth.toFile(), s, outStructureJson.toFile());
    }

    /**
     * Dump structure next to the Python pth using {@link #defaultStructurePath}.
     * Locked to Python ZIP pth only.
     */
    public static StructureSpec dumpFromPythonPthNextTo(File pth, File structureOrNull)
            throws IOException {
        requirePythonTorchPth(pth);
        Path out = defaultStructurePath(pth.toPath());
        return dumpFromPythonPth(pth, structureOrNull, out.toFile());
    }

    public static StructureSpec dumpFromPythonPthNextTo(File pth) throws IOException {
        return dumpFromPythonPthNextTo(pth, null);
    }

    /**
     * Reject anything that is not an original Python {@code torch.save} ZIP archive.
     * Blocks safetensors, bare unknown files, and (by ZIP check) typical misuse of
     * native archives that are not Python pickle ZIP state_dicts.
     */
    public static void requirePythonTorchPth(File pth) throws IOException {
        Objects.requireNonNull(pth, "pth");
        if (!pth.isFile()) {
            throw new IOException("fromPythonPth: not a file: " + pth);
        }
        // Explicit reject of common non-Python formats
        String name = pth.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".safetensors")) {
            throw new IOException(
                    "fromPythonPth: refused safetensors '" + pth.getName()
                    + "' — locked to original Python torch.save ZIP .pth only "
                    + "(use TorchPthReader path, not safetensors / native Module.load).");
        }
        if (name.endsWith(".javacpp.pt") || name.contains(".javacpp.")) {
            throw new IOException(
                    "fromPythonPth: refused JavaCPP native archive '" + pth.getName()
                    + "' — locked to original Python torch.save ZIP .pth only.");
        }

        ModelWeights.Format fmt;
        try {
            fmt = ModelWeights.detect(pth);
        } catch (Exception e) {
            throw new IOException("fromPythonPth: detect failed for " + pth + ": " + e, e);
        }
        if (fmt == ModelWeights.Format.SAFETENSORS) {
            throw new IOException(
                    "fromPythonPth: file is safetensors, not Python torch ZIP: " + pth);
        }
        if (fmt != ModelWeights.Format.TORCH_PTH_ZIP) {
            // Extra ZIP check for clearer error
            boolean zip;
            try {
                zip = TorchPthReader.isZipTorch(pth);
            } catch (Exception e) {
                zip = false;
            }
            if (!zip) {
                throw new IOException(
                        "fromPythonPth: not a Python torch.save ZIP checkpoint: " + pth
                        + " (detect=" + fmt + "). "
                        + "Only original Python .pth/.pt ZIP state_dict is allowed; "
                        + "NativeModuleIO / Module.load paths are forbidden here.");
            }
        } else {
            // Still require isZipTorch for defense in depth
            if (!TorchPthReader.isZipTorch(pth)) {
                throw new IOException(
                        "fromPythonPth: TORCH_PTH_ZIP detect but isZipTorch=false: " + pth);
            }
        }
    }

    private static void walkModule(Module m, String path, Map<String, Node> nodes) {
        if (m == null) return;
        try {
            if (m.isNull()) return;
        } catch (Throwable ignored) { return; }

        Module typed = recoverTyped(m, null);
        List<String> children = childNames(typed);
        KindHyper kh = classifyWithHyper(typed, path);
        List<String> ownParams = ownParameterNames(typed);
        List<String> ownBufs = ownBufferNames(typed);
        nodes.put(path, new Node(kh.kind, kh.className, children, kh.hyper, ownParams, ownBufs));

        // Recurse
        try {
            StringSharedModuleDict kids = typed.named_children();
            if (kids == null || kids.isNull() || kids.size() == 0) return;
            long n = kids.size();
            for (long i = 0; i < n; i++) {
                StringSharedModuleDictItem item = kids.get((int) i);
                if (item == null || item.isNull()) continue;
                String key = item.key() != null ? item.key().getString() : String.valueOf(i);
                Module child = item.value();
                if (child == null) continue;
                try {
                    if (child.isNull()) continue;
                } catch (Throwable ignored) { continue; }
                String childPath = path.isEmpty() ? key : path + "." + key;
                Module childTyped = recoverTyped(child, typed);
                // Skip bogus recoveries that point back at parent
                if (childTyped == typed) childTyped = child;
                walkModule(childTyped, childPath, nodes);
            }
        } catch (Throwable ignored) {}
    }

    private static Module recoverTyped(Module m, Module parent) {
        if (m == null) return null;
        try {
            Module recovered = ModuleAsHelper.recover(m);
            if (recovered != null && recovered != parent) {
                try {
                    if (parent != null && recovered.address() == parent.address()) return m;
                } catch (Throwable ignored) {}
                return recovered;
            }
        } catch (Throwable ignored) {}
        return m;
    }

    private static List<String> childNames(Module m) {
        List<String> out = new ArrayList<>();
        if (m == null) return out;
        try {
            StringSharedModuleDict kids = m.named_children();
            if (kids == null || kids.isNull()) return out;
            long n = kids.size();
            for (long i = 0; i < n; i++) {
                StringSharedModuleDictItem item = kids.get((int) i);
                if (item == null || item.isNull()) continue;
                String key = item.key() != null ? item.key().getString() : String.valueOf(i);
                if (key != null) out.add(key);
            }
        } catch (Throwable ignored) {}
        return out;
    }

    private static List<String> ownParameterNames(Module m) {
        List<String> out = new ArrayList<>();
        if (m == null) return out;
        try {
            StringTensorDict dict = m.named_parameters(/*recurse=*/false);
            if (dict == null || dict.isNull()) return out;
            long n = dict.size();
            for (long i = 0; i < n; i++) {
                StringTensorDictItem item = dict.get(i);
                if (item == null || item.isNull()) continue;
                String name = item.key() != null ? item.key().getString() : null;
                if (name != null) out.add(name);
            }
        } catch (Throwable ignored) {}
        return out;
    }

    private static List<String> ownBufferNames(Module m) {
        List<String> out = new ArrayList<>();
        if (m == null) return out;
        try {
            StringTensorDict dict = m.named_buffers(/*recurse=*/false);
            if (dict == null || dict.isNull()) return out;
            long n = dict.size();
            for (long i = 0; i < n; i++) {
                StringTensorDictItem item = dict.get(i);
                if (item == null || item.isNull()) continue;
                String name = item.key() != null ? item.key().getString() : null;
                if (name != null) out.add(name);
            }
        } catch (Throwable ignored) {}
        return out;
    }

    private static void collectParamBufferKeys(Module m, List<String> parameters, List<String> buffers) {
        if (m == null) return;
        try {
            StringTensorDict dict = m.named_parameters(/*recurse=*/true);
            if (dict != null && !dict.isNull()) {
                long n = dict.size();
                for (long i = 0; i < n; i++) {
                    StringTensorDictItem item = dict.get(i);
                    if (item == null || item.isNull()) continue;
                    String name = item.key() != null ? item.key().getString() : null;
                    if (name != null) parameters.add(name);
                }
            }
        } catch (Throwable ignored) {}
        try {
            StringTensorDict dict = m.named_buffers(/*recurse=*/true);
            if (dict != null && !dict.isNull()) {
                long n = dict.size();
                for (long i = 0; i < n; i++) {
                    StringTensorDictItem item = dict.get(i);
                    if (item == null || item.isNull()) continue;
                    String name = item.key() != null ? item.key().getString() : null;
                    if (name != null) buffers.add(name);
                }
            }
        } catch (Throwable ignored) {}
    }

    private static final class KindHyper {
        final String kind;
        final String className;
        final Map<String, Object> hyper;
        KindHyper(String kind, String className, Map<String, Object> hyper) {
            this.kind = kind;
            this.className = className;
            this.hyper = hyper != null ? hyper : Collections.emptyMap();
        }
    }

    private static KindHyper classifyWithHyper(Module m, String path) {
        Map<String, Object> hyper = new LinkedHashMap<>();
        String cn = simpleName(m);

        if (m instanceof SequentialImpl) {
            return new KindHyper("SEQUENTIAL", "Sequential", hyper);
        }
        if (m instanceof LinearImpl) {
            LinearImpl lin = (LinearImpl) m;
            try {
                LinearOptions opt = lin.options();
                if (opt != null) {
                    hyper.put("in_features", readLong(opt.in_features()));
                    hyper.put("out_features", readLong(opt.out_features()));
                    hyper.put("bias", readBool(opt.bias(), true));
                } else if (lin.weight() != null && lin.weight().defined()) {
                    // fallback from weight shape [out, in]
                    hyper.put("out_features", lin.weight().sizes().get(0));
                    hyper.put("in_features", lin.weight().sizes().get(1));
                    hyper.put("bias", lin.bias() != null && lin.bias().defined());
                }
            } catch (Throwable t) {
                try {
                    if (lin.weight() != null && lin.weight().defined()) {
                        hyper.put("out_features", lin.weight().sizes().get(0));
                        hyper.put("in_features", lin.weight().sizes().get(1));
                        hyper.put("bias", lin.bias() != null && lin.bias().defined());
                    }
                } catch (Throwable ignored) {}
            }
            return new KindHyper("LINEAR", "Linear", hyper);
        }
        if (m instanceof EmbeddingImpl) {
            EmbeddingImpl emb = (EmbeddingImpl) m;
            try {
                EmbeddingOptions opt = emb.options();
                if (opt != null) {
                    hyper.put("num_embeddings", readLong(opt.num_embeddings()));
                    hyper.put("embedding_dim", readLong(opt.embedding_dim()));
                    try {
                        LongOptional pi = opt.padding_idx();
                        if (pi != null && !pi.isNull() && pi.has_value()) {
                            hyper.put("padding_idx", pi.get());
                        }
                    } catch (Throwable ignored) {}
                } else if (emb.weight() != null && emb.weight().defined()) {
                    hyper.put("num_embeddings", emb.weight().sizes().get(0));
                    hyper.put("embedding_dim", emb.weight().sizes().get(1));
                }
            } catch (Throwable t) {
                try {
                    if (emb.weight() != null && emb.weight().defined()) {
                        hyper.put("num_embeddings", emb.weight().sizes().get(0));
                        hyper.put("embedding_dim", emb.weight().sizes().get(1));
                    }
                } catch (Throwable ignored) {}
            }
            return new KindHyper("EMBEDDING", "Embedding", hyper);
        }
        if (m instanceof LayerNormImpl) {
            return new KindHyper("LAYER_NORM", "LayerNorm", hyper);
        }
        if (m instanceof BatchNorm1dImpl) {
            fillBnHyper(((BatchNorm1dImpl) m).options(), hyper);
            return new KindHyper("BATCH_NORM_1D", "BatchNorm1d", hyper);
        }
        if (m instanceof BatchNorm2dImpl) {
            fillBnHyper(((BatchNorm2dImpl) m).options(), hyper);
            return new KindHyper("BATCH_NORM_2D", "BatchNorm2d", hyper);
        }
        if (m instanceof BatchNorm3dImpl) {
            fillBnHyper(((BatchNorm3dImpl) m).options(), hyper);
            return new KindHyper("BATCH_NORM_3D", "BatchNorm3d", hyper);
        }
        if (m instanceof GroupNormImpl) {
            return new KindHyper("GROUP_NORM", "GroupNorm", hyper);
        }
        if (m instanceof DropoutImpl) {
            double p = 0.5;
            boolean inplace = false;
            try {
                DropoutOptions opt = ((DropoutImpl) m).options();
                if (opt != null && opt.p() != null) p = opt.p().get();
                if (opt != null && opt.inplace() != null) inplace = opt.inplace().get();
            } catch (Throwable ignored) {}
            hyper.put("p", p);
            hyper.put("inplace", inplace);
            return new KindHyper("DROPOUT", "Dropout", hyper);
        }
        if (m instanceof SoftmaxImpl) {
            long dim = -1;
            try {
                SoftmaxOptions opt = ((SoftmaxImpl) m).options();
                if (opt != null && opt.dim() != null) dim = opt.dim().get();
            } catch (Throwable ignored) {}
            hyper.put("dim", dim);
            return new KindHyper("SOFTMAX", "Softmax", hyper);
        }
        if (m instanceof ReLUImpl) return new KindHyper("RELU", "ReLU", hyper);
        if (m instanceof ReLU6Impl) return new KindHyper("RELU6", "ReLU6", hyper);
        if (m instanceof LeakyReLUImpl) return new KindHyper("LEAKY_RELU", "LeakyReLU", hyper);
        if (m instanceof GELUImpl) return new KindHyper("GELU", "GELU", hyper);
        if (m instanceof SiLUImpl) return new KindHyper("SILU", "SiLU", hyper);
        if (m instanceof TanhImpl) return new KindHyper("TANH", "Tanh", hyper);
        if (m instanceof SigmoidImpl) return new KindHyper("SIGMOID", "Sigmoid", hyper);
        if (m instanceof IdentityImpl) return new KindHyper("IDENTITY", "Identity", hyper);

        // Fall back to compact token from StateDictModuleBuilder
        String token = null;
        try {
            token = StateDictModuleBuilder.classifyModule(m);
        } catch (Throwable ignored) {}
        if (token != null) {
            String kind = token;
            int colon = token.indexOf(':');
            if (colon > 0) {
                kind = token.substring(0, colon);
                String rest = token.substring(colon + 1);
                if ("DROPOUT".equalsIgnoreCase(kind)) {
                    try { hyper.put("p", Double.parseDouble(rest)); } catch (Exception ignored) {}
                } else if ("SOFTMAX".equalsIgnoreCase(kind)) {
                    try { hyper.put("dim", Long.parseLong(rest)); } catch (Exception ignored) {}
                }
            }
            // MODULE_LIST / MODULE_DICT detection by children all-integer or not
            if ("CONTAINER".equalsIgnoreCase(kind) || token == null) {
                // continue below
            } else if (!"MODULE".equalsIgnoreCase(kind)) {
                return new KindHyper(kind.toUpperCase(Locale.ROOT),
                        cn != null ? cn : kind, hyper);
            }
        }

        // Container heuristics from children names
        List<String> kids = childNames(m);
        if (!kids.isEmpty() && allIntegerIndices(kids)) {
            // Sequential vs ModuleList: SequentialImpl already handled;
            // remaining integer-named containers → MODULE_LIST
            return new KindHyper("MODULE_LIST",
                    cn != null && !cn.equals("Module") ? cn : "ModuleList", hyper);
        }
        if (!kids.isEmpty()) {
            return new KindHyper("COMPOSITE",
                    cn != null ? cn : "Module", hyper);
        }
        // bare leaf-ish module with only parameters
        List<String> ops = ownParameterNames(m);
        if (!ops.isEmpty() && kids.isEmpty()) {
            return new KindHyper("COMPOSITE", cn != null ? cn : "Module", hyper);
        }
        return new KindHyper("COMPOSITE", cn != null ? cn : "Module", hyper);
    }

    private static void fillBnHyper(BatchNormOptions opt, Map<String, Object> hyper) {
        if (opt == null) return;
        try {
            hyper.put("num_features", readLong(opt.num_features()));
            if (opt.eps() != null) hyper.put("eps", opt.eps().get());
            try {
                DoubleOptional mom = opt.momentum();
                if (mom != null && !mom.isNull() && mom.has_value()) {
                    hyper.put("momentum", mom.get());
                }
            } catch (Throwable ignored) {}
            hyper.put("affine", readBool(opt.affine(), true));
            hyper.put("track_running_stats", readBool(opt.track_running_stats(), true));
        } catch (Throwable ignored) {}
    }

    private static long readLong(LongPointer p) {
        if (p == null) return 0;
        try { return p.get(); } catch (Throwable t) { return 0; }
    }

    private static boolean readBool(BoolPointer p, boolean def) {
        if (p == null) return def;
        try {
            // BoolPointer.get() returns boolean on this JavaCPP binding
            return p.get();
        } catch (Throwable t) {
            try { return p.get(0); } catch (Throwable t2) { return def; }
        }
    }

    private static boolean allIntegerIndices(List<String> names) {
        if (names == null || names.isEmpty()) return false;
        for (String n : names) {
            if (n == null || n.isEmpty()) return false;
            for (int i = 0; i < n.length(); i++) {
                char c = n.charAt(i);
                if (c < '0' || c > '9') return false;
            }
        }
        return true;
    }

    private static String simpleName(Module m) {
        if (m == null) return null;
        try {
            String sn = m.getClass().getSimpleName();
            if (sn.endsWith("Impl")) sn = sn.substring(0, sn.length() - 4);
            if (sn.isEmpty() || "Module".equals(sn)) {
                // try C++ name
                try {
                    org.bytedeco.javacpp.BytePointer bp = m.name();
                    if (bp != null && !bp.isNull()) {
                        String raw = bp.getString();
                        if (raw != null && !raw.isEmpty()) {
                            int cc = raw.lastIndexOf("::");
                            if (cc >= 0) raw = raw.substring(cc + 2);
                            if (raw.endsWith("Impl")) raw = raw.substring(0, raw.length() - 4);
                            return raw;
                        }
                    }
                } catch (Throwable ignored) {}
            }
            return sn;
        } catch (Throwable t) {
            return null;
        }
    }

    // ---- load / save ------------------------------------------------------------

    public static StructureSpec load(Path path) throws IOException {
        Objects.requireNonNull(path, "path");
        String text = Files.readString(path, StandardCharsets.UTF_8);
        return parse(text);
    }

    public static StructureSpec load(java.io.File file) throws IOException {
        return load(file.toPath());
    }

    public static StructureSpec parse(String json) {
        Objects.requireNonNull(json, "json");
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        int version = root.has("version") ? root.get("version").getAsInt() : 1;
        if (version < 2) {
            throw new IllegalArgumentException(
                    "StructureSpec requires version>=2 (got " + version
                    + "). Re-export with StructureSpec.dump(module, path) "
                    + "or scripts/dump_module_structure.py");
        }
        String rootName = root.has("root") && !root.get("root").isJsonNull()
                ? root.get("root").getAsString() : "Module";

        Map<String, Node> nodes = new LinkedHashMap<>();
        if (root.has("nodes") && root.get("nodes").isJsonObject()) {
            JsonObject nodesObj = root.getAsJsonObject("nodes");
            for (Map.Entry<String, JsonElement> e : nodesObj.entrySet()) {
                nodes.put(e.getKey(), parseNode(e.getValue().getAsJsonObject()));
            }
        }
        List<String> parameters = stringList(root, "parameters");
        List<String> buffers = stringList(root, "buffers");
        if (!nodes.containsKey("")) {
            // synthesize root if missing
            nodes.put("", new Node("CONTAINER", rootName, Collections.emptyList(),
                    Collections.emptyMap(), Collections.emptyList(), Collections.emptyList()));
        }
        return new StructureSpec(version, rootName, nodes, parameters, buffers);
    }

    private static Node parseNode(JsonObject o) {
        String kind = o.has("kind") ? o.get("kind").getAsString() : "COMPOSITE";
        String className = o.has("class_name") && !o.get("class_name").isJsonNull()
                ? o.get("class_name").getAsString() : null;
        List<String> children = stringList(o, "children");
        Map<String, Object> hyper = new LinkedHashMap<>();
        if (o.has("hyper") && o.get("hyper").isJsonObject()) {
            hyper = GSON.fromJson(o.get("hyper"), new TypeToken<Map<String, Object>>(){}.getType());
            if (hyper == null) hyper = new LinkedHashMap<>();
        }
        List<String> ownParams = stringList(o, "own_parameters");
        List<String> ownBufs = stringList(o, "own_buffers");
        return new Node(kind, className, children, hyper, ownParams, ownBufs);
    }

    private static List<String> stringList(JsonObject o, String key) {
        List<String> out = new ArrayList<>();
        if (o == null || !o.has(key) || o.get(key).isJsonNull()) return out;
        if (!o.get(key).isJsonArray()) return out;
        for (JsonElement el : o.getAsJsonArray(key)) {
            if (el != null && !el.isJsonNull()) out.add(el.getAsString());
        }
        return out;
    }

    public void save(Path path) throws IOException {
        Map<String, Object> doc = new LinkedHashMap<>();
        doc.put("version", version);
        doc.put("root", root);
        Map<String, Object> nodesOut = new LinkedHashMap<>();
        for (Map.Entry<String, Node> e : nodes.entrySet()) {
            Node n = e.getValue();
            Map<String, Object> no = new LinkedHashMap<>();
            no.put("kind", n.kind);
            if (n.className != null) no.put("class_name", n.className);
            if (!n.children.isEmpty()) no.put("children", n.children);
            if (!n.hyper.isEmpty()) no.put("hyper", n.hyper);
            if (!n.ownParameters.isEmpty()) no.put("own_parameters", n.ownParameters);
            if (!n.ownBuffers.isEmpty()) no.put("own_buffers", n.ownBuffers);
            nodesOut.put(e.getKey(), no);
        }
        doc.put("nodes", nodesOut);
        doc.put("parameters", parameters);
        doc.put("buffers", buffers);
        Files.writeString(path, GSON.toJson(doc) + "\n", StandardCharsets.UTF_8);
    }

    /**
     * Resolve sibling structure file for a .pth:
     * {@code model.pth} → {@code model.structure.json} or {@code model_structure.json}.
     */
    public static Path findSibling(Path pth) {
        if (pth == null) return null;
        String name = pth.getFileName().toString();
        String stem = name;
        int dot = name.lastIndexOf('.');
        if (dot > 0) stem = name.substring(0, dot);
        Path dir = pth.getParent() != null ? pth.getParent() : Path.of(".");
        Path[] candidates = {
            dir.resolve(stem + ".structure.json"),
            dir.resolve(stem + "_structure.json"),
            dir.resolve("module_structure.json"),
            dir.resolve("structure.json"),
        };
        for (Path c : candidates) {
            if (Files.isRegularFile(c)) return c;
        }
        // e.g. dssm_1pct_state_dict.pth → dssm_1pct.structure.json
        if (stem.endsWith("_state_dict")) {
            String s2 = stem.substring(0, stem.length() - "_state_dict".length());
            Path c = dir.resolve(s2 + ".structure.json");
            if (Files.isRegularFile(c)) return c;
            c = dir.resolve(s2 + "_structure.json");
            if (Files.isRegularFile(c)) return c;
        }
        return null;
    }

    @Override
    public String toString() {
        return "StructureSpec{version=" + version + ", root=" + root
                + ", nodes=" + nodes.size() + ", parameters=" + parameters.size() + "}";
    }
}
