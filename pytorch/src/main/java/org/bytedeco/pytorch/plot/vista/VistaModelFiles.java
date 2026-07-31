package org.bytedeco.pytorch.plot.vista;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.serialize.ModelWeights;
import org.bytedeco.pytorch.data.serialize.NativeModuleIO;
import org.bytedeco.pytorch.data.serialize.StructureModuleBuilder;
import org.bytedeco.pytorch.data.serialize.StructureSpec;
import org.bytedeco.pytorch.data.serialize.WeightBagModule;
import org.bytedeco.pytorch.inductor.AOTIModelPackageLoader;
import org.bytedeco.pytorch.nn.Module;

/**
 * Load model / weight files into something {@link Vista} can visualise.
 *
 * <p>Supported formats (all via existing jnitorch serializers — no invented IO):
 * <ul>
 *   <li>{@code .safetensors} → {@link SafeTensors#toModule} / {@link WeightBagModule}</li>
 *   <li>Python {@code .pth}/{@code .pt} (ZIP torch) → {@link WeightBagModule#fromPythonPth}
 *       (uses sibling {@code *.structure.json} when present for precise topology)</li>
 *   <li>JavaCPP / LibTorch {@code .pt} archive → {@link WeightBagModule#fromJavacppPth}
 *       or {@link NativeModuleIO#load} into a pre-built module</li>
 *   <li>{@code *.structure.json} alone → structure-only graph (no live forward)</li>
 *   <li>AOTI package directory / archive → {@link AOTIModelPackageLoader}
 *       metadata + call-spec graph (constants as Parameter-like nodes)</li>
 * </ul>
 *
 * <pre>
 *   // Auto: detect format, build Module when possible, else structure graph
 *   TraceGraph g = Vista.traceFile("model.safetensors");
 *   TraceGraph g2 = Vista.traceFile("model.pth", sampleInput);
 *   TraceGraph g3 = Vista.traceFile("model.structure.json"); // topology only
 * </pre>
 */
public final class VistaModelFiles {
    private VistaModelFiles() {}

    public enum Kind {
        SAFETENSORS,
        PYTHON_PTH,
        JAVACPP_PTH,
        STRUCTURE_JSON,
        AOTI_PACKAGE,
        UNKNOWN
    }

    /** Result of opening a model file for visualisation. */
    public static final class Loaded {
        public final Kind kind;
        public final Module module;           // nullable for structure-only / AOT metadata
        public final StructureSpec structure; // nullable
        public final AOTIModelPackageLoader aot; // nullable — caller must close/deallocate carefully
        public final Map<String, Tensor> weights; // nullable snapshot
        public final String sourcePath;
        public final String note;

        public Loaded(Kind kind, Module module, StructureSpec structure,
                      AOTIModelPackageLoader aot, Map<String, Tensor> weights,
                      String sourcePath, String note) {
            this.kind = kind;
            this.module = module;
            this.structure = structure;
            this.aot = aot;
            this.weights = weights;
            this.sourcePath = sourcePath;
            this.note = note;
        }

        public boolean hasRunnableModule() {
            return module != null && !module.isNull();
        }

        public boolean hasStructure() {
            return structure != null && structure.nodes != null && !structure.nodes.isEmpty();
        }
    }

    public static Kind detect(File file) throws IOException {
        if (file == null) return Kind.UNKNOWN;
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".structure.json") || name.endsWith(".json") && name.contains("structure")) {
            return Kind.STRUCTURE_JSON;
        }
        if (name.endsWith(".safetensors")) return Kind.SAFETENSORS;
        if (file.isDirectory()) {
            // Heuristic: AOTI package often has data.pt / metadata / so libs
            if (looksLikeAotiPackage(file.toPath())) return Kind.AOTI_PACKAGE;
            return Kind.UNKNOWN;
        }
        // AOTI single-file package (zip-like path ending without .pth)
        if (name.endsWith(".pt2") || name.contains("aoti") || name.endsWith(".aoti")) {
            return Kind.AOTI_PACKAGE;
        }
        ModelWeights.Format fmt = ModelWeights.detect(file);
        switch (fmt) {
            case SAFETENSORS:
                return Kind.SAFETENSORS;
            case TORCH_PTH_ZIP:
                return Kind.PYTHON_PTH;
            default:
                // Try structure json by content
                if (name.endsWith(".json")) return Kind.STRUCTURE_JSON;
                // Native libtorch archive often .pt but not ZIP — treat as javacpp
                if (name.endsWith(".pt") || name.endsWith(".pth")) return Kind.JAVACPP_PTH;
                return Kind.UNKNOWN;
        }
    }

    private static boolean looksLikeAotiPackage(Path dir) {
        try {
            if (!Files.isDirectory(dir)) return false;
            boolean hasData = Files.exists(dir.resolve("data.pt"))
                    || Files.exists(dir.resolve("data"))
                    || Files.list(dir).anyMatch(p -> {
                        String n = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        return n.endsWith(".so") || n.contains("model") || n.equals("metadata.json");
                    });
            return hasData;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Open a model file and produce the best available Module / structure.
     * Does <em>not</em> run a forward pass — pair with {@link Vista#trace} or
     * {@link StructureGraphBuilder}.
     */
    public static Loaded open(File file) throws IOException {
        return open(file, true);
    }

    public static Loaded open(String path) throws IOException {
        return open(new File(path), true);
    }

    public static Loaded open(File file, boolean requiresGrad) throws IOException {
        if (file == null) throw new IllegalArgumentException("file is null");
        if (!file.exists()) throw new IOException("not found: " + file);

        Kind kind = detect(file);
        String path = file.getAbsolutePath();

        switch (kind) {
            case STRUCTURE_JSON: {
                StructureSpec spec = StructureSpec.load(file);
                Module mod = null;
                String note = "structure-only (no weights bound)";
                Path parent = file.toPath().getParent();
                String base = stripStructureSuffix(file.getName());
                if (parent != null) {
                    File st = parent.resolve(base + ".safetensors").toFile();
                    File pth = parent.resolve(base + ".pth").toFile();
                    File pt = parent.resolve(base + ".pt").toFile();
                    try {
                        if (pth.isFile()) {
                            mod = WeightBagModule.fromPythonPthPrecise(pth, file, requiresGrad);
                            note = "structure.json + sibling python pth (precise)";
                        } else if (pt.isFile()) {
                            try {
                                mod = WeightBagModule.fromPythonPthPrecise(pt, file, requiresGrad);
                                note = "structure.json + sibling .pt (precise)";
                            } catch (Throwable t) {
                                // may be javacpp native archive — structure-only module
                                mod = StructureModuleBuilder.buildEmpty(spec);
                                note = "structure.json + empty precise module (.pt not python ZIP): " + t.getMessage();
                            }
                        } else if (st.isFile()) {
                            Map<String, Tensor> weights = SafeTensors.loadAsTensors(st, true);
                            StructureSpec enriched = enrichStructureFromWeights(spec, weights);
                            mod = StructureModuleBuilder.build(enriched, weights, requiresGrad);
                            spec = enriched;
                            note = "structure.json + sibling safetensors → StructureModuleBuilder";
                        }
                    } catch (Throwable t) {
                        note = "structure-only; weight bind failed: " + t.getMessage();
                    }
                }
                return new Loaded(kind, mod, spec, null, null, path, note);
            }
            case SAFETENSORS: {
                StructureSpec spec = null;
                Path sib = StructureSpec.findSibling(file.toPath());
                if (sib == null) {
                    String base = stripExt(file.getName());
                    Path cand = file.toPath().resolveSibling(base + ".structure.json");
                    if (Files.isRegularFile(cand)) sib = cand;
                }
                if (sib != null) {
                    try {
                        spec = StructureSpec.load(sib.toFile());
                    } catch (Throwable ignored) {}
                }

                Map<String, Tensor> weights = SafeTensors.loadAsTensors(file, true);
                WeightBagModule bag;
                String note;
                if (spec != null) {
                    try {
                        StructureSpec enriched = enrichStructureFromWeights(spec, weights);
                        bag = StructureModuleBuilder.build(enriched, weights, requiresGrad);
                        spec = enriched;
                        note = "safetensors + structure.json → StructureModuleBuilder (precise)";
                    } catch (Throwable t) {
                        bag = WeightBagModule.fromSafetensors(file, requiresGrad);
                        note = "safetensors → WeightBagModule (precise build failed: "
                                + t.getMessage() + ")";
                    }
                } else {
                    bag = WeightBagModule.fromSafetensors(file, requiresGrad);
                    try {
                        spec = StructureSpec.fromModule(bag);
                    } catch (Throwable ignored) {}
                    note = "safetensors → WeightBagModule (inferred; export *.structure.json for param-free layers)";
                }
                return new Loaded(kind, bag, spec, null, weights, path, note);
            }
            case PYTHON_PTH: {
                WeightBagModule bag = WeightBagModule.fromPythonPth(file, requiresGrad);
                StructureSpec spec = null;
                try {
                    Path sib = StructureSpec.findSibling(file.toPath());
                    if (sib != null) spec = StructureSpec.load(sib.toFile());
                    else spec = StructureSpec.fromModule(bag);
                } catch (Throwable t) {
                    try {
                        spec = StructureSpec.fromModule(bag);
                    } catch (Throwable ignored) {}
                }
                return new Loaded(kind, bag, spec, null, null, path,
                        "python pth → WeightBagModule"
                                + (StructureSpec.findSibling(file.toPath()) != null
                                ? " (precise structure)" : " (inferred)"));
            }
            case JAVACPP_PTH: {
                WeightBagModule bag;
                try {
                    bag = WeightBagModule.fromJavacppPth(file, requiresGrad);
                } catch (IOException e) {
                    // Fall back to python path if ZIP
                    try {
                        bag = WeightBagModule.fromPythonPth(file, requiresGrad);
                    } catch (IOException e2) {
                        throw new IOException("javacpp/python pth load failed for " + file
                                + ": " + e.getMessage() + " / " + e2.getMessage(), e2);
                    }
                }
                StructureSpec spec = null;
                try {
                    spec = StructureSpec.fromModule(bag);
                } catch (Throwable ignored) {}
                return new Loaded(kind, bag, spec, null, null, path,
                        "javacpp/libtorch pth → WeightBagModule");
            }
            case AOTI_PACKAGE: {
                AOTIModelPackageLoader loader = null;
                try {
                    loader = new AOTIModelPackageLoader(file.getAbsolutePath());
                } catch (Throwable t) {
                    throw new IOException("AOTI package open failed for " + file + ": " + t.getMessage(), t);
                }
                return new Loaded(kind, null, null, loader, null, path,
                        "AOTI package (metadata + call_spec; run needs sample inputs)");
            }
            default:
                throw new IOException("Unrecognized model file: " + file
                        + " (expected .safetensors / .pth / .pt / .structure.json / AOTI package)");
        }
    }

    /**
     * Load weights into a pre-built architecture (JavaCPP native module archive).
     */
    public static void loadNativeInto(Module module, File javacppPt) throws IOException {
        NativeModuleIO.load(module, javacppPt);
    }

    /**
     * Run an AOTI package with the given inputs (ownership kept by caller).
     */
    public static TensorVector runAoti(AOTIModelPackageLoader loader, TensorVector inputs) {
        if (loader == null || loader.isNull()) {
            throw new IllegalArgumentException("AOTI loader is null");
        }
        return loader.run(inputs);
    }

    public static List<String> aotiCallSpec(AOTIModelPackageLoader loader) {
        List<String> out = new ArrayList<>();
        if (loader == null || loader.isNull()) return out;
        try {
            org.bytedeco.pytorch.StringVector spec = loader.get_call_spec();
            if (spec == null || spec.isNull()) return out;
            long n = spec.size();
            for (long i = 0; i < n; i++) {
                try {
                    out.add(spec.get(i).getString());
                } catch (Throwable e) {
                    out.add("arg_" + i);
                }
            }
        } catch (Throwable ignored) {}
        return out;
    }

    public static List<String> aotiConstantFqns(AOTIModelPackageLoader loader) {
        List<String> out = new ArrayList<>();
        if (loader == null || loader.isNull()) return out;
        try {
            org.bytedeco.pytorch.StringVector v = loader.get_constant_fqns();
            if (v == null || v.isNull()) return out;
            long n = v.size();
            for (long i = 0; i < n; i++) {
                try {
                    out.add(v.get(i).getString());
                } catch (Throwable e) {
                    out.add("const_" + i);
                }
            }
        } catch (Throwable ignored) {}
        return out;
    }

    private static String stripExt(String name) {
        int d = name.lastIndexOf('.');
        return d > 0 ? name.substring(0, d) : name;
    }

    private static String stripStructureSuffix(String name) {
        String lower = name.toLowerCase(Locale.ROOT);
        if (lower.endsWith(".structure.json")) {
            return name.substring(0, name.length() - ".structure.json".length());
        }
        return stripExt(name);
    }

    /**
     * Fill missing Linear/Embedding hypers from weight tensor shapes so
     * {@link StructureModuleBuilder} can materialise param-free layers even when
     * {@code structure.json} omitted {@code in_features}/{@code out_features}
     * (common for {@link StructureSpec#fromModule} dumps of Sequential children).
     */
    static StructureSpec enrichStructureFromWeights(StructureSpec spec,
                                                    Map<String, Tensor> weights) {
        if (spec == null || spec.nodes == null || weights == null || weights.isEmpty()) {
            return spec;
        }
        Map<String, StructureSpec.Node> enriched = new LinkedHashMap<>();
        for (Map.Entry<String, StructureSpec.Node> e : spec.nodes.entrySet()) {
            String path = e.getKey();
            StructureSpec.Node n = e.getValue();
            if (n == null) {
                enriched.put(path, null);
                continue;
            }
            String kind = n.kind == null ? "" : n.kind.toUpperCase(Locale.ROOT);
            Map<String, Object> hyper = new LinkedHashMap<>();
            if (n.hyper != null) hyper.putAll(n.hyper);

            if ("LINEAR".equals(kind)) {
                if (n.hyperLong("in_features", 0) <= 0 || n.hyperLong("out_features", 0) <= 0) {
                    Tensor w = findWeight(weights, path, "weight");
                    if (w != null && !w.isNull()) {
                        try {
                            long[] sh = w.shape();
                            if (sh != null && sh.length >= 2) {
                                // Linear weight is [out, in]
                                hyper.putIfAbsent("out_features", sh[0]);
                                hyper.putIfAbsent("in_features", sh[1]);
                            }
                        } catch (Throwable ignored) {}
                    }
                    Tensor b = findWeight(weights, path, "bias");
                    if (b != null && !b.isNull()) {
                        hyper.putIfAbsent("bias", true);
                    } else if (!hyper.containsKey("bias")) {
                        hyper.put("bias", false);
                    }
                }
            } else if ("EMBEDDING".equals(kind)) {
                if (n.hyperLong("num_embeddings", 0) <= 0 || n.hyperLong("embedding_dim", 0) <= 0) {
                    Tensor w = findWeight(weights, path, "weight");
                    if (w != null && !w.isNull()) {
                        try {
                            long[] sh = w.shape();
                            if (sh != null && sh.length >= 2) {
                                hyper.putIfAbsent("num_embeddings", sh[0]);
                                hyper.putIfAbsent("embedding_dim", sh[1]);
                            }
                        } catch (Throwable ignored) {}
                    }
                }
            } else if (kind.startsWith("BATCH_NORM") || kind.startsWith("INSTANCE_NORM")) {
                if (n.hyperLong("num_features", 0) <= 0) {
                    Tensor w = findWeight(weights, path, "weight");
                    if (w == null) w = findWeight(weights, path, "running_mean");
                    if (w != null && !w.isNull()) {
                        try {
                            long[] sh = w.shape();
                            if (sh != null && sh.length >= 1) {
                                hyper.putIfAbsent("num_features", sh[0]);
                            }
                        } catch (Throwable ignored) {}
                    }
                }
            }

            enriched.put(path, new StructureSpec.Node(
                    n.kind, n.className, n.children, hyper, n.ownParameters, n.ownBuffers));
        }
        return new StructureSpec(spec.version, spec.root, enriched, spec.parameters, spec.buffers);
    }

    private static Tensor findWeight(Map<String, Tensor> weights, String path, String leaf) {
        if (weights == null) return null;
        String key = (path == null || path.isEmpty()) ? leaf : path + "." + leaf;
        Tensor t = weights.get(key);
        if (t != null) return t;
        // also try bare leaf under path variants
        for (Map.Entry<String, Tensor> e : weights.entrySet()) {
            String k = e.getKey();
            if (k != null && (k.equals(key) || k.endsWith("." + key) || k.endsWith("." + leaf)
                    && (path == null || path.isEmpty() || k.contains(path + ".")))) {
                if (k.equals(key) || k.endsWith("." + leaf) && k.startsWith(path.isEmpty() ? leaf : path + ".")) {
                    return e.getValue();
                }
            }
        }
        return weights.get(leaf);
    }
}
