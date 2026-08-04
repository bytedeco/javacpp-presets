package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.optim.options.*;
import org.bytedeco.pytorch.serialize.*;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.StringTensorDict;
import org.bytedeco.pytorch.StringTensorDictItem;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.safetensors.LoadOptions;
import org.bytedeco.pytorch.data.safetensors.SafeTensors;
import org.bytedeco.pytorch.data.safetensors.ShardedSafeTensors;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModulePrinter;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.io.File;
import java.io.IOException;
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
 * Trainable {@link Module} reconstructed from an arbitrary state-dict
 * ({@code Map&lt;String, Tensor&gt;} from .pth / optional safetensors).
 *
 * <p>Rebuilds a real Module tree that mirrors the Python checkpoint:
 * <ul>
 *   <li>hierarchical keys → nested Modules;</li>
 *   <li>typed leaves: Linear / Embedding / LayerNorm / BatchNorm / Conv…;</li>
 *   <li><b>precise</b> param-free layers (ReLU / Dropout p / Softmax / Sigmoid / …)
 *       when a sibling {@code *.structure.json} (schema v2) is present;</li>
 *   <li>heuristic Sequential gap-fill only as an explicit degraded fallback;</li>
 *   <li>{@code named_parameters(true)} keeps original dotted keys for Adam / freeze / save.</li>
 * </ul>
 *
 * <p>Primary loaders (all return a trainable Module ready for fine-tuning):
 * <pre>{@code
 *   // PRECISE: Python .pth + structure.json  →  Module  (no safetensors)
 *   WeightBagModule bag = WeightBagModule.fromPythonPthPrecise(
 *       new File("model.pth"), new File("model.structure.json"));
 *   // or auto-discover sibling *.structure.json:
 *   WeightBagModule bag = WeightBagModule.fromPythonPth("model.pth");
 *
 *   // Native JavaCPP/LibTorch archive round-trip (architecture must match):
 *   bag.saveNative(new File("model.javacpp.pt"));
 *   WeightBagModule bag2 = StructureModuleBuilder.buildEmpty(spec);
 *   bag2.loadNative(new File("model.javacpp.pt"));
 *
 *   // HuggingFace safetensors (optional mid-format, not required)
 *   WeightBagModule bag = WeightBagModule.fromSafetensors("model.safetensors");
 *
 *   bag.freezePrefix("embedding_layer.");
 *   Adam opt = new Adam(bag.parameters(), new AdamOptions(1e-4));
 * }</pre>
 */
public class WeightBagModule extends Module {

    /** Own / live storage for every leaf tensor (state-dict key → Tensor). */
    private final Map<String, Tensor> ownedParams = new LinkedHashMap<>();

    /** Keep intermediate nested modules and typed leaves reachable from Java. */
    private final Map<String, Module> children = new LinkedHashMap<>();

    /** Inferred layer metadata (path → info), filled by typed builder. */
    private final List<StateDictModuleBuilder.LayerInfo> layerInfos = new ArrayList<>();

    /**
     * Structure metadata used/produced for param-free layers
     * (path → kind token, e.g. {@code "mlp.1" → "RELU"}).
     */
    private final Map<String, String> structureMeta = new LinkedHashMap<>();

    private final boolean requiresGrad;

    /** When true, leaf modules are typed Linear/Embedding/…; else raw bags. */
    private final boolean typed;

    // ---- constructors -------------------------------------------------------

    public WeightBagModule(Map<String, Tensor> weights) {
        this(weights, true, true, true, null);
    }

    public WeightBagModule(Map<String, Tensor> weights, boolean requiresGrad) {
        this(weights, requiresGrad, true, true, null);
    }

    /**
     * @param weights      state-dict style map (keys may contain {@code .})
     * @param requiresGrad set {@code requires_grad} on every registered param
     * @param clone        when true, clone each tensor so the bag owns storage
     *                     (required for training when sources are mmap/from_blob)
     */
    public WeightBagModule(Map<String, Tensor> weights, boolean requiresGrad, boolean clone) {
        this(weights, requiresGrad, clone, true, null);
    }

    /**
     * @param typed when true (default), infer and instantiate real layer types
     *              ({@link LinearImpl}, {@link EmbeddingImpl}, …) with
     *              hyperparameters recovered from shapes; when false, only
     *              nested containers + {@code register_parameter} (legacy bag).
     */
    public WeightBagModule(Map<String, Tensor> weights, boolean requiresGrad,
                           boolean clone, boolean typed) {
        this(weights, requiresGrad, clone, typed, null);
    }

    /**
     * Full constructor with optional structure metadata for param-free layers.
     *
     * @param structureMeta path → kind (e.g. {@code "mlp.1"="RELU"}, {@code "mlp.2"="DROPOUT:0.1"});
     *                      may be null (heuristic gap-fill only)
     */
    public WeightBagModule(Map<String, Tensor> weights, boolean requiresGrad,
                           boolean clone, boolean typed,
                           Map<String, String> structureMeta) {
        super("WeightBagModule");
        this.requiresGrad = requiresGrad;
        this.typed = typed;
        if (structureMeta != null) this.structureMeta.putAll(structureMeta);
        if (weights == null || weights.isEmpty()) return;
        if (typed) {
            buildTyped(weights, requiresGrad, clone, this.structureMeta);
        } else {
            buildBagOnly(weights, requiresGrad, clone);
        }
        // Refresh owned handles from named_parameters so freeze/get see live tensors.
        refreshOwnedFromModule();
        // Capture full structure from typed children we retain (safe instanceof),
        // then merge native walk (best-effort).
        if (typed) {
            for (Map.Entry<String, Module> e : children.entrySet()) {
                String token = StateDictModuleBuilder.classifyModule(e.getValue());
                if (token != null) this.structureMeta.putIfAbsent(e.getKey(), token);
            }
            try {
                Map<String, String> live = StateDictModuleBuilder.extractStructureMeta(this);
                for (Map.Entry<String, String> e : live.entrySet()) {
                    this.structureMeta.putIfAbsent(e.getKey(), e.getValue());
                }
            } catch (Throwable ignored) {}
        }
    }

    /** Empty bag; fill via {@link #put(String, Tensor)}. */
    public WeightBagModule() {
        super("WeightBagModule");
        this.requiresGrad = true;
        this.typed = true;
    }

    // ---- primary loaders (the three formats) --------------------------------

    /**
     * Load a <b>Python</b> {@code torch.save} checkpoint ({@code .pth}/{@code .pt} ZIP)
     * and rebuild a trainable Module.
     *
     * <p><b>Precise path (preferred):</b> if a sibling {@code *.structure.json}
     * (schema v2 from {@code dump_module_structure.py}) exists next to the pth,
     * rebuild with exact topology + hypers — <b>no safetensors required</b>.
     *
     * <p><b>Degraded path:</b> when structure is missing, fall back to typed
     * leaf inference + Sequential gap-fill heuristics and print a warning.
     * This mode is not claimed as precise.
     *
     * <pre>{@code
     *   WeightBagModule bag = WeightBagModule.fromPythonPth("model.pth");
     *   System.out.println(bag); // ModulePrinter tree
     * }</pre>
     */
    public static WeightBagModule fromPythonPth(File file) throws IOException {
        return fromPythonPth(file, true);
    }

    public static WeightBagModule fromPythonPth(File file, boolean requiresGrad) throws IOException {
        Objects.requireNonNull(file, "file");
        if (!file.isFile()) throw new IOException("not a file: " + file);

        // 1) Precise: sibling structure.json (no safetensors mid-format)
        Path structurePath = StructureSpec.findSibling(file.toPath());
        if (structurePath != null) {
            try {
                return fromPythonPthPrecise(file, structurePath.toFile(), requiresGrad);
            } catch (Exception e) {
                System.err.println("[WeightBagModule] precise load failed (" + e
                        + "); falling back to heuristic rebuild");
            }
        }

        // 2) Optional: sibling safetensors with compact module_structure meta
        File siblingSt = PthToSafeTensors.defaultOutput(file);
        if (siblingSt.isFile() && siblingSt.lastModified() >= file.lastModified()) {
            try {
                return fromSafetensors(siblingSt, requiresGrad);
            } catch (Exception ignored) {
                // fall through
            }
        }

        // 3) Degraded: pure state_dict + heuristics
        Map<String, Tensor> sd = TorchPthReader.loadStateDict(file);
        if (sd.isEmpty()) {
            throw new IOException("no tensors extracted from Python pth: " + file);
        }
        if (structurePath == null) {
            System.err.println("[WeightBagModule] WARNING: no sibling *.structure.json for "
                    + file.getName()
                    + " — using heuristic rebuild (param-free layers may be wrong)."
                    + " Export structure with scripts/dump_module_structure.py for precise mode.");
        }
        return new WeightBagModule(sd, requiresGrad, true, true, null);
    }

    public static WeightBagModule fromPythonPth(String path) throws IOException {
        return fromPythonPth(new File(path));
    }

    public static WeightBagModule fromPythonPth(Path path) throws IOException {
        return fromPythonPth(path.toFile());
    }

    /**
     * <b>Precise</b> load: Python {@code .pth} state_dict + schema-v2 structure JSON.
     * Does <b>not</b> use safetensors or any other intermediate format.
     *
     * <pre>{@code
     *   WeightBagModule bag = WeightBagModule.fromPythonPthPrecise(
     *       new File("dssm_1pct_state_dict.pth"),
     *       new File("dssm_1pct.structure.json"));
     * }</pre>
     */
    public static WeightBagModule fromPythonPthPrecise(File pth, File structureJson)
            throws IOException {
        return fromPythonPthPrecise(pth, structureJson, true);
    }

    public static WeightBagModule fromPythonPthPrecise(File pth, File structureJson,
                                                       boolean requiresGrad) throws IOException {
        Objects.requireNonNull(pth, "pth");
        Objects.requireNonNull(structureJson, "structureJson");
        if (!pth.isFile()) throw new IOException("not a file: " + pth);
        if (!structureJson.isFile()) throw new IOException("not a file: " + structureJson);

        StructureSpec spec = StructureSpec.load(structureJson);
        Map<String, Tensor> sd = TorchPthReader.loadStateDict(pth);
        if (sd.isEmpty()) {
            throw new IOException("no tensors extracted from Python pth: " + pth);
        }
        return StructureModuleBuilder.build(spec, sd, requiresGrad, /*strict=*/false);
    }

    public static WeightBagModule fromPythonPthPrecise(String pth, String structureJson)
            throws IOException {
        return fromPythonPthPrecise(new File(pth), new File(structureJson));
    }

    public static WeightBagModule fromPythonPthPrecise(Path pth, Path structureJson)
            throws IOException {
        return fromPythonPthPrecise(pth.toFile(), structureJson.toFile());
    }

    /**
     * Adopt a tree produced by {@link StructureModuleBuilder} into this empty bag.
     * Package-private — used only by the precise builder.
     */
    void adoptPreciseBuild(Map<String, Tensor> owned,
                           Map<String, Module> childMap,
                           List<StateDictModuleBuilder.LayerInfo> layers,
                           Map<String, String> compactMeta,
                           boolean requiresGradIgnored) {
        if (owned != null) {
            ownedParams.clear();
            for (Map.Entry<String, Tensor> e : owned.entrySet()) {
                Tensor t = e.getValue();
                Tensor r = retainHandle(t);
                if (r != null) ownedParams.put(e.getKey(), r);
            }
        }
        if (childMap != null) {
            children.clear();
            children.putAll(childMap);
        }
        if (layers != null) {
            layerInfos.clear();
            layerInfos.addAll(layers);
        }
        if (compactMeta != null) {
            structureMeta.clear();
            structureMeta.putAll(compactMeta);
        }
        // Re-sync live Module parameter handles (typed leaves own weight/bias)
        refreshOwnedFromModule();
        // Merge live structure tokens
        try {
            for (Map.Entry<String, Module> e : children.entrySet()) {
                String token = StateDictModuleBuilder.classifyModule(e.getValue());
                if (token != null) structureMeta.putIfAbsent(e.getKey(), token);
            }
        } catch (Throwable ignored) {}
    }

    // ---- native JavaCPP/LibTorch .pt I/O --------------------------------------

    /**
     * Save this Module as a native LibTorch archive ({@code OutputArchive}),
     * same format as {@code samples/SimpleMNIST} checkpoints.
     * Does not use safetensors.
     */
    public void saveNative(File file) throws IOException {
        NativeModuleIO.save(this, file);
    }

    public void saveNative(Path path) throws IOException {
        NativeModuleIO.save(this, path.toFile());
    }

    /**
     * Load weights from a native LibTorch archive into this pre-built Module.
     * Architecture must already match (use {@link StructureModuleBuilder#buildEmpty}).
     */
    public void loadNative(File file) throws IOException {
        NativeModuleIO.load(this, file);
        refreshOwnedFromModule();
    }

    public void loadNative(Path path) throws IOException {
        loadNative(path.toFile());
    }

    /**
     * Dump a schema-v2 {@code *.structure.json} from this live Module tree —
     * <b>pure Java</b>, no Python required.
     *
     * <p>Same format as {@code scripts/dump_module_structure.py}, so the file can
     * be reloaded via {@link #fromPythonPthPrecise} / {@link StructureModuleBuilder}.
     *
     * <pre>{@code
     *   WeightBagModule bag = WeightBagModule.fromPythonPthPrecise(pth, structure);
     *   bag.saveStructure(new File("roundtrip.structure.json"));
     *   // or auto path next to a pth:
     *   bag.saveStructureNextTo(new File("model_state_dict.pth"));
     * }</pre>
     *
     * @return the dumped {@link StructureSpec}
     */
    public StructureSpec saveStructure(File file) throws IOException {
        return StructureSpec.dump(this, file);
    }

    public StructureSpec saveStructure(Path path) throws IOException {
        return StructureSpec.dump(this, path);
    }

    public StructureSpec saveStructure(String path) throws IOException {
        return StructureSpec.dump(this, path);
    }

    /**
     * Dump structure next to a Python/JavaCPP weight file using the default
     * sibling naming ({@code foo_state_dict.pth} → {@code foo.structure.json}).
     */
    public StructureSpec saveStructureNextTo(File pthOrPt) throws IOException {
        Path out = StructureSpec.defaultStructurePath(pthOrPt.toPath());
        return StructureSpec.dump(this, out);
    }

    public StructureSpec saveStructureNextTo(Path pthOrPt) throws IOException {
        return StructureSpec.dump(this, StructureSpec.defaultStructurePath(pthOrPt));
    }

    /** Build a {@link StructureSpec} snapshot of this Module without writing a file. */
    public StructureSpec toStructureSpec() {
        return StructureSpec.fromModule(this);
    }

    /**
     * <b>Locked:</b> export schema-v2 structure from an original Python
     * {@code torch.save} ZIP {@code .pth} only ({@link TorchPthReader}).
     * Refuses safetensors and JavaCPP native archives.
     *
     * @see StructureSpec#fromPythonPth(File, File)
     * @see StructureSpec#dumpFromPythonPth(File, File, File)
     */
    public static StructureSpec structureFromPythonPth(File pth, File structureOrNull)
            throws IOException {
        return StructureSpec.fromPythonPth(pth, structureOrNull);
    }

    public static StructureSpec structureFromPythonPth(File pth) throws IOException {
        return StructureSpec.fromPythonPth(pth, null);
    }

    public static StructureSpec dumpStructureFromPythonPth(File pth, File structureOrNull,
                                                          File outStructureJson) throws IOException {
        return StructureSpec.dumpFromPythonPth(pth, structureOrNull, outStructureJson);
    }

    public static StructureSpec dumpStructureFromPythonPthNextTo(File pth, File structureOrNull)
            throws IOException {
        return StructureSpec.dumpFromPythonPthNextTo(pth, structureOrNull);
    }

    /**
     * Load a <b>JavaCPP / LibTorch</b> {@code torch::save} archive, or auto-detect
     * Python ZIP .pth / safetensors when the extension is ambiguous.
     *
     * <p>JavaCPP {@code torch::save} of a full Module requires a pre-built
     * architecture to {@code load} into — without that, we extract what we can:
     * <ul>
     *   <li>if file is Python ZIP torch → same as {@link #fromPythonPth}</li>
     *   <li>if file is safetensors → {@link #fromSafetensors}</li>
     *   <li>otherwise: attempt to treat as Python ZIP; clear error if not</li>
     * </ul>
     * Prefer saving/loading via {@link #saveSafetensors} / {@link #fromSafetensors}
     * for complete structure (including ReLU/Dropout) round-trips.
     */
    public static WeightBagModule fromJavacppPth(File file) throws IOException {
        return fromJavacppPth(file, true);
    }

    public static WeightBagModule fromJavacppPth(File file, boolean requiresGrad) throws IOException {
        Objects.requireNonNull(file, "file");
        if (!file.isFile()) throw new IOException("not a file: " + file);
        String name = file.getName().toLowerCase(Locale.ROOT);
        if (name.endsWith(".safetensors")) {
            return fromSafetensors(file, requiresGrad);
        }
        // Magic-byte / extension detect
        ModelWeights.Format fmt = ModelWeights.detect(file);
        switch (fmt) {
            case SAFETENSORS:
                return fromSafetensors(file, requiresGrad);
            case TORCH_PTH_ZIP:
                // Python-style ZIP — same pure-Java path (works for both origins
                // when the payload is a state_dict pickle)
                return fromPythonPth(file, requiresGrad);
            default:
                // Try ZIP anyway (some JavaCPP dumps still use PK zip)
                if (TorchPthReader.isZipTorch(file)) {
                    return fromPythonPth(file, requiresGrad);
                }
                throw new IOException(
                        "fromJavacppPth: unrecognized format for " + file
                        + " — expected Python torch ZIP .pth, safetensors, or a"
                        + " checkpoint convertible via TorchPthReader."
                        + " For complete structure round-trip prefer"
                        + " WeightBagModule.saveSafetensors / fromSafetensors.");
        }
    }

    public static WeightBagModule fromJavacppPth(String path) throws IOException {
        return fromJavacppPth(new File(path));
    }

    public static WeightBagModule fromJavacppPth(Path path) throws IOException {
        return fromJavacppPth(path.toFile());
    }

    /**
     * Load HuggingFace {@code .safetensors} and rebuild a trainable Module.
     * Reads optional {@code __metadata__.module_structure} for exact ReLU/Dropout
     * restoration; otherwise applies Sequential gap-fill heuristics.
     */
    public static WeightBagModule fromSafetensors(File file) throws IOException {
        return fromSafetensors(file, true);
    }

    public static WeightBagModule fromSafetensors(File file, boolean requiresGrad) throws IOException {
        return fromSafetensors(file, requiresGrad, LoadOptions.defaults());
    }

    /**
     * Load with full {@link LoadOptions} (zeroCopy, map_location/device, dtype, dequantFp8).
     *
     * <p>When {@code opts.weightsOnly} is true this still builds a Module — use
     * {@link SafeTensors#loadFile(File, LoadOptions)} for a pure tensor map.
     * Directory / index inputs are accepted (sharded HF checkpoints).
     */
    public static WeightBagModule fromSafetensors(File file, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        Objects.requireNonNull(file, "file");
        if (opts == null) opts = LoadOptions.defaults();

        Map<String, Tensor> sd;
        Map<String, String> meta = null;
        if (file.isDirectory()) {
            sd = ShardedSafeTensors.loadDirectory(file.toPath(), opts);
            // Prefer metadata from the first usable shard when present
            try {
                List<Path> shards = ShardedSafeTensors.resolveShards(file.toPath());
                if (!shards.isEmpty()) meta = SafeTensors.readMetadata(shards.get(0).toFile());
            } catch (Throwable ignored) {}
        } else if (file.getName().toLowerCase(Locale.ROOT).endsWith("index.json")) {
            sd = ShardedSafeTensors.loadIndex(file.toPath(), opts);
        } else {
            if (!file.isFile()) throw new IOException("not a file: " + file);
            sd = SafeTensors.loadAsTensors(file, opts.zeroCopy);
            if (opts.dequantFp8) {
                sd = ShardedSafeTensors.tryDequantFp8(sd);
            }
            sd = SafeTensors.applyMapLocation(sd, opts);
            meta = SafeTensors.readMetadata(file);
        }

        Map<String, String> structure = null;
        if (meta != null) {
            String enc = meta.get("module_structure");
            if (enc == null) enc = meta.get("structure");
            if (enc != null && !enc.isEmpty()) {
                structure = StateDictModuleBuilder.decodeStructureMeta(enc);
            }
        }
        // clone=true so bag owns storage even when source was mmap/zero-copy
        return new WeightBagModule(sd, requiresGrad, /*clone=*/true, /*typed=*/true, structure);
    }

    /**
     * Python-style flags: {@code map_location} + {@code strict} (strict reserved for
     * inject paths; structure rebuild always soft-fills missing param-free layers).
     */
    public static WeightBagModule fromSafetensors(File file, boolean requiresGrad,
                                                   Device mapLocation, boolean strict)
            throws IOException {
        LoadOptions opts = LoadOptions.builder()
                .mapLocation(mapLocation)
                .strict(strict)
                .build();
        return fromSafetensors(file, requiresGrad, opts);
    }

    public static WeightBagModule fromSafetensors(String path) throws IOException {
        return fromSafetensors(new File(path));
    }

    public static WeightBagModule fromSafetensors(Path path) throws IOException {
        return fromSafetensors(path.toFile());
    }

    public static WeightBagModule fromSafetensors(String path, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        return fromSafetensors(new File(path), requiresGrad, opts);
    }

    public static WeightBagModule fromSafetensors(Path path, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        return fromSafetensors(path.toFile(), requiresGrad, opts);
    }

    /**
     * Auto-detect format (Python pth / safetensors / ZIP) and load as trainable Module.
     * Equivalent to {@link ModelWeights#toModule(File)} but always typed + gap-fill.
     */
    public static WeightBagModule fromFile(File file) throws IOException {
        return fromFile(file, true);
    }

    public static WeightBagModule fromFile(File file, boolean requiresGrad) throws IOException {
        return fromFile(file, requiresGrad, LoadOptions.defaults());
    }

    /**
     * Auto-detect + honour {@link LoadOptions} for safetensors paths (single file,
     * directory, or index). For Python {@code .pth} only {@code requiresGrad} applies
     * (map_location is applied post-load when possible).
     */
    public static WeightBagModule fromFile(File file, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        Objects.requireNonNull(file, "file");
        if (opts == null) opts = LoadOptions.defaults();

        if (file.isDirectory()) {
            // Prefer sharded safetensors under HF-style layout
            try {
                if (!ShardedSafeTensors.resolveShards(file.toPath()).isEmpty()) {
                    return fromSafetensors(file, requiresGrad, opts);
                }
            } catch (IOException ignored) {}
            return ModelWeights.toModuleFromDirectory(file.toPath(), requiresGrad, opts);
        }

        String lower = file.getName().toLowerCase(Locale.ROOT);
        if (lower.endsWith("index.json") || lower.endsWith(".safetensors")) {
            return fromSafetensors(file, requiresGrad, opts);
        }

        ModelWeights.Format fmt = ModelWeights.detect(file);
        switch (fmt) {
            case SAFETENSORS:
                return fromSafetensors(file, requiresGrad, opts);
            case TORCH_PTH_ZIP: {
                WeightBagModule bag = fromPythonPth(file, requiresGrad);
                if (opts.mapLocation != null || opts.dtype != null) {
                    // Best-effort: move owned params
                    Map<String, Tensor> moved = SafeTensors.applyMapLocation(bag.stateDict(), opts);
                    return new WeightBagModule(moved, requiresGrad, /*clone=*/true, true,
                            bag.structureMeta());
                }
                return bag;
            }
            default:
                return fromJavacppPth(file, requiresGrad);
        }
    }

    public static WeightBagModule fromFile(File file, boolean requiresGrad,
                                            Device mapLocation, boolean strict) throws IOException {
        return fromFile(file, requiresGrad, LoadOptions.builder()
                .mapLocation(mapLocation)
                .strict(strict)
                .build());
    }

    public static WeightBagModule fromFile(String path) throws IOException {
        return fromFile(new File(path));
    }

    public static WeightBagModule fromFile(String path, boolean requiresGrad, LoadOptions opts)
            throws IOException {
        return fromFile(new File(path), requiresGrad, opts);
    }

    // ---- factories ----------------------------------------------------------

    public static WeightBagModule from(Map<String, Tensor> weights) {
        return new WeightBagModule(weights, true, true, true, null);
    }

    public static WeightBagModule from(Map<String, Tensor> weights, boolean requiresGrad) {
        return new WeightBagModule(weights, requiresGrad, true, true, null);
    }

    /**
     * Typed reconstruction with optional structure metadata for param-free layers.
     */
    public static WeightBagModule from(Map<String, Tensor> weights, boolean requiresGrad,
                                       Map<String, String> structureMeta) {
        return new WeightBagModule(weights, requiresGrad, true, true, structureMeta);
    }

    /**
     * Typed reconstruction (default): real Linear/Embedding/… leaves with
     * hyperparameters inferred from the state-dict + Sequential gap-fill.
     */
    public static WeightBagModule fromTyped(Map<String, Tensor> weights, boolean requiresGrad) {
        return new WeightBagModule(weights, requiresGrad, true, true, null);
    }

    /**
     * Legacy bag-only reconstruction: nested Modules + register_parameter,
     * no typed leaves.
     */
    public static WeightBagModule fromBagOnly(Map<String, Tensor> weights, boolean requiresGrad) {
        return new WeightBagModule(weights, requiresGrad, true, false, null);
    }

    /**
     * Build without cloning — shares storage with the source map. Prefer only
     * when tensors already own writable storage and you accept aliasing.
     * Uses typed reconstruction.
     */
    public static WeightBagModule share(Map<String, Tensor> weights, boolean requiresGrad) {
        return new WeightBagModule(weights, requiresGrad, false, true, null);
    }

    // ---- build paths --------------------------------------------------------

    private void buildTyped(Map<String, Tensor> weights, boolean requiresGrad, boolean clone,
                            Map<String, String> structureMeta) {
        StateDictModuleBuilder.buildInto(
                this,
                weights,
                requiresGrad,
                clone,
                layerInfos,
                ownedParams,
                children,
                structureMeta);
    }

    private void buildBagOnly(Map<String, Tensor> weights, boolean requiresGrad, boolean clone) {
        for (Map.Entry<String, Tensor> e : weights.entrySet()) {
            putInternalBag(e.getKey(), e.getValue(), requiresGrad, clone);
        }
    }

    /**
     * Re-sync {@link #ownedParams} from {@code named_parameters(true)} so
     * freeze/get/Adam see the same live tensors the Module owns (important
     * after typed leaves copy_ into their internal weight/bias).
     */
    private void refreshOwnedFromModule() {
        try {
            StringTensorDict dict = named_parameters(/*recurse=*/true);
            if (dict == null || dict.isNull()) return;
            long n = dict.size();
            for (long i = 0; i < n; i++) {
                StringTensorDictItem item = dict.get(i);
                if (item == null || item.isNull()) continue;
                String name = item.key() != null ? item.key().getString() : null;
                Tensor t = item.value();
                if (name == null || t == null) continue;
                try {
                    if (t.isNull() || !t.defined()) continue;
                } catch (Throwable ex) { continue; }
                // named_parameters returns @ByRef views — must retain before storing
                // or later numel/requires_grad_ SIGSEGV.
                Tensor retained = retainHandle(t);
                if (retained == null) continue;
                ownedParams.putIfAbsent(name, retained);
                if (typed) ownedParams.put(name, retained);
            }
        } catch (Throwable ignored) {
            // keep whatever buildInto already stored
        }
        // Also collect buffers (BatchNorm running_*) so stateDict() is complete
        try {
            StringTensorDict bufs = named_buffers(/*recurse=*/true);
            if (bufs == null || bufs.isNull()) return;
            long n = bufs.size();
            for (long i = 0; i < n; i++) {
                StringTensorDictItem item = bufs.get(i);
                if (item == null || item.isNull()) continue;
                String name = item.key() != null ? item.key().getString() : null;
                Tensor t = item.value();
                if (name == null || t == null) continue;
                try {
                    if (t.isNull() || !t.defined()) continue;
                } catch (Throwable ex) { continue; }
                Tensor retained = retainHandle(t);
                if (retained != null) ownedParams.put(name, retained);
            }
        } catch (Throwable ignored) {}
    }

    /** Tensor copy-ctor retain so a @ByRef view becomes a Java-owned handle. */
    private static Tensor retainHandle(Tensor byRef) {
        if (byRef == null) return null;
        try {
            if (byRef.isNull() || !byRef.defined()) return null;
            return new Tensor(byRef);
        } catch (Throwable t) {
            return null;
        }
    }

    // ---- mutation -----------------------------------------------------------

    /**
     * Insert or replace a parameter by dotted path. Creates intermediate nested
     * modules as needed. Clones storage and enables grad by default.
     * Uses bag-only registration (not typed leaf rewrite).
     *
     * @return this (fluent)
     */
    public WeightBagModule put(String key, Tensor value) {
        return put(key, value, requiresGrad, true);
    }

    public WeightBagModule put(String key, Tensor value, boolean requiresGrad, boolean clone) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        putInternalBag(key, value, requiresGrad, clone);
        return this;
    }

    private void putInternalBag(String key, Tensor value, boolean requiresGrad, boolean clone) {
        if (key.isEmpty()) {
            throw new IllegalArgumentException("parameter key must not be empty");
        }
        if (value == null || !value.defined()) {
            throw new IllegalArgumentException("undefined tensor for key: " + key);
        }

        String[] parts = key.split("\\.", -1);
        for (String p : parts) {
            if (p.isEmpty()) {
                throw new IllegalArgumentException("empty path segment in key: " + key);
            }
        }

        Module parent = this;
        String pathSoFar = "";
        for (int i = 0; i < parts.length - 1; i++) {
            String seg = parts[i];
            pathSoFar = pathSoFar.isEmpty() ? seg : pathSoFar + "." + seg;
            Module child = children.get(pathSoFar);
            if (child == null) {
                child = new Module(seg);
                parent.register_module(seg, child);
                children.put(pathSoFar, child);
            }
            parent = child;
        }

        String leaf = parts[parts.length - 1];
        Tensor owned;
        if (clone) {
            owned = value.detach().clone().contiguous();
        } else {
            owned = value;
        }
        owned.requires_grad_(requiresGrad);
        // NEVER store register_parameter return (ByRef dangles)
        parent.register_parameter(leaf, owned, requiresGrad);
        ownedParams.put(key, owned);
    }

    // ---- access -------------------------------------------------------------

    /** Number of leaf tensors (parameters + buffers) tracked by this bag. */
    public int size() {
        return ownedParams.size();
    }

    public boolean isEmpty() {
        return ownedParams.isEmpty();
    }

    public boolean contains(String key) {
        return ownedParams.containsKey(key);
    }

    /**
     * Owned leaf tensor for {@code key} (exact state-dict name). Prefer this over
     * {@code named_parameters()} when you need the Java handle that will not dangle.
     */
    public Tensor get(String key) {
        return ownedParams.get(key);
    }

    /** Unmodifiable view of owned leaf tensors (state-dict order). */
    public Map<String, Tensor> parametersMap() {
        return Collections.unmodifiableMap(ownedParams);
    }

    /** State-dict snapshot suitable for {@code SafeTensors.save}. */
    public Map<String, Tensor> stateDict() {
        return new LinkedHashMap<>(ownedParams);
    }

    /**
     * Collect named parameters via Module API and return as a Java map.
     * Keys match the original dotted state-dict names when hierarchy was rebuilt.
     */
    public Map<String, Tensor> namedParametersMap() {
        Map<String, Tensor> out = new LinkedHashMap<>();
        StringTensorDict dict = named_parameters(/*recurse=*/true);
        if (dict == null || dict.isNull()) return out;
        long n = dict.size();
        for (long i = 0; i < n; i++) {
            StringTensorDictItem item = dict.get(i);
            if (item == null || item.isNull()) continue;
            String name = item.key() != null ? item.key().getString() : null;
            Tensor t = item.value();
            if (name == null || t == null) continue;
            out.put(name, t);
        }
        return out;
    }

    /** Keys in insertion order. */
    public List<String> keys() {
        return new ArrayList<>(ownedParams.keySet());
    }

    /**
     * Parameters as {@link TensorVector} for optimizers that take a vector
     * (same as {@link Module#parameters()} but ordered by state-dict insertion).
     */
    public TensorVector parameterVector() {
        return parameters(/*recurse=*/true);
    }

    /** Whether this bag was built with typed leaf inference. */
    public boolean isTyped() {
        return typed;
    }

    /** Inferred layer list (empty when bag-only). */
    public List<StateDictModuleBuilder.LayerInfo> layerInfos() {
        return Collections.unmodifiableList(layerInfos);
    }

    /** Child module registered at dotted path, or null. */
    public Module child(String path) {
        return children.get(path);
    }

    /** Unmodifiable view of all retained child modules (containers + leaves). */
    public Map<String, Module> childrenMap() {
        return Collections.unmodifiableMap(children);
    }

    /**
     * Cast the child at {@code path} to {@link LinearImpl} when it was inferred
     * as Linear (or when the native module is actually Linear).
     *
     * @return LinearImpl or null
     */
    public LinearImpl asLinear(String path) {
        Module m = children.get(path);
        if (m == null) return null;
        try {
            LinearImpl lin = m.asLinear();
            if (lin != null && !lin.isNull()) return lin;
        } catch (Throwable ignored) {}
        return (m instanceof LinearImpl) ? (LinearImpl) m : null;
    }

    /** Cast child at {@code path} to {@link EmbeddingImpl}, or null. */
    public EmbeddingImpl asEmbedding(String path) {
        Module m = children.get(path);
        if (m == null) return null;
        try {
            EmbeddingImpl emb = m.asEmbedding();
            if (emb != null && !emb.isNull()) return emb;
        } catch (Throwable ignored) {}
        return (m instanceof EmbeddingImpl) ? (EmbeddingImpl) m : null;
    }

    /**
     * Find first child path whose leaf name equals {@code name}
     * (e.g. {@code "fc1"} matches {@code "encoder.fc1"}).
     */
    public Module findChild(String name) {
        if (name == null) return null;
        Module exact = children.get(name);
        if (exact != null) return exact;
        for (Map.Entry<String, Module> e : children.entrySet()) {
            String p = e.getKey();
            if (p.equals(name) || p.endsWith("." + name)) return e.getValue();
        }
        return null;
    }

    // ---- train helpers ------------------------------------------------------

    /** Freeze all parameters ({@code requires_grad=false}). */
    public WeightBagModule freeze() {
        for (Tensor t : ownedParams.values()) {
            if (t == null || t.isNull() || !t.defined()) continue;
            try {
                t.requires_grad_(false);
            } catch (Throwable ignored) {}
        }
        return this;
    }

    /** Unfreeze all parameters. */
    public WeightBagModule unfreeze() {
        for (Tensor t : ownedParams.values()) {
            if (t == null || t.isNull() || !t.defined()) continue;
            try {
                t.requires_grad_(true);
            } catch (Throwable ignored) {}
        }
        return this;
    }

    /**
     * Freeze parameters whose name starts with any of {@code prefixes}
     * (e.g. {@code "embedding_layer."}, {@code "transformer_encoder."}).
     *
     * <p>Defensive against dangling {@code @ByRef} handles historically stored
     * from {@code lin.weight()} — null / undefined / native errors are skipped
     * so freeze never SIGBUS.
     */
    public WeightBagModule freezePrefix(String... prefixes) {
        if (prefixes == null) return this;
        for (Map.Entry<String, Tensor> e : ownedParams.entrySet()) {
            Tensor t = e.getValue();
            if (t == null || t.isNull() || !t.defined()) continue;
            for (String p : prefixes) {
                if (p != null && e.getKey().startsWith(p)) {
                    try {
                        t.requires_grad_(false);
                    } catch (Throwable ignored) {
                        // best-effort freeze
                    }
                    break;
                }
            }
        }
        return this;
    }

    /**
     * Unfreeze only parameters matching {@code prefixes}; freeze the rest.
     * Useful for PEFT-style partial fine-tuning.
     */
    public WeightBagModule trainOnlyPrefix(String... prefixes) {
        if (prefixes == null || prefixes.length == 0) return unfreeze();
        for (Map.Entry<String, Tensor> e : ownedParams.entrySet()) {
            Tensor t = e.getValue();
            if (t == null || t.isNull() || !t.defined()) continue;
            boolean match = false;
            for (String p : prefixes) {
                if (p != null && e.getKey().startsWith(p)) {
                    match = true;
                    break;
                }
            }
            try {
                t.requires_grad_(match);
            } catch (Throwable ignored) {
                // best-effort
            }
        }
        return this;
    }

    /** Count parameters with {@code requires_grad=true}. */
    public long trainableParamCount() {
        long n = 0;
        for (Tensor t : ownedParams.values()) {
            if (t == null || t.isNull() || !t.defined()) continue;
            try {
                if (t.requires_grad()) n += t.numel();
            } catch (Throwable ignored) {}
        }
        return n;
    }

    public long totalParamCount() {
        long n = 0;
        for (Tensor t : ownedParams.values()) {
            if (t == null || t.isNull() || !t.defined()) continue;
            try {
                n += t.numel();
            } catch (Throwable ignored) {}
        }
        return n;
    }

    // ---- structure meta -----------------------------------------------------

    /** Structure metadata (path → kind token) used for param-free reconstruction. */
    public Map<String, String> structureMeta() {
        return Collections.unmodifiableMap(structureMeta);
    }

    // ---- I/O ----------------------------------------------------------------

    /**
     * Save owned parameters to a safetensors file, embedding
     * {@code module_structure} metadata so reload restores ReLU/Dropout exactly.
     */
    public void saveSafetensors(java.io.File file) throws java.io.IOException {
        saveSafetensors(file, null);
    }

    public void saveSafetensors(java.io.File file, Map<String, String> metadata)
            throws java.io.IOException {
        Map<String, String> meta = metadata == null
                ? new LinkedHashMap<>()
                : new LinkedHashMap<>(metadata);
        // Always refresh structure from the live tree so param-free layers are recorded.
        Map<String, String> live = StateDictModuleBuilder.extractStructureMeta(this);
        structureMeta.putAll(live);
        if (!structureMeta.isEmpty()) {
            meta.put("module_structure",
                    StateDictModuleBuilder.encodeStructureMeta(structureMeta));
        }
        meta.putIfAbsent("format", "pt");
        meta.putIfAbsent("converted_by", "org.bytedeco.pytorch.data.serialize.WeightBagModule");
        SafeTensors.save(stateDict(), file, meta);
    }

    /**
     * Print structure like Python {@code print(model)} via {@link ModulePrinter},
     * then optional inferred-layer summary.
     */
    public void printStructure() {
        System.out.println(ModulePrinter.format(this));
        if (!layerInfos.isEmpty()) {
            StateDictModuleBuilder.printLayers("inferred layers", layerInfos);
        }
    }

    /**
     * Nested tree via {@link ModulePrinter} — same style as Python {@code print(model)},
     * including reconstructed ReLU/Dropout when gap-fill / structure meta applied.
     */
    @Override
    public String toString() {
        try {
            return ModulePrinter.format(this);
        } catch (Throwable t) {
            return summary();
        }
    }

    /** One-line summary (counts) — use {@link #toString()} for the full tree. */
    public String summary() {
        return "WeightBagModule{params=" + ownedParams.size()
                + ", children=" + children.size()
                + ", layers=" + layerInfos.size()
                + ", typed=" + typed
                + ", total=" + totalParamCount()
                + ", trainable=" + trainableParamCount() + "}";
    }
}
