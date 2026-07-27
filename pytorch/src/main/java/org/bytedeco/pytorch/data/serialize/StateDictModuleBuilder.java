package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm3dImpl;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.Conv3dImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.GroupNormImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.GELUImpl;
import org.bytedeco.pytorch.nn.modules.IdentityImpl;
import org.bytedeco.pytorch.nn.modules.LeakyReLUImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLU6Impl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.SiLUImpl;
import org.bytedeco.pytorch.nn.modules.SigmoidImpl;
import org.bytedeco.pytorch.nn.modules.SoftmaxImpl;
import org.bytedeco.pytorch.nn.modules.TanhImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDict;
import org.bytedeco.pytorch.nn.modules.container.StringSharedModuleDictItem;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.nn.options.Conv1dOptions;
import org.bytedeco.pytorch.nn.options.Conv2dOptions;
import org.bytedeco.pytorch.nn.options.Conv3dOptions;
import org.bytedeco.pytorch.nn.options.DropoutOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.GroupNormOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * Infer real LibTorch layer types + hyperparameters from an arbitrary
 * state-dict ({@code Map&lt;String, Tensor&gt;}) and materialize them as nested
 * {@link Module}s under a root bag.
 *
 * <p>Why this exists: a pure parameter bag (register_parameter on nested
 * containers) preserves dotted names for optimizers / freeze / save, but does
 * <b>not</b> recreate {@code nn.Linear}, {@code nn.Embedding}, etc. Callers that
 * need typed leaves (e.g. {@code asLinear()}, LoRA wrap, architecture-aware
 * inject, pretty-print of layer kinds) require this builder.
 *
 * <p>Inference is best-effort from tensor shapes + leaf names (the only info a
 * safetensors / state_dict carries). Rules cover the common PyTorch patterns:
 * <ul>
 *   <li>Linear — 2D {@code weight} [{@code out},{@code in}], optional {@code bias}</li>
 *   <li>Embedding — 2D {@code weight} [{@code num},{@code dim}], no bias, emb-ish name</li>
 *   <li>LayerNorm — 1D weight/bias, optional name hints ({@code ln}, {@code layernorm}, …)</li>
 *   <li>BatchNorm1d/2d/3d — weight/bias + running_mean/var (+ num_batches_tracked)</li>
 *   <li>GroupNorm — 1D weight/bias with groupnorm name hint</li>
 *   <li>Conv1d/2d/3d — 3D/4D/5D weight, optional bias</li>
 *   <li>fallback — register raw parameters/buffers on a plain Module container</li>
 * </ul>
 *
 * <p>Ownership: every source tensor is {@code detach().clone().contiguous()}
 * before {@code copy_} into the typed leaf, so mmap/from_blob sources are safe
 * for training. Intermediate containers are retained by the root bag.
 *
 * @see WeightBagModule
 * @see org.bytedeco.pytorch.data.safetensors.SafeTensors#toModule
 */
public final class StateDictModuleBuilder {

    private StateDictModuleBuilder() {}

    // ---- public types -------------------------------------------------------

    public enum LayerKind {
        LINEAR,
        EMBEDDING,
        LAYER_NORM,
        BATCH_NORM_1D,
        BATCH_NORM_2D,
        BATCH_NORM_3D,
        GROUP_NORM,
        CONV_1D,
        CONV_2D,
        CONV_3D,
        // ---- parameter-free (structure-only; never in state-dict tensors) ----
        RELU,
        RELU6,
        LEAKY_RELU,
        GELU,
        SILU,
        TANH,
        SIGMOID,
        DROPOUT,
        IDENTITY,
        SOFTMAX,
        /** Leaf tensors that could not be typed — raw Module + register_parameter. */
        PARAMETER_BAG,
        /** Intermediate container Module (no own tensors). */
        CONTAINER,
        SEQUENTIAL
    }

    /**
     * One inferred leaf module (or raw bag) with hyperparameters and tensor
     * roles. Paths use the original state-dict dotted names without a trailing
     * {@code .weight}/leaf suffix (e.g. {@code embedding_layer.item_id}).
     */
    public static final class LayerInfo {
        public final String path;
        public final LayerKind kind;
        public final Map<String, long[]> shapes;   // role → shape
        public final Map<String, Object> hyper;    // hyperparameter snapshot
        public final List<String> tensorKeys;      // full state-dict keys consumed

        LayerInfo(String path, LayerKind kind,
                  Map<String, long[]> shapes,
                  Map<String, Object> hyper,
                  List<String> tensorKeys) {
            this.path = path;
            this.kind = kind;
            this.shapes = Collections.unmodifiableMap(new LinkedHashMap<>(shapes));
            this.hyper = Collections.unmodifiableMap(new LinkedHashMap<>(hyper));
            this.tensorKeys = Collections.unmodifiableList(new ArrayList<>(tensorKeys));
        }

        @Override
        public String toString() {
            return kind + "@" + path + hyper + " keys=" + tensorKeys.size();
        }
    }

    // ---- name heuristics ----------------------------------------------------

    private static final Pattern EMB_NAME = Pattern.compile(
            "(?i)(^|[._])(emb(edding)?s?|embed|token_?emb|wte|wpe|word_?emb|"
                    + "item_?id|user_?id|user_gmf|item_gmf|user_mlp|item_mlp|"
                    + "sparse|lookup|table)([._]|$)");
    private static final Pattern LINEAR_NAME = Pattern.compile(
            "(?i)(^|[._])(fc|linear|dense|proj(ection)?|classifier|lm_?head|"
                    + "predict|mlp|ffn|feed_?forward|q_proj|k_proj|v_proj|"
                    + "o_proj|out_proj|c_attn|c_proj|gate_proj|up_proj|down_proj)([._]|$)");
    private static final Pattern LN_NAME = Pattern.compile(
            "(?i)(^|[._])(ln|layer_?norm|layernorm|norm|rms_?norm|rmsnorm|"
                    + "final_layer_norm|input_layernorm|post_attention_layernorm)([._]|$)");
    private static final Pattern BN_NAME = Pattern.compile(
            "(?i)(^|[._])(bn|batch_?norm|batchnorm)([._]|$)");
    private static final Pattern GN_NAME = Pattern.compile(
            "(?i)(^|[._])(gn|group_?norm|groupnorm)([._]|$)");
    private static final Pattern CONV_NAME = Pattern.compile(
            "(?i)(^|[._])(conv|convolution|depthwise|pointwise)([._]|$)");

    // ---- public API ---------------------------------------------------------

    /**
     * Infer every leaf module from {@code stateDict} without building Modules.
     * Useful for structure reports.
     */
    public static List<LayerInfo> infer(Map<String, Tensor> stateDict) {
        Map<String, Map<String, Tensor>> groups = groupByParent(stateDict);
        List<LayerInfo> out = new ArrayList<>();
        for (Map.Entry<String, Map<String, Tensor>> e : groups.entrySet()) {
            out.add(inferOne(e.getKey(), e.getValue()));
        }
        return out;
    }

    /**
     * Build a typed nested Module tree under a fresh root named
     * {@code WeightBagModule}. Prefer {@link #buildInto} when the caller already
     * has a {@link WeightBagModule} instance.
     */
    public static Module build(Map<String, Tensor> stateDict,
                               boolean requiresGrad,
                               boolean clone,
                               List<LayerInfo> layersOut,
                               Map<String, Tensor> ownedOut,
                               Map<String, Module> childrenOut) {
        Module root = new Module("WeightBagModule");
        buildInto(root, stateDict, requiresGrad, clone, layersOut, ownedOut, childrenOut);
        return root;
    }

    /**
     * Build under a supplied root Module (e.g. {@link WeightBagModule}).
     * Same as {@link #buildInto(Module, Map, boolean, boolean, List, Map, Map, Map)}
     * with no structure metadata (gap-fill heuristics only for Sequential holes).
     */
    public static void buildInto(Module root,
                                 Map<String, Tensor> stateDict,
                                 boolean requiresGrad,
                                 boolean clone,
                                 List<LayerInfo> layersOut,
                                 Map<String, Tensor> ownedOut,
                                 Map<String, Module> childrenOut) {
        buildInto(root, stateDict, requiresGrad, clone, layersOut, ownedOut, childrenOut, null);
    }

    /**
     * Build under a supplied root Module with optional structure metadata.
     *
     * <p>Container policy (so {@link org.bytedeco.pytorch.nn.ModulePrinter} matches Python):
     * <ul>
     *   <li>if a node's <b>direct children names</b> are all integer indices
     *       ({@code "0"}, {@code "2"}, {@code "4"}, …) → {@link SequentialImpl}</li>
     *   <li>otherwise → plain {@link Module} container (named branches)</li>
     *   <li>typed leaves ({@link LinearImpl}, {@link EmbeddingImpl}, …) from tensors</li>
     *   <li><b>parameter-free</b> ReLU / Dropout / GELU / … filled into Sequential
     *       index holes via {@code structureMeta} (exact) or heuristics (best-effort)</li>
     * </ul>
     *
     * @param structureMeta optional map path → kind name (e.g. {@code "mlp.1" → "RELU"},
     *                      {@code "mlp.2" → "DROPOUT:0.1"}). When present, gap indices
     *                      are filled exactly; when null, common MLP patterns are inferred.
     */
    public static void buildInto(Module root,
                                 Map<String, Tensor> stateDict,
                                 boolean requiresGrad,
                                 boolean clone,
                                 List<LayerInfo> layersOut,
                                 Map<String, Tensor> ownedOut,
                                 Map<String, Module> childrenOut,
                                 Map<String, String> structureMeta) {
        Objects.requireNonNull(root, "root");
        if (stateDict == null || stateDict.isEmpty()) return;
        // Empty map must behave like "no meta" so heuristic trailing activations apply.
        if (structureMeta != null && structureMeta.isEmpty()) structureMeta = null;

        Map<String, Map<String, Tensor>> groups = groupByParent(stateDict);

        // All module paths that appear as parents of tensors or as prefixes thereof.
        Set<String> allPaths = new LinkedHashSet<>();
        for (String path : groups.keySet()) {
            if (path == null || path.isEmpty()) continue;
            String[] parts = path.split("\\.", -1);
            String soFar = "";
            for (String p : parts) {
                soFar = soFar.isEmpty() ? p : soFar + "." + p;
                allPaths.add(soFar);
            }
        }
        // Structure-meta paths may introduce param-free-only branches (rare).
        if (structureMeta != null) {
            for (String path : structureMeta.keySet()) {
                if (path == null || path.isEmpty()) continue;
                String[] parts = path.split("\\.", -1);
                String soFar = "";
                for (String p : parts) {
                    soFar = soFar.isEmpty() ? p : soFar + "." + p;
                    allPaths.add(soFar);
                }
            }
        }

        // Direct child names under each container path ("" = root).
        Map<String, Set<String>> directChildren = new LinkedHashMap<>();
        for (String path : allPaths) {
            String parent = parentOf(path);
            String name = leafNameOf(path);
            directChildren.computeIfAbsent(parent, k -> new LinkedHashSet<>()).add(name);
        }

        // Which container paths should be SequentialImpl (children are integer indices).
        Set<String> sequentialPaths = new LinkedHashSet<>();
        for (Map.Entry<String, Set<String>> e : directChildren.entrySet()) {
            if (e.getKey().isEmpty()) continue;
            if (isAllIntegerIndices(e.getValue())) {
                sequentialPaths.add(e.getKey());
            }
        }

        Map<String, Module> nodes = new LinkedHashMap<>();
        nodes.put("", root);

        // Collect parameterized leaves per Sequential parent: index → (path, info, roles)
        // so we can rebuild Sequential with holes filled, in index order.
        Map<String, TreeMap<Integer, LeafSlot>> sequentialLeaves = new LinkedHashMap<>();
        List<LeafSlot> nonSeqLeaves = new ArrayList<>();
        List<LayerInfo> layers = new ArrayList<>();
        // Top-level registration order (first-seen root segment from state-dict).
        List<String> topLevelOrder = new ArrayList<>();
        Set<String> topLevelSeen = new LinkedHashSet<>();

        for (Map.Entry<String, Map<String, Tensor>> e : groups.entrySet()) {
            String path = e.getKey();
            Map<String, Tensor> roles = e.getValue();
            if (path == null) path = "";

            LayerInfo info = inferOne(path, roles);
            layers.add(info);

            if (path.isEmpty()) {
                attachRaw(root, "", roles, requiresGrad, clone, ownedOut);
                continue;
            }

            // root segment for order
            String top = path;
            int dot = path.indexOf('.');
            if (dot >= 0) top = path.substring(0, dot);
            if (topLevelSeen.add(top)) topLevelOrder.add(top);

            String parentPath = parentOf(path);
            String leafName = leafNameOf(path);
            if (sequentialPaths.contains(parentPath) && isIntegerIndex(leafName)) {
                sequentialLeaves
                        .computeIfAbsent(parentPath, k -> new TreeMap<>())
                        .put(Integer.parseInt(leafName), new LeafSlot(path, info, roles));
            } else {
                nonSeqLeaves.add(new LeafSlot(path, info, roles));
            }
        }

        // Index non-seq leaves by path for ordered attach
        Map<String, LeafSlot> nonSeqByPath = new LinkedHashMap<>();
        for (LeafSlot s : nonSeqLeaves) nonSeqByPath.put(s.path, s);

        // Attach in top-level state-dict order so ModulePrinter matches Python.
        Set<String> builtSeq = new LinkedHashSet<>();
        for (String top : topLevelOrder) {
            // If this top is itself a Sequential/ModuleList path (children are
            // integer indices), materialize it — but do NOT `continue` afterward.
            // ModuleList leaves are often deeper than the index, e.g. AITM
            //   aits.0.q_layer.weight  (Sequential "aits" + named Linear under "0")
            // which lands in nonSeqLeaves / nested sequentialLeaves, not as a
            // direct Sequential leaf at aits.<i>. Early-continue used to drop them.
            if (sequentialPaths.contains(top) && !builtSeq.contains(top)) {
                TreeMap<Integer, LeafSlot> direct = sequentialLeaves.get(top);
                if (direct != null && !direct.isEmpty()) {
                    buildSequential(top, direct, sequentialPaths, nodes,
                            childrenOut, structureMeta, layers, requiresGrad, clone, ownedOut);
                } else {
                    // Empty direct leaves: still create the Sequential container so
                    // nested ensureParent(aits.0) / bottoms.0.mlp can attach under it.
                    ensureParent(top, sequentialPaths, nodes, childrenOut);
                }
                builtSeq.add(top);
            }
            // Nested sequential under top (e.g. bottoms.0.mlp, user_tower.mlp).
            for (String seqPath : sequentialLeaves.keySet()) {
                if (builtSeq.contains(seqPath)) continue;
                if (seqPath.equals(top) || seqPath.startsWith(top + ".")) {
                    buildSequential(seqPath, sequentialLeaves.get(seqPath), sequentialPaths,
                            nodes, childrenOut, structureMeta, layers, requiresGrad, clone, ownedOut);
                    builtSeq.add(seqPath);
                }
            }
            // Non-sequential leaves whose root segment is top
            // (aits.0.q_layer, item_proj, interest_extractor.*, …).
            for (LeafSlot slot : nonSeqLeaves) {
                String p = slot.path;
                String pTop = p;
                int d = p.indexOf('.');
                if (d >= 0) pTop = p.substring(0, d);
                if (!pTop.equals(top)) continue;
                // Skip if already placed as part of a sequential
                if (nodes.containsKey(p)) continue;
                Module leaf = materialize(slot.info, slot.roles, requiresGrad, clone, ownedOut);
                String parentPath = parentOf(p);
                String leafName = leafNameOf(p);
                Module parent = ensureParent(parentPath, sequentialPaths, nodes, childrenOut);
                attachChild(parent, leafName, leaf, false);
                nodes.put(p, leaf);
                if (childrenOut != null) childrenOut.put(p, leaf);
                recordOwned(ownedOut, p, slot.roles, leaf, slot.info.kind);
            }
        }
        // Any sequential not yet built (orphan)
        for (Map.Entry<String, TreeMap<Integer, LeafSlot>> se : sequentialLeaves.entrySet()) {
            if (builtSeq.contains(se.getKey())) continue;
            buildSequential(se.getKey(), se.getValue(), sequentialPaths, nodes, childrenOut,
                    structureMeta, layers, requiresGrad, clone, ownedOut);
        }

        if (layersOut != null) {
            layersOut.clear();
            layersOut.addAll(layers);
        }
    }

    /** Build one Sequential path with param-free gap-fill, indices 0..max. */
    private static void buildSequential(String seqPath,
                                        TreeMap<Integer, LeafSlot> byIdx,
                                        Set<String> sequentialPaths,
                                        Map<String, Module> nodes,
                                        Map<String, Module> childrenOut,
                                        Map<String, String> structureMeta,
                                        List<LayerInfo> layers,
                                        boolean requiresGrad,
                                        boolean clone,
                                        Map<String, Tensor> ownedOut) {
        if (byIdx == null || byIdx.isEmpty()) return;
        Module seq = ensureParent(seqPath, sequentialPaths, nodes, childrenOut);
        int minIdx = byIdx.firstKey();
        int maxIdx = byIdx.lastKey();
        boolean hasHoles = (maxIdx - minIdx + 1) > byIdx.size() || minIdx > 0;
        if (structureMeta != null) {
            String prefix = seqPath.isEmpty() ? "" : seqPath + ".";
            for (String mp : structureMeta.keySet()) {
                if (seqPath.isEmpty()) {
                    if (!isIntegerIndex(mp) || mp.indexOf('.') >= 0) continue;
                } else if (!mp.startsWith(prefix)) {
                    continue;
                }
                String rest = seqPath.isEmpty() ? mp : mp.substring(prefix.length());
                if (isIntegerIndex(rest) && rest.indexOf('.') < 0) {
                    int idx = Integer.parseInt(rest);
                    if (idx > maxIdx) maxIdx = idx;
                }
            }
        } else if (hasHoles) {
            // Fill interior holes always (handled by looping 0..maxIdx).
            // Trailing activation only when exactly 2 weighted leaves — classic
            // Linear→…→Linear→Act (e.g. NCF mlp ends with ReLU). With 3+ leaves
            // the last Linear is almost always an output projection without
            // activation (FuxiCTR MLP_Block(output_dim=…), DSSM towers, …).
            if (byIdx.size() == 2) {
                maxIdx = maxIdx + 1;
            }
        }
        for (int i = 0; i <= maxIdx; i++) {
            String childPath = seqPath.isEmpty() ? Integer.toString(i) : seqPath + "." + i;
            LeafSlot slot = byIdx.get(i);
            Module child;
            if (slot != null) {
                child = materialize(slot.info, slot.roles, requiresGrad, clone, ownedOut);
                recordOwned(ownedOut, slot.path, slot.roles, child, slot.info.kind);
            } else {
                String metaKind = structureMeta != null ? structureMeta.get(childPath) : null;
                LayerInfo info = inferParamFree(childPath, metaKind, seqPath, i, byIdx);
                child = materializeParamFree(info);
                layers.add(info);
            }
            attachChild(seq, Integer.toString(i), child, false);
            nodes.put(childPath, child);
            if (childrenOut != null) childrenOut.put(childPath, child);
        }
    }

    // LeafSlot needs to be a static nested class for buildSequential
    private static final class LeafSlot {
        final String path;
        final LayerInfo info;
        final Map<String, Tensor> roles;
        LeafSlot(String path, LayerInfo info, Map<String, Tensor> roles) {
            this.path = path; this.info = info; this.roles = roles;
        }
    }

    private static void recordOwned(Map<String, Tensor> ownedOut, String path,
                                    Map<String, Tensor> roles, Module leaf, LayerKind kind) {
        if (ownedOut == null || roles == null) return;
        for (Map.Entry<String, Tensor> re : roles.entrySet()) {
            String fullKey = path + "." + re.getKey();
            // Prefer a retained handle from the live Module leaf. lin.weight() etc.
            // are @ByRef — must retainTensor() before storing in a Java map or the
            // temporary view dangles → SIGBUS on later requires_grad_/numel.
            Tensor owned = lookupOwnedFromLeaf(leaf, re.getKey(), kind);
            if (owned != null && owned.defined()) {
                ownedOut.put(fullKey, owned);
            } else {
                // Fallback: keep a clone of the source role tensor so freeze/Adam
                // still have a defined handle (values may diverge from the Module
                // leaf if the leaf path failed — better than a null crash).
                Tensor src = re.getValue();
                if (src != null && src.defined()) {
                    ownedOut.put(fullKey, safeClone(src));
                }
            }
        }
    }

    // ---- param-free inference / materialize ---------------------------------

    /**
     * Infer a parameter-free layer for a Sequential hole.
     * Priority: explicit structureMeta → path-name heuristics → MLP pattern.
     */
    private static LayerInfo inferParamFree(String path, String metaKind,
                                            String seqPath, int index,
                                            TreeMap<Integer, ?> known) {
        Map<String, Object> hyper = new LinkedHashMap<>();
        Map<String, long[]> shapes = Collections.emptyMap();
        List<String> keys = Collections.singletonList(path);

        if (metaKind != null && !metaKind.isEmpty()) {
            return parseMetaKind(path, metaKind, shapes, keys);
        }

        String leaf = leafNameOf(path);
        String pathLower = path == null ? "" : path.toLowerCase(Locale.ROOT);
        String seqLower = seqPath == null ? "" : seqPath.toLowerCase(Locale.ROOT);

        // Name-based (if meta stored kind under a named path somehow)
        if (pathLower.contains("dropout") || leaf.toLowerCase(Locale.ROOT).contains("drop")) {
            hyper.put("p", 0.1);
            return new LayerInfo(path, LayerKind.DROPOUT, shapes, hyper, keys);
        }
        if (pathLower.contains("relu6")) {
            return new LayerInfo(path, LayerKind.RELU6, shapes, hyper, keys);
        }
        if (pathLower.contains("leaky")) {
            hyper.put("negative_slope", 0.01);
            return new LayerInfo(path, LayerKind.LEAKY_RELU, shapes, hyper, keys);
        }
        if (pathLower.contains("gelu")) {
            return new LayerInfo(path, LayerKind.GELU, shapes, hyper, keys);
        }
        if (pathLower.contains("silu") || pathLower.contains("swish")) {
            return new LayerInfo(path, LayerKind.SILU, shapes, hyper, keys);
        }
        if (pathLower.contains("tanh")) {
            return new LayerInfo(path, LayerKind.TANH, shapes, hyper, keys);
        }
        if (pathLower.contains("sigmoid")) {
            return new LayerInfo(path, LayerKind.SIGMOID, shapes, hyper, keys);
        }
        if (pathLower.contains("softmax")) {
            hyper.put("dim", -1L);
            return new LayerInfo(path, LayerKind.SOFTMAX, shapes, hyper, keys);
        }
        if (pathLower.contains("relu") || pathLower.contains("act")
                || pathLower.contains("activation")) {
            return new LayerInfo(path, LayerKind.RELU, shapes, hyper, keys);
        }

        // Sequential hole pattern (common MLP / residual FF):
        //   Linear(i) → ReLU/GELU(i+1) → [Dropout(i+2)] → Linear(i+3)
        Integer lower = known.lowerKey(index);
        Integer higher = known.higherKey(index);

        boolean transformerish = seqLower.contains("transformer") || seqLower.contains("ffn")
                || seqLower.contains("feed_forward")
                || seqLower.contains("encoder") || seqLower.contains("decoder");
        // "mlp" alone is often ReLU (NCF/CTR); only GELU when clearly transformer FF.

        if (lower != null && higher != null) {
            int gapStart = lower + 1;
            int gapEnd = higher - 1;
            int gapSize = gapEnd - gapStart + 1;
            int posInGap = index - gapStart; // 0-based within gap
            if (gapSize == 1) {
                // single hole between two Linears → activation
                return new LayerInfo(path,
                        transformerish ? LayerKind.GELU : LayerKind.RELU,
                        shapes, hyper, keys);
            }
            if (gapSize == 2) {
                // Linear → act → Dropout → Linear  (or act → act)
                if (posInGap == 0) {
                    return new LayerInfo(path,
                            transformerish ? LayerKind.GELU : LayerKind.RELU,
                            shapes, hyper, keys);
                }
                hyper.put("p", 0.1);
                return new LayerInfo(path, LayerKind.DROPOUT, shapes, hyper, keys);
            }
            if (gapSize >= 3) {
                // Linear → act → Dropout → Identity… → Linear
                if (posInGap == 0) {
                    return new LayerInfo(path,
                            transformerish ? LayerKind.GELU : LayerKind.RELU,
                            shapes, hyper, keys);
                }
                if (posInGap == 1) {
                    hyper.put("p", 0.1);
                    return new LayerInfo(path, LayerKind.DROPOUT, shapes, hyper, keys);
                }
                return new LayerInfo(path, LayerKind.IDENTITY, shapes, hyper, keys);
            }
        }

        // Trailing holes after last Linear (e.g. final ReLU)
        if (lower != null && higher == null) {
            if (index == lower + 1) {
                return new LayerInfo(path,
                        transformerish ? LayerKind.GELU : LayerKind.RELU,
                        shapes, hyper, keys);
            }
            if (index == lower + 2) {
                hyper.put("p", 0.1);
                return new LayerInfo(path, LayerKind.DROPOUT, shapes, hyper, keys);
            }
        }

        // Leading holes before first Linear — uncommon; Identity
        // Default fallback: ReLU (most common activation)
        return new LayerInfo(path, LayerKind.RELU, shapes, hyper, keys);
    }

    /** Parse structure-meta value: {@code "RELU"}, {@code "DROPOUT:0.1"}, {@code "SOFTMAX:-1"}. */
    static LayerInfo parseMetaKind(String path, String meta,
                                   Map<String, long[]> shapes, List<String> keys) {
        Map<String, Object> hyper = new LinkedHashMap<>();
        String raw = meta.trim();
        String kindPart = raw;
        String arg = null;
        int colon = raw.indexOf(':');
        if (colon >= 0) {
            kindPart = raw.substring(0, colon).trim();
            arg = raw.substring(colon + 1).trim();
        }
        String k = kindPart.toUpperCase(Locale.ROOT).replace('-', '_');
        // strip Impl / nn. prefixes
        if (k.endsWith("IMPL")) k = k.substring(0, k.length() - 4);
        if (k.startsWith("TORCH::NN::")) k = k.substring("TORCH::NN::".length());
        if (k.startsWith("NN.")) k = k.substring(3);

        LayerKind kind;
        switch (k) {
            case "RELU": kind = LayerKind.RELU; break;
            case "RELU6": kind = LayerKind.RELU6; break;
            case "LEAKY_RELU": case "LEAKYRELU": kind = LayerKind.LEAKY_RELU;
                hyper.put("negative_slope", arg != null ? Double.parseDouble(arg) : 0.01);
                break;
            case "GELU": kind = LayerKind.GELU; break;
            case "SILU": case "SWISH": kind = LayerKind.SILU; break;
            case "TANH": kind = LayerKind.TANH; break;
            case "SIGMOID": kind = LayerKind.SIGMOID; break;
            case "DROPOUT": case "DROPOUT1D": kind = LayerKind.DROPOUT;
                hyper.put("p", arg != null ? Double.parseDouble(arg) : 0.5);
                break;
            case "IDENTITY": case "ID": kind = LayerKind.IDENTITY; break;
            case "SOFTMAX": kind = LayerKind.SOFTMAX;
                hyper.put("dim", arg != null ? Long.parseLong(arg) : -1L);
                break;
            case "LINEAR": kind = LayerKind.LINEAR; break;
            case "EMBEDDING": kind = LayerKind.EMBEDDING; break;
            case "LAYER_NORM": case "LAYERNORM": kind = LayerKind.LAYER_NORM; break;
            case "SEQUENTIAL": kind = LayerKind.SEQUENTIAL; break;
            default:
                kind = LayerKind.RELU; // safe default for unknown param-free
                break;
        }
        return new LayerInfo(path, kind, shapes, hyper, keys);
    }

    private static Module materializeParamFree(LayerInfo info) {
        try {
            switch (info.kind) {
                case RELU: return new ReLUImpl();
                case RELU6: return new ReLU6Impl();
                case LEAKY_RELU: return new LeakyReLUImpl();
                case GELU: return new GELUImpl();
                case SILU: return new SiLUImpl();
                case TANH: return new TanhImpl();
                case SIGMOID: return new SigmoidImpl();
                case DROPOUT: {
                    double p = 0.5;
                    Object pv = info.hyper.get("p");
                    if (pv instanceof Number) p = ((Number) pv).doubleValue();
                    return new DropoutImpl(new DropoutOptions(p));
                }
                case IDENTITY: return new IdentityImpl();
                case SOFTMAX: {
                    long dim = -1L;
                    Object dv = info.hyper.get("dim");
                    if (dv instanceof Number) dim = ((Number) dv).longValue();
                    return new SoftmaxImpl(dim);
                }
                case SEQUENTIAL: return new SequentialImpl();
                default: return new IdentityImpl();
            }
        } catch (Throwable t) {
            return new IdentityImpl();
        }
    }

    private static boolean isIntegerIndex(String name) {
        if (name == null || name.isEmpty()) return false;
        for (int i = 0; i < name.length(); i++) {
            char c = name.charAt(i);
            if (c < '0' || c > '9') return false;
        }
        return true;
    }

    // ---- structure metadata (for complete round-trip of param-free layers) ----

    /**
     * Walk a live Module tree and emit structure metadata:
     * dotted path → kind token (e.g. {@code "mlp.1" → "RELU"}, {@code "mlp.2" → "DROPOUT:0.1"}).
     * Used when saving safetensors so reload restores ReLU/Dropout exactly.
     */
    public static Map<String, String> extractStructureMeta(Module module) {
        Map<String, String> out = new LinkedHashMap<>();
        if (module == null) return out;
        walkStructure(module, "", out);
        return out;
    }

    private static void walkStructure(Module m, String prefix, Map<String, String> out) {
        if (m == null) return;
        try {
            if (m.isNull()) return;
        } catch (Throwable ignored) {}
        try {
            StringSharedModuleDict kids = m.named_children();
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

                // Recover typed Java peer so instanceof works. Reject recoveries that
                // point back at the parent (seen with Sequential index slots).
                Module typed = child;
                try {
                    Module recovered = org.bytedeco.pytorch.nn.ModuleAsHelper.recover(child);
                    if (recovered != null && recovered != m && !sameNative(recovered, m)) {
                        typed = recovered;
                    }
                } catch (Throwable ignored) {}

                String path = prefix.isEmpty() ? key : prefix + "." + key;
                String token = classifyModule(typed);
                // Integer child of Sequential classified as SEQUENTIAL ⇒ recover failed
                if ("SEQUENTIAL".equals(token) && isIntegerIndex(key) && m instanceof SequentialImpl) {
                    token = null;
                    typed = child; // fall back to raw child for recursion only
                }
                if (token != null) out.put(path, token);

                // Recurse into containers only (Sequential / named Module with kids).
                // Leaf Linear/Embedding/ReLU/Dropout have no meaningful children.
                if (!isLeafToken(token) || typed instanceof SequentialImpl) {
                    walkStructure(typed, path, out);
                }
            }
        } catch (Throwable ignored) {}
    }

    private static boolean sameNative(Module a, Module b) {
        try {
            return a != null && b != null && a.address() == b.address();
        } catch (Throwable t) {
            return a == b;
        }
    }

    private static boolean isParamFreeToken(String t) {
        if (t == null) return false;
        String u = t.toUpperCase(Locale.ROOT);
        return u.startsWith("RELU") || u.startsWith("DROPOUT") || u.equals("GELU")
                || u.equals("SILU") || u.equals("TANH") || u.equals("SIGMOID")
                || u.equals("IDENTITY") || u.startsWith("SOFTMAX") || u.startsWith("LEAKY");
    }

    private static boolean isLeafToken(String t) {
        if (t == null) return false;
        String u = t.toUpperCase(Locale.ROOT);
        return u.equals("LINEAR") || u.equals("EMBEDDING") || u.startsWith("LAYER_NORM")
                || u.startsWith("BATCH_NORM") || u.startsWith("CONV") || u.startsWith("GROUP_NORM")
                || isParamFreeToken(t);
    }

    /**
     * Classify a live Module into a structure-meta token, or null if unknown.
     * Prefer Java {@code instanceof} (typed peers), then C++ {@code Module::name()}
     * RTTI (same source {@link org.bytedeco.pytorch.nn.ModulePrinter} uses — works
     * even when Sequential children come back as plain Module peers).
     * Never call {@code as*()} here (native dynamic_cast can SIGSEGV).
     */
    public static String classifyModule(Module m) {
        if (m == null) return null;
        try {
            if (m.isNull()) return null;
        } catch (Throwable ignored) {}

        // 1) Java typed peer
        if (m instanceof SequentialImpl) return "SEQUENTIAL";
        if (m instanceof LinearImpl) return "LINEAR";
        if (m instanceof EmbeddingImpl) return "EMBEDDING";
        if (m instanceof LayerNormImpl) return "LAYER_NORM";
        if (m instanceof BatchNorm1dImpl) return "BATCH_NORM_1D";
        if (m instanceof BatchNorm2dImpl) return "BATCH_NORM_2D";
        if (m instanceof BatchNorm3dImpl) return "BATCH_NORM_3D";
        if (m instanceof Conv1dImpl) return "CONV_1D";
        if (m instanceof Conv2dImpl) return "CONV_2D";
        if (m instanceof Conv3dImpl) return "CONV_3D";
        if (m instanceof ReLUImpl) return "RELU";
        if (m instanceof ReLU6Impl) return "RELU6";
        if (m instanceof GELUImpl) return "GELU";
        if (m instanceof SiLUImpl) return "SILU";
        if (m instanceof LeakyReLUImpl) return "LEAKY_RELU";
        if (m instanceof TanhImpl) return "TANH";
        if (m instanceof SigmoidImpl) return "SIGMOID";
        if (m instanceof SoftmaxImpl) return "SOFTMAX";
        if (m instanceof IdentityImpl) return "IDENTITY";
        if (m instanceof DropoutImpl) {
            double p = 0.5;
            try {
                DropoutImpl d = (DropoutImpl) m;
                if (d.options() != null && d.options().p() != null) {
                    p = d.options().p().get();
                }
            } catch (Throwable ignored) {}
            return "DROPOUT:" + p;
        }

        // 2) C++ RTTI name — works for plain Module peers of real *Impl natives
        try {
            org.bytedeco.javacpp.BytePointer bp = m.name();
            if (bp != null && !bp.isNull()) {
                String raw = bp.getString();
                String token = tokenFromCppName(raw);
                if (token != null) return token;
            }
        } catch (Throwable ignored) {}

        // 3) Java simple name fallback
        try {
            String sn = m.getClass().getSimpleName();
            if (sn.endsWith("Impl")) sn = sn.substring(0, sn.length() - 4);
            if (!sn.isEmpty() && !sn.equals("Module") && !sn.equals("WeightBagModule")) {
                return sn.toUpperCase(Locale.ROOT);
            }
        } catch (Throwable ignored) {}
        return null;
    }

    /** Map C++ / demangled name to structure-meta token. */
    static String tokenFromCppName(String raw) {
        if (raw == null || raw.isEmpty()) return null;
        String s = raw;
        // Lightweight demangle: JavaCPP_torch_0003a_0003ann_0003a_0003aReLUImpl → …ReLUImpl
        if (s.startsWith("JavaCPP_")) s = s.substring("JavaCPP_".length());
        s = s.replace("_0003a_0003a", "::").replace("_0003a", ":");
        // take last segment after ::
        int cc = s.lastIndexOf("::");
        if (cc >= 0) s = s.substring(cc + 2);
        int dot = s.lastIndexOf('.');
        if (dot >= 0) s = s.substring(dot + 1);
        if (s.endsWith("Impl")) s = s.substring(0, s.length() - 4);
        String u = s.toUpperCase(Locale.ROOT);
        switch (u) {
            case "LINEAR": return "LINEAR";
            case "EMBEDDING": return "EMBEDDING";
            case "LAYERNORM": return "LAYER_NORM";
            case "BATCHNORM1D": return "BATCH_NORM_1D";
            case "BATCHNORM2D": return "BATCH_NORM_2D";
            case "BATCHNORM3D": return "BATCH_NORM_3D";
            case "GROUPNORM": return "GROUP_NORM";
            case "CONV1D": return "CONV_1D";
            case "CONV2D": return "CONV_2D";
            case "CONV3D": return "CONV_3D";
            case "RELU": return "RELU";
            case "RELU6": return "RELU6";
            case "LEAKYRELU": return "LEAKY_RELU";
            case "GELU": return "GELU";
            case "SILU": return "SILU";
            case "TANH": return "TANH";
            case "SIGMOID": return "SIGMOID";
            case "SOFTMAX": return "SOFTMAX";
            case "DROPOUT": case "DROPOUT1D": return "DROPOUT:0.5";
            case "IDENTITY": return "IDENTITY";
            case "SEQUENTIAL": return "SEQUENTIAL";
            case "MODULE": case "MODULEBASE": return null;
            default:
                if (u.isEmpty()) return null;
                return u;
        }
    }

    /**
     * Encode structure meta as a single safetensors metadata string
     * ({@code path=KIND;path2=DROPOUT:0.1;…}).
     */
    public static String encodeStructureMeta(Map<String, String> meta) {
        if (meta == null || meta.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, String> e : meta.entrySet()) {
            if (e.getKey() == null || e.getValue() == null) continue;
            if (sb.length() > 0) sb.append(';');
            sb.append(e.getKey().replace(";", "\\;").replace("=", "\\="));
            sb.append('=');
            sb.append(e.getValue().replace(";", "\\;").replace("=", "\\="));
        }
        return sb.toString();
    }

    /** Decode structure meta from {@link #encodeStructureMeta}. */
    public static Map<String, String> decodeStructureMeta(String encoded) {
        Map<String, String> out = new LinkedHashMap<>();
        if (encoded == null || encoded.isEmpty()) return out;
        // Split on unescaped ';'
        StringBuilder cur = new StringBuilder();
        List<String> parts = new ArrayList<>();
        for (int i = 0; i < encoded.length(); i++) {
            char c = encoded.charAt(i);
            if (c == '\\' && i + 1 < encoded.length()) {
                cur.append(encoded.charAt(++i));
            } else if (c == ';') {
                parts.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        if (cur.length() > 0) parts.add(cur.toString());
        for (String part : parts) {
            int eq = -1;
            for (int i = 0; i < part.length(); i++) {
                if (part.charAt(i) == '=' && (i == 0 || part.charAt(i - 1) != '\\')) {
                    eq = i;
                    break;
                }
            }
            // simpler: first '='
            eq = part.indexOf('=');
            if (eq <= 0) continue;
            String k = part.substring(0, eq).replace("\\=", "=").replace("\\;", ";");
            String v = part.substring(eq + 1).replace("\\=", "=").replace("\\;", ";");
            out.put(k, v);
        }
        return out;
    }

    /**
     * Ensure the container at {@code path} exists, creating ancestors lazily.
     * First creation attaches the container under its parent at this moment so
     * root {@code named_children} order follows first leaf appearance in the
     * state-dict (Python registration order).
     */
    private static Module ensureParent(String path,
                                       Set<String> sequentialPaths,
                                       Map<String, Module> nodes,
                                       Map<String, Module> childrenOut) {
        if (path == null || path.isEmpty()) return nodes.get("");
        Module existing = nodes.get(path);
        if (existing != null) return existing;
        // Ensure grandparent first (depth recursion).
        String parentPath = parentOf(path);
        Module parent = ensureParent(parentPath, sequentialPaths, nodes, childrenOut);
        String name = leafNameOf(path);
        boolean sequential = sequentialPaths.contains(path);
        Module child = sequential ? new SequentialImpl() : new Module(name);
        attachChild(parent, name, child, /*replaceIfPresent=*/false);
        nodes.put(path, child);
        if (childrenOut != null) childrenOut.put(path, child);
        return child;
    }

    /** True when every child name is a non-negative integer (Sequential indices). */
    private static boolean isAllIntegerIndices(Set<String> names) {
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

    /**
     * Attach {@code child} under {@code parent} with the given name.
     * Sequential parents use {@code push_back(name, child)}; others use
     * {@code register_module}. When {@code replaceIfPresent} is true, unregisters
     * any prior child with the same name first (placeholder replacement).
     */
    private static void attachChild(Module parent, String name, Module child,
                                    boolean replaceIfPresent) {
        if (parent == null || child == null || name == null) return;
        // Keep typed Java peer for later structure extraction / asLinear etc.
        try {
            org.bytedeco.pytorch.nn.ModuleAsHelper.remember(child);
        } catch (Throwable ignored) {}
        if (replaceIfPresent) {
            try {
                parent.unregister_module(name);
            } catch (Throwable ignored) {}
        }

        if (parent instanceof SequentialImpl) {
            SequentialImpl seq = (SequentialImpl) parent;
            try {
                seq.push_back(name, child);
                try { org.bytedeco.pytorch.nn.ModuleAsHelper.remember(child); } catch (Throwable ignored) {}
                return;
            } catch (Throwable t) {
                // Fall through to register_module
            }
        }
        // Also try asSequential() carefully — only when parent is SequentialImpl Java type
        // (already handled). Avoid asSequential on plain Module (can be wrong).

        parent.register_module(name, child);
        try {
            org.bytedeco.pytorch.nn.ModuleAsHelper.remember(child);
        } catch (Throwable ignored) {}
    }

    // ---- grouping -----------------------------------------------------------

    /**
     * Group tensors by parent path. Key {@code "fc1.weight"} → parent {@code "fc1"},
     * role {@code "weight"}. Root-level {@code "weight"} → parent {@code ""}.
     */
    static Map<String, Map<String, Tensor>> groupByParent(Map<String, Tensor> stateDict) {
        Map<String, Map<String, Tensor>> groups = new LinkedHashMap<>();
        if (stateDict == null) return groups;
        for (Map.Entry<String, Tensor> e : stateDict.entrySet()) {
            String key = e.getKey();
            if (key == null || key.isEmpty()) continue;
            Tensor t = e.getValue();
            if (t == null || !t.defined()) continue;
            int dot = key.lastIndexOf('.');
            String parent = dot < 0 ? "" : key.substring(0, dot);
            String role = dot < 0 ? key : key.substring(dot + 1);
            if (role.isEmpty()) continue;
            groups.computeIfAbsent(parent, k -> new LinkedHashMap<>()).put(role, t);
        }
        return groups;
    }

    // ---- inference ----------------------------------------------------------

    static LayerInfo inferOne(String path, Map<String, Tensor> roles) {
        Map<String, long[]> shapes = new LinkedHashMap<>();
        List<String> keys = new ArrayList<>();
        for (Map.Entry<String, Tensor> e : roles.entrySet()) {
            shapes.put(e.getKey(), shapeOf(e.getValue()));
            keys.add(path == null || path.isEmpty() ? e.getKey() : path + "." + e.getKey());
        }

        String leaf = path == null || path.isEmpty()
                ? ""
                : leafNameOf(path);
        String pathLower = path == null ? "" : path.toLowerCase(Locale.ROOT);

        Tensor weight = roles.get("weight");
        Tensor bias = roles.get("bias");
        boolean hasRunning = roles.containsKey("running_mean") || roles.containsKey("running_var");
        boolean hasNumBatches = roles.containsKey("num_batches_tracked");

        Map<String, Object> hyper = new LinkedHashMap<>();

        // ---- BatchNorm family (running stats are decisive) ------------------
        if (hasRunning || hasNumBatches) {
            long[] wShape = weight != null ? shapeOf(weight) : null;
            long numFeatures = 0;
            if (wShape != null && wShape.length == 1) numFeatures = wShape[0];
            else if (roles.get("running_mean") != null) {
                long[] s = shapeOf(roles.get("running_mean"));
                if (s.length == 1) numFeatures = s[0];
            }
            hyper.put("num_features", numFeatures);
            hyper.put("affine", weight != null || bias != null);
            hyper.put("track_running_stats", hasRunning);

            LayerKind bnKind = LayerKind.BATCH_NORM_1D;
            // Name hints for 2d/3d; default 1d (most common in state_dicts for channels-only)
            if (pathLower.contains("2d") || pathLower.contains("bn2")
                    || leaf.toLowerCase(Locale.ROOT).contains("2d")) {
                bnKind = LayerKind.BATCH_NORM_2D;
            } else if (pathLower.contains("3d") || pathLower.contains("bn3")
                    || leaf.toLowerCase(Locale.ROOT).contains("3d")) {
                bnKind = LayerKind.BATCH_NORM_3D;
            } else if (BN_NAME.matcher(path).find() || BN_NAME.matcher(leaf).find()) {
                // keep 1d default
                bnKind = LayerKind.BATCH_NORM_1D;
            }
            return new LayerInfo(path, bnKind, shapes, hyper, keys);
        }

        // ---- Conv family (weight rank 3/4/5) --------------------------------
        if (weight != null) {
            long[] ws = shapeOf(weight);
            if (ws.length == 3) {
                // Conv1d: [out, in/groups, k]
                hyper.put("out_channels", ws[0]);
                hyper.put("in_channels", ws[1]);
                hyper.put("kernel_size", ws[2]);
                hyper.put("bias", bias != null);
                return new LayerInfo(path, LayerKind.CONV_1D, shapes, hyper, keys);
            }
            if (ws.length == 4) {
                // Conv2d: [out, in/groups, kH, kW]
                hyper.put("out_channels", ws[0]);
                hyper.put("in_channels", ws[1]);
                hyper.put("kernel_size", new long[]{ws[2], ws[3]});
                hyper.put("bias", bias != null);
                return new LayerInfo(path, LayerKind.CONV_2D, shapes, hyper, keys);
            }
            if (ws.length == 5) {
                hyper.put("out_channels", ws[0]);
                hyper.put("in_channels", ws[1]);
                hyper.put("kernel_size", new long[]{ws[2], ws[3], ws[4]});
                hyper.put("bias", bias != null);
                return new LayerInfo(path, LayerKind.CONV_3D, shapes, hyper, keys);
            }
        }

        // ---- GroupNorm ------------------------------------------------------
        if (weight != null && shapeOf(weight).length == 1
                && (GN_NAME.matcher(path).find() || GN_NAME.matcher(leaf).find())) {
            long numChannels = shapeOf(weight)[0];
            hyper.put("num_channels", numChannels);
            // num_groups unknown from weights alone — default 32 or 1
            long numGroups = guessNumGroups(numChannels);
            hyper.put("num_groups", numGroups);
            hyper.put("affine", true);
            return new LayerInfo(path, LayerKind.GROUP_NORM, shapes, hyper, keys);
        }

        // ---- LayerNorm (1D weight/bias, no running stats) -------------------
        if (weight != null && shapeOf(weight).length == 1 && !hasRunning) {
            boolean lnHint = LN_NAME.matcher(path).find() || LN_NAME.matcher(leaf).find();
            boolean embHint = EMB_NAME.matcher(path).find() || EMB_NAME.matcher(leaf).find();
            boolean linearHint = LINEAR_NAME.matcher(path).find() || LINEAR_NAME.matcher(leaf).find();
            // Prefer LN when named as norm, or when only weight(+bias) 1D without emb/linear hints
            if (lnHint || (!embHint && !linearHint && (bias != null || roles.size() <= 2))) {
                long[] ns = shapeOf(weight);
                hyper.put("normalized_shape", ns.clone());
                hyper.put("elementwise_affine", true);
                hyper.put("bias", bias != null);
                return new LayerInfo(path, LayerKind.LAYER_NORM, shapes, hyper, keys);
            }
        }

        // ---- Embedding vs Linear (2D weight) --------------------------------
        if (weight != null && shapeOf(weight).length == 2) {
            long[] ws = shapeOf(weight);
            boolean embHint = EMB_NAME.matcher(path).find() || EMB_NAME.matcher(leaf).find();
            boolean linearHint = LINEAR_NAME.matcher(path).find() || LINEAR_NAME.matcher(leaf).find();
            boolean hasBias = bias != null;

            // Embedding wins when name is emb-like and there is no bias — even if the
            // name also matches a linear pattern (e.g. user_mlp / item_mlp / wte).
            // Bias almost never exists on nn.Embedding; presence of bias ⇒ Linear.
            if (embHint && !hasBias) {
                hyper.put("num_embeddings", ws[0]);
                hyper.put("embedding_dim", ws[1]);
                return new LayerInfo(path, LayerKind.EMBEDDING, shapes, hyper, keys);
            }
            // Linear: explicit linear name, or bias present, or no emb hint
            if (linearHint || hasBias || !embHint) {
                // weight is [out_features, in_features]
                hyper.put("out_features", ws[0]);
                hyper.put("in_features", ws[1]);
                hyper.put("bias", hasBias);
                return new LayerInfo(path, LayerKind.LINEAR, shapes, hyper, keys);
            }
            // embHint && hasBias (rare) — still treat as Embedding by name
            hyper.put("num_embeddings", ws[0]);
            hyper.put("embedding_dim", ws[1]);
            return new LayerInfo(path, LayerKind.EMBEDDING, shapes, hyper, keys);
        }

        // ---- Embedding without "weight" name? unlikely ----------------------
        // ---- fallback PARAMETER_BAG -----------------------------------------
        hyper.put("roles", new ArrayList<>(roles.keySet()));
        return new LayerInfo(path, LayerKind.PARAMETER_BAG, shapes, hyper, keys);
    }

    // ---- materialize typed leaves -------------------------------------------

    private static Module materialize(LayerInfo info,
                                      Map<String, Tensor> roles,
                                      boolean requiresGrad,
                                      boolean clone,
                                      Map<String, Tensor> ownedOut) {
        try {
            switch (info.kind) {
                case LINEAR:
                    return makeLinear(info, roles, requiresGrad, clone);
                case EMBEDDING:
                    return makeEmbedding(info, roles, requiresGrad, clone);
                case LAYER_NORM:
                    return makeLayerNorm(info, roles, requiresGrad, clone);
                case BATCH_NORM_1D:
                    return makeBatchNorm(info, roles, requiresGrad, clone, 1);
                case BATCH_NORM_2D:
                    return makeBatchNorm(info, roles, requiresGrad, clone, 2);
                case BATCH_NORM_3D:
                    return makeBatchNorm(info, roles, requiresGrad, clone, 3);
                case GROUP_NORM:
                    return makeGroupNorm(info, roles, requiresGrad, clone);
                case CONV_1D:
                    return makeConv1d(info, roles, requiresGrad, clone);
                case CONV_2D:
                    return makeConv2d(info, roles, requiresGrad, clone);
                case CONV_3D:
                    return makeConv3d(info, roles, requiresGrad, clone);
                case RELU: case RELU6: case LEAKY_RELU: case GELU: case SILU:
                case TANH: case SIGMOID: case DROPOUT: case IDENTITY: case SOFTMAX:
                case SEQUENTIAL:
                    return materializeParamFree(info);
                case PARAMETER_BAG:
                default: {
                    Module bag = new Module(leafNameOf(info.path.isEmpty() ? "params" : info.path));
                    attachRaw(bag, info.path, roles, requiresGrad, clone, ownedOut);
                    return bag;
                }
            }
        } catch (Throwable t) {
            // Never fail the whole load — fall back to raw parameter bag.
            Module bag = new Module(leafNameOf(info.path.isEmpty() ? "params" : info.path));
            attachRaw(bag, info.path, roles, requiresGrad, clone, ownedOut);
            return bag;
        }
    }

    private static LinearImpl makeLinear(LayerInfo info, Map<String, Tensor> roles,
                                         boolean requiresGrad, boolean clone) {
        long outF = ((Number) info.hyper.get("out_features")).longValue();
        long inF = ((Number) info.hyper.get("in_features")).longValue();
        boolean withBias = Boolean.TRUE.equals(info.hyper.get("bias"));
        LinearOptions opt = new LinearOptions(inF, outF).bias(withBias);
        LinearImpl lin = new LinearImpl(opt);
        copyInto(lin.weight(), roles.get("weight"), requiresGrad, clone);
        if (withBias && roles.get("bias") != null) {
            copyInto(lin.bias(), roles.get("bias"), requiresGrad, clone);
        }
        applyGrad(lin, requiresGrad);
        return lin;
    }

    private static EmbeddingImpl makeEmbedding(LayerInfo info, Map<String, Tensor> roles,
                                               boolean requiresGrad, boolean clone) {
        long num = ((Number) info.hyper.get("num_embeddings")).longValue();
        long dim = ((Number) info.hyper.get("embedding_dim")).longValue();
        EmbeddingImpl emb = new EmbeddingImpl(new EmbeddingOptions(num, dim));
        copyInto(emb.weight(), roles.get("weight"), requiresGrad, clone);
        applyGrad(emb, requiresGrad);
        return emb;
    }

    private static LayerNormImpl makeLayerNorm(LayerInfo info, Map<String, Tensor> roles,
                                               boolean requiresGrad, boolean clone) {
        long[] ns = (long[]) info.hyper.get("normalized_shape");
        LongVector shape = new LongVector(ns);
        LayerNormOptions opt = new LayerNormOptions(shape);
        // elementwise_affine true when weight present
        opt.elementwise_affine(roles.get("weight") != null);
        LayerNormImpl ln = new LayerNormImpl(opt);
        if (roles.get("weight") != null && ln.weight() != null && ln.weight().defined()) {
            copyInto(ln.weight(), roles.get("weight"), requiresGrad, clone);
        }
        if (roles.get("bias") != null && ln.bias() != null && ln.bias().defined()) {
            copyInto(ln.bias(), roles.get("bias"), requiresGrad, clone);
        }
        applyGrad(ln, requiresGrad);
        return ln;
    }

    private static Module makeBatchNorm(LayerInfo info, Map<String, Tensor> roles,
                                        boolean requiresGrad, boolean clone, int dim) {
        long numFeatures = ((Number) info.hyper.get("num_features")).longValue();
        boolean affine = Boolean.TRUE.equals(info.hyper.get("affine"));
        boolean track = Boolean.TRUE.equals(info.hyper.get("track_running_stats"));
        BatchNormOptions opt = new BatchNormOptions(numFeatures)
                .affine(affine)
                .track_running_stats(track);

        Module bn;
        switch (dim) {
            case 2:
                bn = new BatchNorm2dImpl(opt);
                break;
            case 3:
                bn = new BatchNorm3dImpl(opt);
                break;
            default:
                bn = new BatchNorm1dImpl(opt);
                break;
        }

        // Copy via as* accessors
        try {
            if (dim == 1) {
                BatchNorm1dImpl m = (BatchNorm1dImpl) bn;
                if (roles.get("weight") != null) copyInto(m.weight(), roles.get("weight"), requiresGrad, clone);
                if (roles.get("bias") != null) copyInto(m.bias(), roles.get("bias"), requiresGrad, clone);
                if (roles.get("running_mean") != null) copyInto(m.running_mean(), roles.get("running_mean"), false, clone);
                if (roles.get("running_var") != null) copyInto(m.running_var(), roles.get("running_var"), false, clone);
                if (roles.get("num_batches_tracked") != null
                        && m.num_batches_tracked() != null && m.num_batches_tracked().defined()) {
                    copyInto(m.num_batches_tracked(), roles.get("num_batches_tracked"), false, clone);
                }
            } else if (dim == 2) {
                BatchNorm2dImpl m = (BatchNorm2dImpl) bn;
                if (roles.get("weight") != null) copyInto(m.weight(), roles.get("weight"), requiresGrad, clone);
                if (roles.get("bias") != null) copyInto(m.bias(), roles.get("bias"), requiresGrad, clone);
                if (roles.get("running_mean") != null) copyInto(m.running_mean(), roles.get("running_mean"), false, clone);
                if (roles.get("running_var") != null) copyInto(m.running_var(), roles.get("running_var"), false, clone);
                if (roles.get("num_batches_tracked") != null
                        && m.num_batches_tracked() != null && m.num_batches_tracked().defined()) {
                    copyInto(m.num_batches_tracked(), roles.get("num_batches_tracked"), false, clone);
                }
            } else {
                BatchNorm3dImpl m = (BatchNorm3dImpl) bn;
                if (roles.get("weight") != null) copyInto(m.weight(), roles.get("weight"), requiresGrad, clone);
                if (roles.get("bias") != null) copyInto(m.bias(), roles.get("bias"), requiresGrad, clone);
                if (roles.get("running_mean") != null) copyInto(m.running_mean(), roles.get("running_mean"), false, clone);
                if (roles.get("running_var") != null) copyInto(m.running_var(), roles.get("running_var"), false, clone);
                if (roles.get("num_batches_tracked") != null
                        && m.num_batches_tracked() != null && m.num_batches_tracked().defined()) {
                    copyInto(m.num_batches_tracked(), roles.get("num_batches_tracked"), false, clone);
                }
            }
        } catch (Throwable ignored) {
            // partial copy is ok
        }
        applyGrad(bn, requiresGrad);
        return bn;
    }

    private static GroupNormImpl makeGroupNorm(LayerInfo info, Map<String, Tensor> roles,
                                               boolean requiresGrad, boolean clone) {
        long numChannels = ((Number) info.hyper.get("num_channels")).longValue();
        long numGroups = ((Number) info.hyper.get("num_groups")).longValue();
        GroupNormOptions opt = new GroupNormOptions(numGroups, numChannels);
        GroupNormImpl gn = new GroupNormImpl(opt);
        if (roles.get("weight") != null) copyInto(gn.weight(), roles.get("weight"), requiresGrad, clone);
        if (roles.get("bias") != null) copyInto(gn.bias(), roles.get("bias"), requiresGrad, clone);
        applyGrad(gn, requiresGrad);
        return gn;
    }

    private static Conv1dImpl makeConv1d(LayerInfo info, Map<String, Tensor> roles,
                                         boolean requiresGrad, boolean clone) {
        long outC = ((Number) info.hyper.get("out_channels")).longValue();
        long inC = ((Number) info.hyper.get("in_channels")).longValue();
        long k = ((Number) info.hyper.get("kernel_size")).longValue();
        boolean withBias = Boolean.TRUE.equals(info.hyper.get("bias"));
        LongPointer kernel = new LongPointer(new long[]{k});
        Conv1dOptions opt = new Conv1dOptions(inC, outC, kernel).bias(withBias);
        Conv1dImpl conv = new Conv1dImpl(opt);
        copyInto(conv.weight(), roles.get("weight"), requiresGrad, clone);
        if (withBias && roles.get("bias") != null) {
            copyInto(conv.bias(), roles.get("bias"), requiresGrad, clone);
        }
        applyGrad(conv, requiresGrad);
        return conv;
    }

    private static Conv2dImpl makeConv2d(LayerInfo info, Map<String, Tensor> roles,
                                         boolean requiresGrad, boolean clone) {
        long outC = ((Number) info.hyper.get("out_channels")).longValue();
        long inC = ((Number) info.hyper.get("in_channels")).longValue();
        long[] ks = (long[]) info.hyper.get("kernel_size");
        boolean withBias = Boolean.TRUE.equals(info.hyper.get("bias"));
        LongPointer kernel = new LongPointer(new long[]{ks[0], ks[1]});
        Conv2dOptions opt = new Conv2dOptions(inC, outC, kernel).bias(withBias);
        Conv2dImpl conv = new Conv2dImpl(opt);
        copyInto(conv.weight(), roles.get("weight"), requiresGrad, clone);
        if (withBias && roles.get("bias") != null) {
            copyInto(conv.bias(), roles.get("bias"), requiresGrad, clone);
        }
        applyGrad(conv, requiresGrad);
        return conv;
    }

    private static Conv3dImpl makeConv3d(LayerInfo info, Map<String, Tensor> roles,
                                         boolean requiresGrad, boolean clone) {
        long outC = ((Number) info.hyper.get("out_channels")).longValue();
        long inC = ((Number) info.hyper.get("in_channels")).longValue();
        long[] ks = (long[]) info.hyper.get("kernel_size");
        boolean withBias = Boolean.TRUE.equals(info.hyper.get("bias"));
        LongPointer kernel = new LongPointer(new long[]{ks[0], ks[1], ks[2]});
        Conv3dOptions opt = new Conv3dOptions(inC, outC, kernel).bias(withBias);
        Conv3dImpl conv = new Conv3dImpl(opt);
        copyInto(conv.weight(), roles.get("weight"), requiresGrad, clone);
        if (withBias && roles.get("bias") != null) {
            copyInto(conv.bias(), roles.get("bias"), requiresGrad, clone);
        }
        applyGrad(conv, requiresGrad);
        return conv;
    }

    // ---- raw attach / copy helpers ------------------------------------------

    private static void attachRaw(Module target,
                                  String pathPrefix,
                                  Map<String, Tensor> roles,
                                  boolean requiresGrad,
                                  boolean clone,
                                  Map<String, Tensor> ownedOut) {
        for (Map.Entry<String, Tensor> e : roles.entrySet()) {
            String role = e.getKey();
            Tensor src = e.getValue();
            if (src == null || !src.defined()) continue;
            Tensor owned = clone ? safeClone(src) : src;
            boolean isBuf = isBufferRole(role);
            if (isBuf) {
                owned.requires_grad_(false);
                // NEVER store register_buffer return (ByRef)
                target.register_buffer(role, owned);
            } else {
                owned.requires_grad_(requiresGrad);
                target.register_parameter(role, owned, requiresGrad);
            }
            if (ownedOut != null) {
                String full = (pathPrefix == null || pathPrefix.isEmpty())
                        ? role
                        : pathPrefix + "." + role;
                ownedOut.put(full, owned);
            }
        }
    }

    private static void copyInto(Tensor dest, Tensor src, boolean requiresGrad, boolean clone) {
        if (dest == null || !dest.defined() || src == null || !src.defined()) return;
        Tensor use = clone ? safeClone(src) : src;
        try (NoGradGuard guard = new NoGradGuard()) {
            dest.copy_(use);
        }
        dest.requires_grad_(requiresGrad);
    }

    private static Tensor safeClone(Tensor src) {
        return src.detach().clone().contiguous();
    }

    private static void applyGrad(Module m, boolean requiresGrad) {
        try {
            if (requiresGrad) m.train(/*on=*/true);
            // named parameters: set requires_grad
            // Module.to / parameters iteration is heavy; leaves already set via copyInto
        } catch (Throwable ignored) {}
    }

    private static boolean isBufferRole(String role) {
        if (role == null) return false;
        String r = role.toLowerCase(Locale.ROOT);
        return r.equals("running_mean")
                || r.equals("running_var")
                || r.equals("num_batches_tracked")
                || r.endsWith("_mean")
                || r.endsWith("_var")
                || r.equals("mask")
                || r.equals("position_ids")
                || r.equals("token_type_ids");
    }

    /**
     * Return a Java-owned Tensor that aliases the Module leaf parameter.
     * Module accessors ({@code lin.weight()}, {@code emb.weight()}, …) are
     * {@code @ByRef} temporary views — storing them raw in a map makes later
     * {@code requires_grad_}/{@code numel} SIGBUS. The Tensor(Tensor) copy ctor
     * bumps the shared_ptr so the handle stays alive as long as we hold it
     * (and still points at the same storage as the Module parameter, so
     * freeze/Adam on the owned handle affect the live Module).
     */
    private static Tensor retainTensor(Tensor byRef) {
        if (byRef == null || byRef.isNull() || !byRef.defined()) return null;
        // Tensor copy ctor shares the underlying TensorImpl (refcount++).
        return new Tensor(byRef);
    }

    private static Tensor lookupOwnedFromLeaf(Module leaf, String role, LayerKind kind) {
        if (leaf == null || role == null) return null;
        try {
            switch (kind) {
                case LINEAR: {
                    LinearImpl lin = leaf.asLinear();
                    if (lin == null || lin.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(lin.weight());
                    if ("bias".equals(role)) return retainTensor(lin.bias());
                    break;
                }
                case EMBEDDING: {
                    EmbeddingImpl emb = leaf.asEmbedding();
                    if (emb == null || emb.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(emb.weight());
                    break;
                }
                case LAYER_NORM: {
                    LayerNormImpl ln = leaf.asLayerNorm();
                    if (ln == null || ln.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(ln.weight());
                    if ("bias".equals(role)) return retainTensor(ln.bias());
                    break;
                }
                case BATCH_NORM_1D: {
                    BatchNorm1dImpl m = leaf.asBatchNorm1d();
                    if (m == null || m.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(m.weight());
                    if ("bias".equals(role)) return retainTensor(m.bias());
                    if ("running_mean".equals(role)) return retainTensor(m.running_mean());
                    if ("running_var".equals(role)) return retainTensor(m.running_var());
                    if ("num_batches_tracked".equals(role)) return retainTensor(m.num_batches_tracked());
                    break;
                }
                case BATCH_NORM_2D: {
                    BatchNorm2dImpl m = leaf.asBatchNorm2d();
                    if (m == null || m.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(m.weight());
                    if ("bias".equals(role)) return retainTensor(m.bias());
                    if ("running_mean".equals(role)) return retainTensor(m.running_mean());
                    if ("running_var".equals(role)) return retainTensor(m.running_var());
                    if ("num_batches_tracked".equals(role)) return retainTensor(m.num_batches_tracked());
                    break;
                }
                case BATCH_NORM_3D: {
                    BatchNorm3dImpl m = leaf.asBatchNorm3d();
                    if (m == null || m.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(m.weight());
                    if ("bias".equals(role)) return retainTensor(m.bias());
                    if ("running_mean".equals(role)) return retainTensor(m.running_mean());
                    if ("running_var".equals(role)) return retainTensor(m.running_var());
                    if ("num_batches_tracked".equals(role)) return retainTensor(m.num_batches_tracked());
                    break;
                }
                case CONV_1D: {
                    Conv1dImpl c = leaf.asConv1d();
                    if (c == null || c.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(c.weight());
                    if ("bias".equals(role)) return retainTensor(c.bias());
                    break;
                }
                case CONV_2D: {
                    Conv2dImpl c = leaf.asConv2d();
                    if (c == null || c.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(c.weight());
                    if ("bias".equals(role)) return retainTensor(c.bias());
                    break;
                }
                case CONV_3D: {
                    Conv3dImpl c = leaf.asConv3d();
                    if (c == null || c.isNull()) break;
                    if ("weight".equals(role)) return retainTensor(c.weight());
                    if ("bias".equals(role)) return retainTensor(c.bias());
                    break;
                }
                default:
                    break;
            }
        } catch (Throwable ignored) {}
        return null;
    }

    // ---- path helpers -------------------------------------------------------

    private static String parentOf(String path) {
        if (path == null || path.isEmpty()) return "";
        int dot = path.lastIndexOf('.');
        return dot < 0 ? "" : path.substring(0, dot);
    }

    private static String leafNameOf(String path) {
        if (path == null || path.isEmpty()) return path;
        int dot = path.lastIndexOf('.');
        return dot < 0 ? path : path.substring(dot + 1);
    }

    private static int countDots(String s) {
        int n = 0;
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) == '.') n++;
        return n;
    }

    private static long[] shapeOf(Tensor t) {
        if (t == null || !t.defined()) return new long[0];
        long[] s = new long[(int) t.dim()];
        for (int i = 0; i < s.length; i++) s[i] = t.sizes().get(i);
        return s;
    }

    /** Prefer 32 groups when divisible; else largest divisor ≤ 32; else 1. */
    private static long guessNumGroups(long numChannels) {
        if (numChannels <= 0) return 1;
        if (numChannels % 32 == 0) return 32;
        if (numChannels % 16 == 0) return 16;
        if (numChannels % 8 == 0) return 8;
        if (numChannels % 4 == 0) return 4;
        if (numChannels % 2 == 0) return 2;
        return 1;
    }

    /** Pretty-print inferred layers (for conversion reports). */
    public static void printLayers(String title, List<LayerInfo> layers) {
        System.out.println("======== Inferred Modules: " + title + " ========");
        if (layers == null || layers.isEmpty()) {
            System.out.println("(none)");
            return;
        }
        Map<LayerKind, Integer> counts = new LinkedHashMap<>();
        for (LayerInfo li : layers) {
            counts.merge(li.kind, 1, Integer::sum);
        }
        System.out.print("kinds: ");
        boolean first = true;
        for (Map.Entry<LayerKind, Integer> e : counts.entrySet()) {
            if (!first) System.out.print(", ");
            System.out.print(e.getKey() + "=" + e.getValue());
            first = false;
        }
        System.out.println();
        System.out.printf(Locale.ROOT, "%-10s  %-40s  %s%n", "kind", "path", "hyper");
        int shown = 0;
        for (LayerInfo li : layers) {
            System.out.printf(Locale.ROOT, "%-10s  %-40s  %s%n",
                    li.kind,
                    truncate(li.path.isEmpty() ? "<root>" : li.path, 40),
                    li.hyper);
            if (++shown >= 200) {
                System.out.println("... (" + (layers.size() - shown) + " more)");
                break;
            }
        }
        System.out.println("================================================");
    }

    private static String truncate(String s, int max) {
        if (s == null) return "";
        if (s.length() <= max) return s;
        return s.substring(0, Math.max(0, max - 3)) + "...";
    }
}
