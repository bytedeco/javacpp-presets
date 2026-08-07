package org.bytedeco.pytorch.plot.vista;

import java.util.*;

import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModuleAsHelper;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * Builds a torchvista-style forward-pass graph from a live {@link Module}.
 *
 * <h2>What this ports from torchvista</h2>
 * <ul>
 *   <li>{@code process_graph}: module hierarchy walk, leaf forward recording,
 *       input/output nodes, edge dims, failed-node partial graphs, cleanup.</li>
 *   <li>{@code pre_trace_op} / {@code trace_op}: record a node before/after a
 *       module call, wire edges from tensors tagged with their source node.</li>
 *   <li>Sequential expansion: C++ {@code SequentialImpl.forward} is opaque, so
 *       we re-chain children ourselves to expose intermediate edges (same
 *       observable graph torchvista gets by wrapping each child's
 *       {@code forward}).</li>
 * </ul>
 *
 * <h2>What JavaCPP cannot do (documented gaps vs Python torchvista)</h2>
 * <ul>
 *   <li>No monkey-patch of {@code torch.*} / {@code Tensor} methods → free
 *       functional ops ({@code F.relu}, {@code x + y}, …) inside a custom
 *       {@code Module.forward} are <em>not</em> individual graph nodes. The
 *       custom module is one black-box leaf (or expanded only via
 *       {@link VistaOptions#forcedModuleTracingDepth}).</li>
 *   <li>No {@code tensor._tensor_source_name} attribute → tensor provenance is
 *       tracked with a {@link HashMap}{@code Long} keyed by
 *       {@link TensorUtils#tensorKey(Tensor)} (native address) for the duration
 *       of one trace.</li>
 *   <li>No {@code register_forward_hook} on C++ Module → we drive the walk
 *       ourselves rather than intercepting an opaque top-level forward.</li>
 * </ul>
 *
 * <p>Error tolerance matches torchvista: on exception the partial adjacency
 * list is kept, the exception is stored on {@link TraceGraph}, and the
 * caller / renderer can still visualise up to the failure point.
 */
public final class VistaEngine {

    private final VistaOptions options;
    private final TraceGraph graph;

    private int globalNodeCounter;
    private final List<String> moduleStack = new ArrayList<>();
    /** tensorKey (native address) → source node name for the current trace. */
    private final Map<Long, String> tensorSource = new HashMap<>();
    private final Map<Long, Boolean> tensorImplied = new HashMap<>();
    private final Set<String> seenEdges = new HashSet<>();
    /** Module identity → structural cat(embed) node created when runtime
     *  forward failed (e.g. MPS device). Used by post-processing to wire
     *  downstream modules to the cat node. */
    private final Map<Long, String> structuralEmbeddingCatNodes = new HashMap<>();
    /** Structural cat(embed) node name → inferred output dims string (e.g. "(4,16)"). */
    private final Map<String, String> structuralCatDims = new HashMap<>();
    private final Set<String> outputNodeSet = new HashSet<>();
    private final List<String> nodesToDelete = new ArrayList<>();
    private final List<String> constantNodeNames = new ArrayList<>();
    /** module identity (moduleObjectId/address) → attr name from parent.named_children */
    private final Map<Long, String> moduleToAttrName = new HashMap<>();
    /** module identity → list of node names created for that module (reuse) */
    private final Map<Long, List<String>> moduleToNodeNames = new HashMap<>();
    /** module identity → parent module identity (for compressed ModuleList injection) */
    private final Map<Long, Long> moduleHierarchy = new HashMap<>();
    private final Map<Long, Module> idToModule = new HashMap<>();
    /**
     * Feature catalog harvested from EmbeddingLayer / model fields before the
     * forward pass. Keyed by feature name (and common aliases). Used to annotate
     * Input nodes with sparse/dense/sequence metadata.
     */
    private final Map<String, Map<String, Object>> featureCatalog = new LinkedHashMap<>();
    /** Task / label names discovered on multi-task models (ctr, cvr, …). */
    private final List<String> taskLabelNames = new ArrayList<>();
    /** Raw result of the most recent successful top-level / module forward (may be Map). */
    private Object lastForwardResult;

    private String lastSuccessfulOp;
    private String currentOp;
    /** Depth of open leaf-module frames — free ops inside opaque leaves are suppressed. */
    private int insideLeafModuleDepth;
    /** True when traceLeafModule is called as a structural fallback (from
     *  runForwardSafe). When true, preTraceOp must NOT create constant nodes
     *  for untagged input tensors — they are structural placeholders, not
     *  real constants. */
    private boolean structuralFallback;

    public VistaEngine(VistaOptions options) {
        this.options = options == null ? VistaOptions.defaults() : options;
        this.graph = new TraceGraph();
    }

    public TraceGraph graph() {
        return graph;
    }

    // =========================================================================
    // Public entry
    // =========================================================================

    /**
     * Trace {@code model} with the given input(s) and fill {@link #graph()}.
     *
     * @param model  root module (must not be null)
     * @param inputs single {@link Tensor}, {@code Tensor[]}, {@code List}, or
     *               {@code Map} — same flexibility as torchvista
     * @return the filled graph (also accessible via {@link #graph()})
     */
    public TraceGraph process(Module model, Object inputs) {
        if (model == null || model.isNull()) {
            throw new IllegalArgumentException("model must be a live Module");
        }
        Module root = ModuleDiscovery.concrete(model);

        boolean wasTraining = false;
        try {
            wasTraining = root.is_training();
        } catch (Throwable ignored) {}

        if (options.evalMode()) {
            try {
                root.eval();
            } catch (Throwable ignored) {}
        }

        // Index attr names + hierarchy before the forward pass
        indexModuleTree(root, null, "", 0);
        // Collect Feature / task-name metadata from EmbeddingLayer children etc.
        // so Input/Output cards can show sparse/dense/sequence + label info.
        collectModelFeatureCatalog(root);

        List<Tensor> inputTensors = Collections.emptyList();
        // Bind free-op wrappers (VistaOps.relu/add/…) so custom Module.forward
        // can emit Operation nodes during this trace.
        VistaOps.bind(this);
        try (NoGradGuard guard = new NoGradGuard()) {
            inputTensors = tagInputs(inputs);
            lastForwardResult = null;
            Tensor output = runForward(root, inputs, /*stackDepth=*/0);
            // Prefer a full reflective forward result when the model returns a
            // named Map (MetaHeac ctr/cvr, multi-task heads) so each Output
            // node can carry the correct label name + shape.
            Object richOut = tryCaptureFullResult(root, inputs);
            Object forTag = richOut != null ? richOut : (lastForwardResult != null ? lastForwardResult : output);
            tagOutputs(forTag);
            enrichOutputMeta(root, forTag);
        } catch (Throwable e) {
            graph.setException(e);
            // Keep partial graph — torchvista error-tolerant behaviour
        } finally {
            VistaOps.unbind(this);
            // Restore training flag if we flipped it
            if (options.evalMode() && wasTraining) {
                try {
                    root.train(true);
                } catch (Throwable ignored) {}
            }
            // Clear tensor tags
            tensorSource.clear();
            tensorImplied.clear();
            lastForwardResult = null;
        }

        cleanupGraph();

        // If we produced a usable graph, drop poison-pill exceptions that only
        // reflect a fallback path we no longer take (e.g. historical
        // forward_tensor4 on Map models). Keep real partial-forward failures.
        if (graph.nodeCount() >= 3 && graph.exception() != null) {
            String msg = String.valueOf(graph.exception().getMessage());
            if (msg.contains("forward_tensor")
                    || msg.contains("not implemented for")
                    || msg.contains("refusing to explode feature map")) {
                graph.setException(null);
            }
        }

        if (options.showCompressedView()) {
            GraphTransforms.applyCompressedView(graph);
        }

        return graph;
    }

    // =========================================================================
    // Module tree indexing (attr names, hierarchy) — like traverse_model setup
    // =========================================================================

    private void indexModuleTree(Module module, Module parent, String parentPath, int depth) {
        if (module == null || module.isNull()) return;
        Module m = ModuleDiscovery.concrete(module);
        long id = moduleId(m);
        idToModule.put(id, m);
        if (parent != null) {
            moduleHierarchy.put(id, moduleId(parent));
        }
        for (ModuleChildren.NamedChild child : ModuleChildren.list(m)) {
            long cid = moduleId(child.module);
            String childPath = parentPath.isEmpty() ? child.key : parentPath + "." + child.key;
            if (!moduleToAttrName.containsKey(cid)) {
                // Prefer dotted path for non-root children so inspector shows
                // expertGate.linear / tower_0.sequential.0 instead of bare "linear"/"0".
                String attr = child.key;
                if (!parentPath.isEmpty() && attr != null && !attr.isEmpty()) {
                    attr = childPath;
                }
                moduleToAttrName.put(cid, attr);
            }
            idToModule.put(cid, child.module);
            moduleHierarchy.put(cid, id);

            boolean walkDeeper;
            Integer forced = options.forcedModuleTracingDepth();
            if (forced != null) {
                walkDeeper = depth < forced || !ModuleDiscovery.hasForwardMethod(child.module);
            } else {
                walkDeeper = true;
            }
            if (walkDeeper && ModuleChildren.hasChildren(child.module)) {
                int nextDepth = ModuleDiscovery.hasForwardMethod(child.module) ? depth + 1 : depth;
                indexModuleTree(child.module, m, childPath, nextDepth);
            }
        }
    }

    // =========================================================================
    // Forward walk
    // =========================================================================

    /**
     * Drive a traced forward. Expansion rules:
     * <ul>
     *   <li>Sequential → re-chain children (exposes intermediate edges).</li>
     *   <li>ModuleList / ModuleDict / Parameter* → transparent walk only if
     *       something calls into them; they have no forward of their own.</li>
     *   <li>Leaf (built-in or black-box custom) → single node + real forward.</li>
     *   <li>When {@code forcedModuleTracingDepth} allows, composites with
     *       children are expanded by walking children in named_children order
     *       and calling each child's forward sequentially (best-effort; does
     *       not reconstruct arbitrary custom control flow).</li>
     * </ul>
     */
    private Tensor runForward(Module module, Object inputs, int stackDepth) {
        Module m = ModuleDiscovery.concrete(module);

        // Root / nested Sequential → expand children (exposes intermediate edges)
        if (ModuleDiscovery.isSequential(m)
                && !ModuleDiscovery.isTracedLeaf(m, stackDepth, options.forcedModuleTracingDepth())) {
            return expandSequential(m, inputs, stackDepth);
        }

        // Custom Java Module with overridden forward — NON-INVASIVE:
        // never requires model source to call VistaOps. The engine walks
        // named_children and invokes each child itself (structure + shapes).
        // Free ops inside an opaque forward remain invisible unless the *user*
        // optionally wraps them with VistaOps in their own code.
        if (ModuleDiscovery.isCustomForwardModule(m)
                && !ModuleDiscovery.isTracedLeaf(m, stackDepth, options.forcedModuleTracingDepth())) {
            if (ModuleDiscovery.canChainChildrenAsSequential(m)) {
                return expandSequential(m, inputs, stackDepth);
            }
            if (ModuleChildren.hasChildren(m)) {
                return expandCustomFromChildren(m, inputs, stackDepth);
            }
            // No children: opaque leaf (whole forward as one node)
            return traceLeafModule(m, inputs, stackDepth);
        }

        Integer forced = options.forcedModuleTracingDepth();
        boolean depthAllows = forced == null || stackDepth < forced;

        // Weight bag / single-child wrappers: only chain when topology is actually
        // sequential (or a single Sequential child like recommend MLP). Multi-branch
        // models (SharedBottom / MMOE / ESMM) must NOT be false-chained.
        if (depthAllows
                && ModuleChildren.hasChildren(m)
                && !ModuleDiscovery.isBuiltinLeaf(m)
                && !ModuleDiscovery.isCustomForwardModule(m)) {
            if (ModuleDiscovery.isSequential(m)
                    || ModuleDiscovery.canChainChildrenAsSequential(m)) {
                return expandSequential(m, inputs, stackDepth);
            }
            if (ModuleDiscovery.isModuleListLike(m)
                    || ModuleDiscovery.isModuleDictLike(m)
                    || ModuleDiscovery.isParameterListLike(m)
                    || ModuleDiscovery.isParameterDictLike(m)) {
                // transparent: walk kids at same depth, no container leaf
                return expandParallelChildren(m, inputs, stackDepth, /*transparent=*/true);
            }
            // Multi-child composite without Tensor/Map forward we can drive:
            // structural parallel expand (container frame + each child).
            return expandParallelChildren(m, inputs, stackDepth, /*transparent=*/false);
        }

        // Forced expansion of composite modules when depth allows.
        if (forced != null
                && stackDepth < forced
                && ModuleChildren.hasChildren(m)
                && !ModuleDiscovery.isContainer(m)
                && !ModuleDiscovery.isBuiltinLeaf(m)) {
            if (ModuleDiscovery.isSequential(m)
                    || ModuleDiscovery.canChainChildrenAsSequential(m)) {
                return expandSequential(m, inputs, stackDepth);
            }
            return expandParallelChildren(m, inputs, stackDepth, /*transparent=*/false);
        }

        // Built-in *Impl or opaque leaf
        return traceLeafModule(m, inputs, stackDepth);
    }

    /**
     * Expand children under a container without inventing a false Sequential
     * chain. Used for multi-task models (towers in parallel) and ModuleDict.
     * When {@code inputs} is a Tensor and a child accepts Tensor forward, each
     * child is invoked with the <em>same</em> input (fan-out); outputs are not
     * fed into siblings. Returns the last successful child output (or null).
     */
    private Tensor expandParallelChildren(Module composite, Object inputs, int stackDepth,
                                          boolean transparent) {
        Module m = ModuleDiscovery.concrete(composite);
        String frame = transparent ? null : beginContainerFrame(m);
        if (!transparent) {
            recordOpParameters(frame, inputs);
        }
        Tensor last = null;
        try {
            for (ModuleChildren.NamedChild child : ModuleChildren.list(m)) {
                try {
                    Tensor out = runForward(child.module, inputs, stackDepth + (transparent ? 0 : 1));
                    if (out != null && !out.isNull()) last = out;
                } catch (Throwable childEx) {
                    // Keep going so sibling towers still appear; record failure on child
                    if (graph.exception() == null) graph.setException(childEx);
                }
            }
        } finally {
            if (frame != null) endContainerFrame(frame);
        }
        return last;
    }

    private Tensor expandSequential(Module sequential, Object inputs, int stackDepth) {
        Module m = ModuleDiscovery.concrete(sequential);
        List<ModuleChildren.NamedChild> kids = ModuleChildren.list(m);
        // MLP pattern: when a custom module has both named children (layer_0,
        // layer_1, …) AND a "sequential" SequentialImpl child containing the
        // same modules, skip the sequential container to avoid duplicate
        // nodes and cyclic edges. Only chain the named children.
        if (!ModuleDiscovery.isSequential(m)) {
            boolean hasSeq = false;
            for (ModuleChildren.NamedChild k : kids) {
                if ("sequential".equals(k.key) && (ModuleDiscovery.isSequential(k.module) || ModuleDiscovery.isModuleListLike(k.module))) {
                    hasSeq = true; break;
                }
            }
            if (hasSeq) {
                List<ModuleChildren.NamedChild> filtered = new ArrayList<>();
                for (ModuleChildren.NamedChild k : kids) {
                    if (!("sequential".equals(k.key) && (ModuleDiscovery.isSequential(k.module) || ModuleDiscovery.isModuleListLike(k.module)))) {
                        filtered.add(k);
                    }
                }
                if (!filtered.isEmpty()) kids = filtered;
            }
        }
        if (kids.isEmpty()) {
            // Empty sequential — just forward if possible
            try {
                return callForward(m, inputs);
            } catch (Throwable e) {
                return null;
            }
        }

        // Push a container frame so children get correct ancestors / collapsible grouping.
        String containerName = beginContainerFrame(m);

        Object current = inputs;
        Tensor lastTensor = null;
        String lastNodeName = null; // track last emitted node for structural chaining
        boolean broken = false;
        List<String> structChildNames = new ArrayList<>();
        List<String> allChildNames = new ArrayList<>();
        try {
            for (int i = 0; i < kids.size(); i++) {
                ModuleChildren.NamedChild child = kids.get(i);
                if (broken || current == null) {
                    // Chain is broken — emit remaining kids structurally but also
                    // keep implied chaining from the last successful child to the
                    // first node under the next structural child so the UI shows
                    // continued flow even when runtime forward failed.
                    String leaf = emitStructuralChildReturning(child.module, child.key, stackDepth + 1);
                    if (leaf != null) { structChildNames.add(leaf); allChildNames.add(leaf); }
                    if (leaf != null) {
                        String fromNode = lastNodeName;
                        if (fromNode == null && lastTensor != null) {
                            fromNode = tensorSource.get(TensorUtils.tensorKey(lastTensor));
                        }
                        if (fromNode != null) linkImplied(fromNode, leaf);
                        lastNodeName = leaf;
                    }
                    broken = true;
                    continue;
                }
                try {
                    Tensor out = runForward(child.module, current, stackDepth + 1);
                    if (out == null || out.isNull()) {
                        // Retry with a 2D-flattened input when the child is a
                        // Linear/MLP-like layer that expects a matmul-compatible
                        // input but received a 3D+ embedding tensor (e.g. DeepFM
                        // feeds (4,1,8) into MLP whose first Linear expects (4,8)).
                        Tensor flat = flattenForRetry(current);
                        if (flat != null) {
                            out = runForward(child.module, flat, stackDepth + 1);
                        }
                    }
                    if (out != null && !out.isNull()) {
                        lastTensor = out;
                        current = out;
                        // Update lastNodeName from tensor source
                        String src = tensorSource.get(TensorUtils.tensorKey(out));
                        if (src != null) { lastNodeName = src; allChildNames.add(src); }
                    } else {
                        // Child produced nothing usable — keep structure for the rest
                        broken = true;
                        // If runForward already recorded a node, fine; otherwise structural
                        if (!hasRecordedChild(child.module)) {
                            String leaf = emitStructuralChildReturning(child.module, child.key, stackDepth + 1);
                            if (leaf != null) {
                                String fromNode = lastNodeName;
                                if (fromNode == null && lastTensor != null) {
                                    fromNode = tensorSource.get(TensorUtils.tensorKey(lastTensor));
                                }
                                if (fromNode != null) linkImplied(fromNode, leaf);
                                lastNodeName = leaf;
                            }
                        } else {
                            // Node already created by preTraceOp — still need to
                            // link from previous node and update lastNodeName.
                            String existing = lastRecordedChildName(child.module);
                            if (existing != null) {
                                String fromNode = lastNodeName;
                                if (fromNode == null && lastTensor != null) {
                                    fromNode = tensorSource.get(TensorUtils.tensorKey(lastTensor));
                                }
                                if (fromNode != null) linkImplied(fromNode, existing);
                                lastNodeName = existing;
                            }
                        }
                    }
                } catch (Throwable childEx) {
                    // Shape mismatch? Retry with a 2D-flattened input — many
                    // recommend models do view(batch, -1) before MLP/Linear.
                    Tensor retryOut = retryWithFlatten(child.module, current, stackDepth, childEx);
                    if (retryOut != null && !retryOut.isNull()) {
                        lastTensor = retryOut;
                        current = retryOut;
                        String src = tensorSource.get(TensorUtils.tensorKey(retryOut));
                        if (src != null) { lastNodeName = src; allChildNames.add(src); }
                        continue;
                    }
                    // Record partial failure but NEVER abort the rest of the Sequential —
                    // missing ReLU/Linear after a failed Linear is worse than a failed flag.
                    if (graph.exception() == null) {
                        String msg = String.valueOf(childEx.getMessage());
                        if (!msg.contains("forward_tensor")
                                && !msg.contains("refusing to explode")
                                && !msg.contains("No matching forward")) {
                            graph.setException(childEx);
                        }
                    }
                    if (!hasRecordedChild(child.module)) {
                        String leaf = emitStructuralChildReturning(child.module, child.key, stackDepth + 1);
                        if (leaf != null) {
                            String fromNode = lastNodeName;
                            if (fromNode == null && lastTensor != null) {
                                fromNode = tensorSource.get(TensorUtils.tensorKey(lastTensor));
                            }
                            if (fromNode != null) linkImplied(fromNode, leaf);
                            lastNodeName = leaf;
                        }
                    } else {
                        // Node already created by preTraceOp — still need to
                        // link from previous node and update lastNodeName.
                        String existing = lastRecordedChildName(child.module);
                        if (existing != null) {
                            String fromNode = lastNodeName;
                            if (fromNode == null && lastTensor != null) {
                                fromNode = tensorSource.get(TensorUtils.tensorKey(lastTensor));
                            }
                            if (fromNode != null) linkImplied(fromNode, existing);
                            lastNodeName = existing;
                        }
                    }
                    broken = true;
                    // Do not feed a failed/null tensor into the next layer
                    current = null;
                }
            }

            // Fallback: if the chain broke and structural children have no
            // outgoing edges, try whole-module forward and wire them to the
            // real output source. This prevents EmbeddingImpl structural nodes
            // from becoming sinks that get incorrectly connected to model output.
            if (broken && !structChildNames.isEmpty()) {
                try {
                    Tensor whole = callForward(m, inputs);
                    if (whole != null && !whole.isNull()) {
                        long key = TensorUtils.tensorKey(whole);
                        String realSource = tensorSource.get(key);
                        if (realSource == null && containerName != null) {
                            realSource = containerName;
                            tensorSource.put(key, containerName);
                        }
                        if (realSource != null) {
                            lastTensor = whole;
                            for (String sname : allChildNames) {
                                if (sname == null || sname.equals(realSource)) continue;
                                GraphNode sn = graph.adjList().get(sname);
                                if (sn == null) continue;
                                if (!sn.edges().isEmpty()) continue;
                                sn.addEdge(new GraphEdge(realSource, "", 0L, false));
                            }
                        }
                    }
                } catch (Throwable ignored) {}
            }
        } finally {
            if (containerName != null) {
                endContainerFrame(containerName);
            }
        }
        return lastTensor;
    }

    /**
     * Flatten a 3D+ tensor to 2D {@code (batch, -1)} for retry when a Linear/MLP
     * child rejects a multi-dimensional embedding input. Returns null when the
     * input is already 2D or cannot be flattened (not a Tensor, null, etc.).
     */
    private Tensor flattenForRetry(Object current) {
        if (!(current instanceof Tensor)) return null;
        Tensor t = (Tensor) current;
        if (t.isNull()) return null;
        try {
            int dim = (int) t.dim();
            if (dim <= 1) return null; // nothing to flatten
            if (dim == 2) return null; // already 2D
            long batch = t.size(0);
            if (batch <= 0) return null;
            Tensor flat = t.view(batch, -1);
            // Inherit tensorSource from the original so preTraceOp doesn't
            // create a spurious constant node for the reshaped tensor.
            String src = tensorSource.get(TensorUtils.tensorKey(t));
            if (src != null) tensorSource.put(TensorUtils.tensorKey(flat), src);
            return flat;
        } catch (Throwable ignored) {
            return null;
        }
    }

    /**
     * Try to reshape {@code input} so its last dimension matches the
     * {@code in_features} of a LinearImpl layer. When the total numel is
     * divisible by {@code inFeatures}, produces {@code (batch, inFeatures)}.
     * Returns null when not applicable.
     */
    private Tensor reshapeForLinear(Module child, Object current) {
        if (!(current instanceof Tensor)) return null;
        Tensor t = (Tensor) current;
        if (t.isNull()) return null;
        try {
            Module typed = ModuleDiscovery.concrete(child);
            if (!(typed instanceof LinearImpl)) return null;
            LinearImpl lin = (LinearImpl) typed;
            Tensor w = lin.weight();
            if (w == null || w.isNull()) return null;
            long inFeatures = w.size(1);
            if (inFeatures <= 0) return null;
            long numel = t.numel();
            if (numel % inFeatures != 0) return null;
            long batch = numel / inFeatures;
            if (batch <= 0) return null;
            Tensor reshaped = t.view(batch, inFeatures);
            // Inherit tensorSource from the original so preTraceOp doesn't
            // create a spurious constant node for the reshaped tensor.
            String src = tensorSource.get(TensorUtils.tensorKey(t));
            if (src != null) tensorSource.put(TensorUtils.tensorKey(reshaped), src);
            return reshaped;
        } catch (Throwable ignored) {
            return null;
        }
    }

    /**
     * Retry a child forward after reshaping the input. Used when the
     * original forward threw a shape-mismatch error (e.g. "mat1 and mat2 shapes
     * cannot be multiplied"). This mirrors the common {@code view(batch, -1)}
     * pattern in recommend model forwards (DeepFM, …).
     *
     * <p>Strategy: try Linear-weight-aware reshape first (matches in_features),
     * then fall back to plain 2D flatten.
     *
     * @return the retry output, or null if retry was not applicable / failed
     */
    private Tensor retryWithFlatten(Module child, Object current, int stackDepth, Throwable originalEx) {
        String msg = String.valueOf(originalEx.getMessage());
        boolean shapeMismatch = msg.contains("cannot be multiplied")
                || msg.contains("shapes cannot be")
                || msg.contains("size mismatch")
                || msg.contains("RuntimeError: mat1 and mat2");
        if (!shapeMismatch) return null;
        // Strategy 1: reshape to match Linear's expected in_features
        Tensor reshaped = reshapeForLinear(child, current);
        if (reshaped != null) {
            try {
                Tensor out = runForward(child, reshaped, stackDepth + 1);
                if (out != null && !out.isNull()) return out;
            } catch (Throwable ignored) {}
        }
        // Strategy 2: plain 2D flatten
        Tensor flat = flattenForRetry(current);
        if (flat != null) {
            try {
                return runForward(child, flat, stackDepth + 1);
            } catch (Throwable ignored) {}
        }
        return null;
    }

    /**
     * True if this module already contributed graph content in this trace.
     *
     * <p>Container frames (Sequential/MLP) are NOT themselves in {@code adj_list},
     * but they do get a name via {@link #beginContainerFrame} and book children
     * under {@code parent_module_to_nodes}. Treating only adj_list hits as
     * "recorded" caused a second full structural re-emit of the same Sequential
     * (duplicate Linear/ReLU chains under a second SequentialImpl_*).
     */
    private boolean hasRecordedChild(Module module) {
        if (module == null) return false;
        try {
            Module m = ModuleDiscovery.concrete(module);
            Long id = moduleId(m);
            List<String> names = moduleToNodeNames.get(id);
            if (names != null) {
                for (String n : names) {
                    if (graph.adjList().containsKey(n)) return true;
                    List<String> kids = graph.parentModuleToNodes().get(n);
                    if (kids != null && !kids.isEmpty()) return true;
                }
            }
            // Any direct child already recorded?
            for (ModuleChildren.NamedChild c : ModuleChildren.list(m)) {
                Long cid = moduleId(ModuleDiscovery.concrete(c.module));
                List<String> cn = moduleToNodeNames.get(cid);
                if (cn == null) continue;
                for (String n : cn) {
                    if (graph.adjList().containsKey(n)) return true;
                    List<String> kids = graph.parentModuleToNodes().get(n);
                    if (kids != null && !kids.isEmpty()) return true;
                }
            }
        } catch (Throwable ignored) {}
        return false;
    }

    private String lastRecordedChildName(Module module) {
        if (module == null) return null;
        try {
            Module m = ModuleDiscovery.concrete(module);
            Long id = moduleId(m);
            List<String> names = moduleToNodeNames.get(id);
            if (names != null) {
                for (int i = names.size() - 1; i >= 0; i--) {
                    String n = names.get(i);
                    if (graph.adjList().containsKey(n)) return n;
                }
            }
        } catch (Throwable ignored) {}
        return null;
    }

    /**
     * Best-effort expansion for custom composites under forced depth:
     * feed the same inputs through each child in order only when the previous
     * output is a Tensor (linear pipeline assumption). Stops and falls back to
     * leaf forward if a child cannot accept the current tensor.
     */
    private Tensor expandCompositeBestEffort(Module composite, Object inputs, int stackDepth) {
        Module m = ModuleDiscovery.concrete(composite);
        String containerName = beginContainerFrame(m);
        Object current = inputs;
        Tensor lastTensor = null;
        boolean anyChildRan = false;
        try {
            for (ModuleChildren.NamedChild child : ModuleChildren.list(m)) {
                try {
                    Tensor out = runForward(child.module, current, stackDepth + 1);
                    if (out != null && !out.isNull()) {
                        lastTensor = out;
                        current = out;
                        anyChildRan = true;
                    }
                } catch (Throwable childEx) {
                    // Child failed — mark and rethrow to keep partial graph
                    throw childEx;
                }
            }
            if (!anyChildRan) {
                // Nothing ran — fall back to whole-module forward as leaf
                endContainerFrame(containerName);
                containerName = null;
                return traceLeafModule(m, inputs, stackDepth);
            }
        } finally {
            if (containerName != null) {
                endContainerFrame(containerName);
            }
        }
        return lastTensor;
    }

    /**
     * Non-invasive expansion of a custom multi-child module.
     *
     * <p><b>Does not modify model source.</b> The engine walks
     * {@code named_children()} and invokes each child itself:
     * <ol>
     *   <li>Map-accepting / {@code *embed*} children get the feature map.</li>
     *   <li>A shared trunk ({@code *bottom*}/{@code *shared*}/{@code *backbone*})
     *       consumes the embed output.</li>
     *   <li>Parallel {@code tower_i} / {@code expert_i} fan-out from the shared
     *       tensor; {@code predictLayer_i} / {@code head_i} pairs with
     *       {@code tower_i} by numeric suffix.</li>
     *   <li>Other Tensor children chain from the previous Tensor output.</li>
     *   <li>If nothing ran, fall back to one opaque whole-module leaf.</li>
     * </ol>
     * Free ops inside the original {@code forward} body stay invisible unless
     * the <em>user</em> optionally wraps them with {@link VistaOps} in their
     * own code — models are never patched by vista.
     */
    private Tensor expandCustomFromChildren(Module module, Object inputs, int stackDepth) {
        Module m = ModuleDiscovery.concrete(module);
        String frame = beginContainerFrame(m);
        recordOpParameters(frame, inputs);

        List<ModuleChildren.NamedChild> kids = ModuleChildren.list(m);

        // ── Unpack multi-arg payloads ────────────────────────────────────────
        // Supports:
        //   Map features                         (SharedBottom / MMOE …)
        //   Object[]{Map, Tensor taskIdx}        (MetaHeac)
        //   Tensor[]{tokens, positions, …}       (LLM4Rec / HLLM …)
        //   single Tensor
        Object mapInput = null;
        Tensor taskIdxTensor = null;
        Tensor rootTensor = null;
        // Ordered tensor args so EVERY input gets edges to the right child
        // (input_0 → tokenEmbedding, input_1 → positionEmbedding, …).
        List<Tensor> tensorArgs = new ArrayList<>();
        if (inputs instanceof Map) {
            mapInput = inputs;
        } else if (inputs instanceof Tensor) {
            rootTensor = (Tensor) inputs;
            tensorArgs.add(rootTensor);
        } else if (inputs instanceof Tensor[]) {
            Tensor[] ta = (Tensor[]) inputs;
            for (Tensor t : ta) {
                if (t != null && !t.isNull()) tensorArgs.add(t);
            }
            if (!tensorArgs.isEmpty()) rootTensor = tensorArgs.get(0);
            if (tensorArgs.size() > 1) taskIdxTensor = tensorArgs.get(1);
        } else if (inputs instanceof Object[] && !(inputs instanceof Tensor[])) {
            Object[] arr = (Object[]) inputs;
            // Prefer explicit order: [0]=Map, [1]=taskIdx  OR  all-Tensor multi-arg
            boolean anyMap = false;
            for (Object o : arr) if (o instanceof Map) { anyMap = true; break; }
            if (anyMap) {
                for (Object o : arr) {
                    if (o instanceof Map && mapInput == null) mapInput = o;
                    else if (o instanceof Tensor) {
                        tensorArgs.add((Tensor) o);
                    }
                }
                if (arr.length >= 1 && arr[0] instanceof Map) mapInput = arr[0];
                if (arr.length >= 2 && arr[1] instanceof Tensor) taskIdxTensor = (Tensor) arr[1];
                if (!tensorArgs.isEmpty() && rootTensor == null) rootTensor = tensorArgs.get(0);
                if (taskIdxTensor == null && tensorArgs.size() > 1) taskIdxTensor = tensorArgs.get(1);
            } else {
                for (Object o : arr) {
                    if (o instanceof Tensor && !((Tensor) o).isNull()) tensorArgs.add((Tensor) o);
                }
                if (!tensorArgs.isEmpty()) rootTensor = tensorArgs.get(0);
                if (tensorArgs.size() > 1) taskIdxTensor = tensorArgs.get(1);
            }
        } else if (inputs instanceof List) {
            List<?> list = (List<?>) inputs;
            if (!list.isEmpty() && list.get(0) instanceof Map) {
                mapInput = list.get(0);
                if (list.size() > 1 && list.get(1) instanceof Tensor) {
                    taskIdxTensor = (Tensor) list.get(1);
                    tensorArgs.add(taskIdxTensor);
                }
            } else {
                for (Tensor t : TensorUtils.extractTensors(inputs)) {
                    if (t != null && !t.isNull()) tensorArgs.add(t);
                }
                if (!tensorArgs.isEmpty()) rootTensor = tensorArgs.get(0);
                if (tensorArgs.size() > 1) taskIdxTensor = tensorArgs.get(1);
            }
        } else {
            for (Tensor t : TensorUtils.extractTensors(inputs)) {
                if (t != null && !t.isNull()) tensorArgs.add(t);
            }
            if (!tensorArgs.isEmpty()) rootTensor = tensorArgs.get(0);
        }

        // Classify kids: feature-embed (Map) vs task-embed (Tensor idx) vs rest.
        // CRITICAL: bare key "embedding" is ONLY a feature-embed when we actually
        // have a Map input. Nested MetaEmbedding.embedding (EmbeddingImpl) must
        // receive the Tensor taskIdx — classifying it as feature-embed with
        // mapInput=null caused structural-only emit and broke MetaHeac gates.
        List<ModuleChildren.NamedChild> featEmbedKids = new ArrayList<>();
        List<ModuleChildren.NamedChild> taskEmbedKids = new ArrayList<>();
        List<ModuleChildren.NamedChild> otherKids = new ArrayList<>();
        for (ModuleChildren.NamedChild child : kids) {
            String k = child.key == null ? "" : child.key.toLowerCase();
            Module cm = ModuleDiscovery.concrete(child.module);
            if (isTaskEmbedKey(k)) {
                taskEmbedKids.add(child);
            } else if (mapInput != null && isFeatureEmbedKey(k, cm)) {
                featEmbedKids.add(child);
            } else {
                otherKids.add(child);
            }
        }

        Tensor embedOut = null;
        List<Tensor> embedParts = new ArrayList<>();
        List<String> embedNodeNames = new ArrayList<>();
        List<String> allChildNodeNames = new ArrayList<>();
        Tensor sharedTensor = null;
        Map<String, Tensor> taggedOutputs = new HashMap<>();
        Tensor lastOut = null;
        boolean anyChild = false;
        int successCount = 0;
        int downstreamSuccess = 0;

        try {
            // Pass 1a — feature embeddings (need Map)
            for (ModuleChildren.NamedChild child : featEmbedKids) {
                Module cm = ModuleDiscovery.concrete(child.module);
                String key = child.key == null ? "" : child.key;
                Object childIn = mapInput != null ? mapInput : null;
                // Prefer expandEmbeddingTables so each EmbeddingImpl appears
                if (childIn != null && looksLikeEmbeddingLayer(cm)) {
                    try {
                        anyChild = true;
                        Tensor out = expandEmbeddingLayer(cm, (Map<?, ?>) childIn, stackDepth + 1);
                        if (out != null && !out.isNull()) {
                            lastOut = out;
                            successCount++;
                            taggedOutputs.put(key.toLowerCase(), out);
                            embedParts.add(out);
                            String srcName = tensorSource.get(TensorUtils.tensorKey(out));
                            if (srcName != null) { embedNodeNames.add(srcName); allChildNodeNames.add(srcName); }
                            continue;
                        }
                        // Runtime forward failed but a structural cat(embed)
                        // node may have been created. Register it so downstream
                        // modules can be wired to it.
                        String structCat = structuralEmbeddingCatNodes.get(moduleId(cm));
                        if (structCat != null) {
                            embedNodeNames.add(structCat);
                            allChildNodeNames.add(structCat);
                        }
                    } catch (Throwable ignored) {}
                }
                if (childIn == null) {
                    // Structural fallback — still show the EmbeddingLayer tree
                    String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                    if (sname != null) allChildNodeNames.add(sname);
                    anyChild = true;
                    continue;
                }
                try {
                    anyChild = true;
                    Tensor out = runForwardSafe(cm, childIn, stackDepth + 1);
                    if (out != null && !out.isNull()) {
                        lastOut = out;
                        successCount++;
                        taggedOutputs.put(key.toLowerCase(), out);
                        embedParts.add(out);
                        String srcName = tensorSource.get(TensorUtils.tensorKey(out));
                        if (srcName != null) { embedNodeNames.add(srcName); allChildNodeNames.add(srcName); }
                    } else if (!hasRecordedChild(cm)) {
                        String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                        if (sname != null) allChildNodeNames.add(sname);
                    } else {
                        String sname = lastRecordedChildName(cm);
                        if (sname != null) allChildNodeNames.add(sname);
                    }
                } catch (Throwable childEx) {
                    noteChildException(childEx);
                    if (!hasRecordedChild(cm)) {
                        String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                        if (sname != null) allChildNodeNames.add(sname);
                    } else {
                        String sname = lastRecordedChildName(cm);
                        if (sname != null) allChildNodeNames.add(sname);
                    }
                }
            }

            // Build shared trunk from feature embeddings
            if (embedParts.size() == 1) {
                embedOut = embedParts.get(0);
                sharedTensor = embedOut;
            } else if (embedParts.size() >= 2) {
                try {
                    Tensor catOut = catAlongDim1(embedParts, embedNodeNames);
                    embedOut = catOut;
                    sharedTensor = catOut;
                    lastOut = catOut;
                    successCount++;
                } catch (Throwable catEx) {
                    embedOut = embedParts.get(embedParts.size() - 1);
                    sharedTensor = embedOut;
                }
            }

            // Pass 1b — task embeddings (MetaHeac taskEmbedding_i needs Long indices)
            for (ModuleChildren.NamedChild child : taskEmbedKids) {
                Module cm = ModuleDiscovery.concrete(child.module);
                String key = child.key == null ? "" : child.key;
                Object childIn = taskIdxTensor;
                if (childIn == null && rootTensor != null) {
                    // Prefer integer index tensors for EmbeddingImpl task tables
                    try {
                        org.bytedeco.pytorch.global.torch.ScalarType st =
                                rootTensor.scalar_type().intern();
                        if (st == org.bytedeco.pytorch.global.torch.ScalarType.Long
                                || st == org.bytedeco.pytorch.global.torch.ScalarType.Int
                                || st == org.bytedeco.pytorch.global.torch.ScalarType.Short
                                || st == org.bytedeco.pytorch.global.torch.ScalarType.Byte) {
                            childIn = rootTensor;
                        }
                    } catch (Throwable ignored) {
                        childIn = rootTensor;
                    }
                }
                if (childIn == null) {
                    String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                    if (sname != null) allChildNodeNames.add(sname);
                    anyChild = true;
                    continue;
                }
                try {
                    anyChild = true;
                    Tensor out = runForwardSafe(cm, childIn, stackDepth + 1);
                    if (out != null && !out.isNull()) {
                        lastOut = out;
                        successCount++;
                        taggedOutputs.put(key.toLowerCase(), out);
                        String sname = tensorSource.get(TensorUtils.tensorKey(out));
                        if (sname != null) allChildNodeNames.add(sname);
                    } else if (!hasRecordedChild(cm)) {
                        String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                        if (sname != null) allChildNodeNames.add(sname);
                    } else {
                        String sname = lastRecordedChildName(cm);
                        if (sname != null) allChildNodeNames.add(sname);
                    }
                } catch (Throwable childEx) {
                    noteChildException(childEx);
                    if (!hasRecordedChild(cm)) {
                        String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                        if (sname != null) allChildNodeNames.add(sname);
                    } else {
                        String sname = lastRecordedChildName(cm);
                        if (sname != null) allChildNodeNames.add(sname);
                    }
                }
            }

            // Pass 2 — token/pos embeds, norms, bottoms, towers, experts, gates, critics
            // Track which multi-arg tensor slots have already been claimed so
            // every input node gets at least one outgoing edge (LLM4Rec etc.).
            boolean[] tensorArgUsed = new boolean[tensorArgs.size()];
            for (ModuleChildren.NamedChild child : otherKids) {
                Module cm = ModuleDiscovery.concrete(child.module);
                String key = child.key == null ? "" : child.key;
                String keyL = key.toLowerCase();
                String pairIdx = trailingIndex(keyL);
                String typeSimple = ModuleDiscovery.simpleTypeName(cm).toLowerCase();

                Object childIn = null;
                boolean isGate = keyL.contains("gate");
                boolean isCriticMlp = keyL.contains("critic") && !isGate;
                boolean isPredictLayer = keyL.contains("predict") || keyL.startsWith("pred")
                        || keyL.contains("output_layer")
                        || (keyL.contains("head") && !keyL.contains("gate"));
                boolean isNorm = keyL.contains("norm") || typeSimple.contains("norm")
                        || typeSimple.contains("layernorm") || typeSimple.contains("batchnorm");
                // OneRec/HSTU/HLLM/LLM4Rec: pos_emb, positionEmbedding, pos_embedding, …
                boolean isPosEmbed = isPosEmbedKey(keyL);
                // Bare "embedding"/"embed" is ONLY a token table when the root looks like
                // token ids (Long/Int). VQ/RQVAE register codebook as "embedding" but feed
                // float latents into nearest-neighbour lookup — never call EmbeddingImpl
                // with those floats (indices are computed inside parent forward).
                boolean rootLooksLikeTokenIds = rootTensor != null && isIndexTensor(rootTensor);
                boolean isTokenEmbed = keyL.contains("token") || keyL.contains("tok_emb")
                        || keyL.contains("wte") || keyL.contains("tokenembedding")
                        || keyL.equals("token_embedding") || keyL.equals("tokenembedding")
                        || keyL.contains("item_emb") || keyL.equals("itemembedding")
                        || ((keyL.equals("embed") || keyL.equals("embedding")) && rootLooksLikeTokenIds)
                        || (typeSimple.contains("embedding") && !isPosEmbed
                            && (keyL.contains("token") || keyL.contains("word") || keyL.contains("item")));
                if (!isPosEmbed && typeSimple.contains("embedding")
                        && (keyL.contains("token") || keyL.contains("word") || keyL.contains("item")
                        || keyL.equals("tokenembedding") || keyL.equals("token_embedding")
                        || ((keyL.equals("embedding") || keyL.equals("embed")) && rootLooksLikeTokenIds))) {
                    isTokenEmbed = true;
                }
                // Codebook / residual-VQ EmbeddingImpl: structural only (parent does NN lookup)
                boolean isCodebookEmbed = !isPosEmbed && !isTokenEmbed
                        && typeSimple.contains("embedding")
                        && (keyL.equals("embedding") || keyL.equals("embed") || keyL.isEmpty())
                        && rootTensor != null && isFloatTensor(rootTensor);

                // ── Multi-tensor / synthesized routing for token & position embeds ──
                // Generative models often take ONLY token ids and synthesize positions
                // inside forward (OneRec / OneRecV2 / OpenOneRec / HSTU / HLLM). Vista must
                // mirror that: never feed float activations or raw token ids into pos_emb.
                if (childIn == null && isPosEmbed) {
                    childIn = resolvePosEmbedInput(cm, tensorArgs, rootTensor, tensorArgUsed);
                } else if (childIn == null && isTokenEmbed && !tensorArgs.isEmpty()) {
                    Tensor tok = firstIndexTensor(tensorArgs);
                    if (tok != null) {
                        childIn = tok;
                        markTensorArgUsed(tensorArgs, tensorArgUsed, tok);
                    } else if (rootLooksLikeTokenIds) {
                        childIn = rootTensor;
                        if (!tensorArgs.isEmpty()) tensorArgUsed[0] = true;
                    }
                } else if (childIn == null && keyL.contains("time") && tensorArgs.size() > 1) {
                    // HLLM/HSTU timeDiffs often arg[1]
                    childIn = tensorArgs.get(1);
                    tensorArgUsed[1] = true;
                } else if (childIn == null && isCodebookEmbed) {
                    // Do not forward float latents into EmbeddingImpl — structural only.
                    childIn = null;
                }

                // predict/head/critic-MLP paired with tower_i / expert_i by suffix
                if (childIn == null && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && pairIdx != null && (keyL.contains("predict") || keyL.contains("head")
                        || keyL.startsWith("pred") || keyL.contains("output_layer")
                        || isCriticMlp)) {
                    childIn = findTagged(taggedOutputs, pairIdx,
                            "tower", "task", "branch", "critic", "bottom");
                }
                if (childIn == null && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && (keyL.contains("bottom") || keyL.contains("shared")
                        || keyL.contains("backbone") || keyL.contains("trunk"))) {
                    childIn = firstNonNull(sharedTensor, embedOut, rootTensor);
                }
                // experts / towers fan-out from shared embed (exclude *gate*)
                // AITM: tower_i receives bottom_i's output by index pairing,
                // not the raw shared embedding.
                if (childIn == null && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && !isGate && keyL.contains("tower") && pairIdx != null) {
                    childIn = findTagged(taggedOutputs, pairIdx, "bottom", "ait");
                }
                // MoE (MMOE/PLE): tower receives gated mixture of experts —
                // use any available expert output (all experts share the same
                // output dim) so the tower's first Linear gets a compatible input.
                if (childIn == null && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && !isGate && keyL.contains("tower")) {
                    childIn = findAnyTagged(taggedOutputs, "expert");
                }
                if (childIn == null && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && !isGate && (keyL.contains("tower") || keyL.contains("expert")
                        || keyL.startsWith("tower"))) {
                    childIn = firstNonNull(sharedTensor, embedOut);
                }
                // gates FIRST — before generic linear/mlp chain (MetaLinear is
                // canChainChildrenAsSequential and would otherwise steal raw embed).
                // AITM infoGate_k receives bottom_k's output (information transfer).
                if (childIn == null && isGate && keyL.contains("infogate") && pairIdx != null) {
                    childIn = findTagged(taggedOutputs, pairIdx, "bottom");
                }
                if (childIn == null && isGate) {
                    childIn = synthesizeGateInput(taggedOutputs, pairIdx, sharedTensor, embedOut);
                }
                // Norm / Linear / MLP after embeddings: prefer float activations, never raw Long ids
                // (skip gates / pos / token embeds — already handled)
                // AITM ait_k (AttentionLayer) receives cat(bottom_(k+1), infoGate_k).
                // Both have already run by the time ait_k is processed, so create a
                // cat node joining their outputs.
                if (childIn == null && !isGate && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && keyL.contains("ait") && pairIdx != null) {
                    int kIdx = Integer.parseInt(pairIdx);
                    String nextIdx = String.valueOf(kIdx + 1);
                    Tensor bottomNext = findTagged(taggedOutputs, nextIdx, "bottom");
                    Tensor infoGate = findTagged(taggedOutputs, pairIdx, "infogate", "gate");
                    if (bottomNext != null && infoGate != null) {
                        try {
                            org.bytedeco.pytorch.TensorVector tv =
                                    new org.bytedeco.pytorch.TensorVector(bottomNext, infoGate);
                            Tensor catOut = org.bytedeco.pytorch.global.torch.cat(tv, 1L);
                            globalNodeCounter++;
                            String catName = "cat_" + globalNodeCounter;
                            graph.graphNodeNameToWithoutSuffix().put(catName, "cat");
                            graph.graphNodeDisplayNames().put(catName, "cat(bottom,gate)");
                            graph.nodeToModulePath().put(catName, "torch");
                            GraphNode catNode = GraphNode.of(NodeType.OPERATION);
                            graph.adjList().put(catName, catNode);
                            String bottomSrc = tensorSource.get(TensorUtils.tensorKey(bottomNext));
                            String gateSrc = tensorSource.get(TensorUtils.tensorKey(infoGate));
                            if (bottomSrc != null) {
                                GraphNode bn = graph.adjList().get(bottomSrc);
                                if (bn != null) bn.addEdge(new GraphEdge(catName, "", 0L, true));
                            }
                            if (gateSrc != null) {
                                GraphNode gn = graph.adjList().get(gateSrc);
                                if (gn != null) gn.addEdge(new GraphEdge(catName, "", 0L, true));
                            }
                            tensorSource.put(TensorUtils.tensorKey(catOut), catName);
                            childIn = catOut;
                        } catch (Throwable ignored) {
                            childIn = firstNonNull(bottomNext, infoGate,
                                    firstNonNull(sharedTensor, embedOut));
                        }
                    } else {
                        childIn = firstNonNull(bottomNext, infoGate,
                                firstNonNull(sharedTensor, embedOut));
                    }
                }
                if (childIn == null && !isGate && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && (isNorm || keyL.contains("mlp") || keyL.contains("proj")
                        || keyL.contains("head") || keyL.contains("fc") || keyL.contains("linear")
                        || typeSimple.contains("linear") || typeSimple.contains("sequential")
                        || ModuleDiscovery.canChainChildrenAsSequential(cm))) {
                    childIn = firstNonNull(sharedTensor, embedOut, lastOut);
                    if (childIn == null && rootTensor != null && isFloatTensor(rootTensor)) {
                        childIn = rootTensor;
                    }
                }
                // CGC/PLE: when input is a multi-tensor List, route each
                // expert/gate to the tensor matching its index suffix. Shared
                // experts/gates get the last tensor (shared input).
                if (childIn == null && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && tensorArgs.size() > 1 && pairIdx != null
                        && (keyL.contains("expert") || keyL.contains("gate"))) {
                    try {
                        int idx = Integer.parseInt(pairIdx);
                        if (keyL.contains("shared")) {
                            childIn = tensorArgs.get(tensorArgs.size() - 1);
                            tensorArgUsed[tensorArgs.size() - 1] = true;
                        } else if (idx < tensorArgs.size()) {
                            childIn = tensorArgs.get(idx);
                            tensorArgUsed[idx] = true;
                        }
                    } catch (NumberFormatException ignored) {}
                }
                // Default: chain from previous successful float output, else root
                // NEVER re-route pos/token/codebook embeds through this fallback — that is
                // exactly how float activations were incorrectly fed into EmbeddingImpl.
                // Skip predictLayer — it should only receive tower output, not
                // sharedTensor/embedOut. If tower failed, predictLayer will be
                // emitted structurally and wired to tower's last node in
                // post-processing.
                if (childIn == null && !isPosEmbed && !isTokenEmbed && !isCodebookEmbed
                        && !isPredictLayer) {
                    Tensor prefer = firstNonNull(sharedTensor, embedOut, lastOut);
                    if (prefer != null && isFloatTensor(prefer)) {
                        childIn = prefer;
                    } else if (isNorm || typeSimple.contains("linear")) {
                        childIn = null;
                    } else {
                        childIn = firstNonNull(prefer, rootTensor);
                    }
                }
                // Never feed Map into Sequential/MLP/Linear
                if (childIn instanceof Map
                        && (ModuleDiscovery.isSequential(cm)
                        || ModuleDiscovery.canChainChildrenAsSequential(cm)
                        || ModuleDiscovery.isBuiltinLeaf(cm))) {
                    childIn = firstNonNull(sharedTensor, embedOut, null);
                }
                // Don't feed integer index tensors into Norm / non-Embedding Linear
                if (childIn instanceof Tensor && !isFloatTensor((Tensor) childIn)
                        && (isNorm || (typeSimple.contains("linear") && !typeSimple.contains("embedding")))) {
                    childIn = firstNonNull(sharedTensor, embedOut, lastOut);
                    if (childIn instanceof Tensor && !isFloatTensor((Tensor) childIn)) {
                        childIn = null;
                    }
                }
                // Hard guard: EmbeddingImpl always needs Long/Int indices
                if (childIn instanceof Tensor && typeSimple.contains("embedding")
                        && !typeSimple.contains("layer") && isFloatTensor((Tensor) childIn)) {
                    if (isPosEmbed) {
                        childIn = resolvePosEmbedInput(cm, tensorArgs, rootTensor, tensorArgUsed);
                    } else {
                        childIn = null;
                    }
                }
                if (childIn == null) {
                    String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                    if (sname != null) allChildNodeNames.add(sname);
                    // Still wire an implied edge from the best-matching input so the
                    // graph shows the connection even when real forward can't run.
                    // Skip predictLayer — its input is wired in post-processing
                    // to the matching tower's last node.
                    if (!isPredictLayer) {
                        wireImpliedFromInputs(cm, keyL, isPosEmbed, isTokenEmbed, isNorm, tensorArgs);
                    }
                    anyChild = true;
                    continue;
                }
                try {
                    anyChild = true;
                    Tensor out = runForwardSafe(cm, childIn, stackDepth + 1);
                    if (out != null && !out.isNull()) {
                        lastOut = out;
                        successCount++;
                        downstreamSuccess++;
                        taggedOutputs.put(keyL, out);
                        String sname = tensorSource.get(TensorUtils.tensorKey(out));
                        if (sname != null) allChildNodeNames.add(sname);
                        // Per-task bottoms (AITM: bottom_0, bottom_1, …) must NOT
                        // hijack sharedTensor — otherwise bottom_1 receives
                        // bottom_0's output instead of the embedding. Only a bare
                        // "bottom" (SharedBottom) or shared/backbone/trunk updates
                        // the shared trunk.
                        boolean isPerTaskBottom = keyL.contains("bottom") && pairIdx != null;
                        if ((keyL.contains("bottom") && !isPerTaskBottom)
                                || keyL.contains("shared")
                                || keyL.contains("backbone") || keyL.contains("trunk")) {
                            sharedTensor = out;
                        }
                        // Token/pos embedding outputs become the activation trunk
                        if (isTokenEmbed || isPosEmbed || typeSimple.contains("embedding")) {
                            if (embedOut == null) embedOut = out;
                            // Prefer float embedding as shared for downstream norm/mlp
                            if (isFloatTensor(out)) {
                                sharedTensor = out;
                                embedOut = out;
                            }
                        }
                        if (isNorm && isFloatTensor(out)) {
                            sharedTensor = out;
                        }
                    } else if (!hasRecordedChild(cm)) {
                        String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                        if (sname != null) allChildNodeNames.add(sname);
                        wireImpliedFromInputs(cm, keyL, isPosEmbed, isTokenEmbed, isNorm, tensorArgs);
                    } else {
                        String sname = lastRecordedChildName(cm);
                        if (sname != null) allChildNodeNames.add(sname);
                    }
                } catch (Throwable childEx) {
                    noteChildException(childEx);
                    if (!hasRecordedChild(cm)) {
                        String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                        if (sname != null) allChildNodeNames.add(sname);
                        wireImpliedFromInputs(cm, keyL, isPosEmbed, isTokenEmbed, isNorm, tensorArgs);
                    } else {
                        String sname = lastRecordedChildName(cm);
                        if (sname != null) allChildNodeNames.add(sname);
                    }
                }
            }
            // Ensure every multi-arg input tensor that wasn't routed still has a
            // visible edge into the graph (prevents orphan input_1 with no arrows).
            for (int ti = 0; ti < tensorArgs.size(); ti++) {
                if (tensorArgUsed[ti]) continue;
                Tensor t = tensorArgs.get(ti);
                if (t == null || t.isNull()) continue;
                long key = TensorUtils.tensorKey(t);
                String inName = tensorSource.get(key);
                if (inName == null) continue;
                GraphNode inNode = graph.adjList().get(inName);
                if (inNode == null) continue;
                if (!inNode.edges().isEmpty()) continue; // already has edges
                // Link to first non-input module as implied, or to output later
                String target = null;
                for (String n : graph.adjList().keySet()) {
                    GraphNode gn = graph.adjList().get(n);
                    if (gn != null && gn.nodeType() == NodeType.MODULE) {
                        target = n;
                        break;
                    }
                }
                if (target != null) {
                    inNode.addEdge(new GraphEdge(target, TensorUtils.formatDims(t), key, true));
                }
            }

            // If child expansion produced almost nothing useful, try whole-module
            // forward as a fallback leaf (still keep any structural kids we emitted).
            if (!anyChild) {
                endContainerFrame(frame);
                frame = null;
                if (inputs instanceof Map || inputs instanceof List
                        || (inputs instanceof Object[] && !(inputs instanceof Tensor[]))) {
                    try {
                        return traceLeafModule(m, inputs, stackDepth);
                    } catch (Throwable e) {
                        emitStructuralChild(m, null, stackDepth);
                        return null;
                    }
                }
                return traceLeafModule(m, inputs, stackDepth);
            }

            // Hybrid: if many kids failed shape-wise but whole module can run,
            // run whole-module once to attach real output dims (doesn't remove structure).
            boolean allDownstreamFailed = downstreamSuccess == 0 && !otherKids.isEmpty();
            if ((successCount == 0 && lastOut == null) || allDownstreamFailed) {
                try {
                    Tensor whole = callForward(m, inputs);
                    if (whole != null && !whole.isNull()) {
                        // Tag output provenance — create a proper operation node
                        // (e.g. "add" for DeepFM's linearOut.add(fmOut).add(mlpOut))
                        // so it appears in adjList and can receive edges.
                        long key = TensorUtils.tensorKey(whole);
                        String realSource = tensorSource.get(key);
                        if (realSource == null) {
                            globalNodeCounter++;
                            String opName = "output_op_" + globalNodeCounter;
                            graph.graphNodeNameToWithoutSuffix().put(opName, "output");
                            graph.graphNodeDisplayNames().put(opName, "add");
                            graph.nodeToModulePath().put(opName, "torch");
                            GraphNode opNode = GraphNode.of(NodeType.OPERATION);
                            graph.adjList().put(opName, opNode);
                            recordParentBookkeeping(opName);
                            graph.nodeToAncestors().put(opName, currentAncestors());
                            tensorSource.put(key, opName);
                            realSource = opName;
                        }
                        lastOut = whole;
                        // Clear poison exception if whole forward worked
                        graph.setException(null);

                        // Embedding source for wiring embedding → failed children
                        String embedSource = null;
                        if (sharedTensor != null && !sharedTensor.isNull()) {
                            embedSource = tensorSource.get(TensorUtils.tensorKey(sharedTensor));
                        }
                        if (embedSource == null && embedOut != null && !embedOut.isNull()) {
                            embedSource = tensorSource.get(TensorUtils.tensorKey(embedOut));
                        }
                        // Structural fallback: if runtime forward failed but a
                        // structural cat(embed) node was created, use it as the
                        // embedding source so downstream modules get wired to it.
                        if (embedSource == null && !embedNodeNames.isEmpty()) {
                            embedSource = embedNodeNames.get(embedNodeNames.size() - 1);
                        }

                        // Build a set of embedding source node names so we can
                        // distinguish them from downstream modules (linear/fm/mlp).
                        // Embedding nodes must NOT connect directly to the output op —
                        // they feed into the downstream modules which then feed output.
                        Set<String> embedNodeSet = new HashSet<>(embedNodeNames);
                        if (embedSource != null) embedNodeSet.add(embedSource);

                        // Wire downstream children → output op, and embedding → downstream children.
                        // The graph should show: embedding → linear/fm/mlp → add(output)
                        if (realSource != null) {
                            for (String sname : allChildNodeNames) {
                                if (sname == null || sname.equals(realSource)) continue;
                                // Skip embedding nodes — they feed children, not output
                                if (embedNodeSet.contains(sname)) continue;
                                GraphNode sn = graph.adjList().get(sname);
                                if (sn == null) continue;

                                // downstream child → output op
                                boolean hasOutEdge = false;
                                for (GraphEdge ge : sn.edges()) {
                                    if (realSource.equals(ge.target())) { hasOutEdge = true; break; }
                                }
                                if (!hasOutEdge) {
                                    sn.addEdge(new GraphEdge(realSource, "", 0L, false));
                                }

                                // embedding → downstream child (only if child has no incoming)
                                if (embedSource != null) {
                                    boolean hasIncoming = false;
                                    for (GraphNode gn : graph.adjList().values()) {
                                        for (GraphEdge ge : gn.edges()) {
                                            if (sname.equals(ge.target())) { hasIncoming = true; break; }
                                        }
                                        if (hasIncoming) break;
                                    }
                                    if (!hasIncoming) {
                                        GraphNode es = graph.adjList().get(embedSource);
                                        if (es != null) {
                                            boolean hasEmbEdge = false;
                                            for (GraphEdge ge : es.edges()) {
                                                if (sname.equals(ge.target())) { hasEmbEdge = true; break; }
                                            }
                                            if (!hasEmbEdge) {
                                                String embDims = structuralCatDims.getOrDefault(embedSource, "");
                                                es.addEdge(new GraphEdge(sname, embDims, 0L, false));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } catch (Throwable wholeEx) {
                    noteChildException(wholeEx);
                }
            }

            if (graph.exception() != null) {
                String msg = String.valueOf(graph.exception().getMessage());
                if (msg.contains("forward_tensor") || msg.contains("not implemented")
                        || msg.contains("refusing to explode")
                        || msg.contains("No matching forward")
                        || msg.contains("index out of range")
                        || msg.contains("mat1 and mat2")) {
                    // Keep structural graph; drop noisy shape-mismatch banner when
                    // we still have a usable multi-node graph.
                    if (graph.nodeCount() >= 4) {
                        graph.setException(null);
                    }
                }
            }

            // Post-process: wire predictLayer/head nodes to their matching
            // tower/expert's last structural node. When tower forward failed,
            // predictLayer has no real tensor input and would otherwise be
            // left unconnected (or incorrectly wired to sharedTensor/output
            // by the final sink-fan-out pass). Also collect predictLayer node
            // names so the Final pass below can skip them — predictLayer
            // should connect to output, not to bestTarget (which may be a
            // sibling predictLayer).
            Set<String> predictLayerNodes = new HashSet<>();
            for (ModuleChildren.NamedChild predKid : kids) {
                String pk = predKid.key == null ? "" : predKid.key.toLowerCase();
                boolean isPredLayer = pk.contains("predict") || pk.startsWith("pred")
                        || pk.contains("output_layer")
                        || (pk.contains("head") && !pk.contains("gate"));
                if (!isPredLayer) continue;
                String predIdx = trailingIndex(pk);
                if (predIdx == null) continue;
                // Match predictLayer_i to tower_i (prefer "tower" over "expert"
                // since both share the same index suffix).
                Module towerMod = null;
                for (ModuleChildren.NamedChild tk : kids) {
                    String tkKey = tk.key == null ? "" : tk.key.toLowerCase();
                    if (!tkKey.contains("tower")) continue;
                    String tkIdx = trailingIndex(tkKey);
                    if (predIdx.equals(tkIdx)) {
                        towerMod = ModuleDiscovery.concrete(tk.module);
                        break;
                    }
                }
                if (towerMod == null) continue;
                String towerLast = findLastLeafInGraph(towerMod);
                if (towerLast == null) continue;
                String predNode = findLastLeafInGraph(predKid.module);
                if (predNode == null) continue;
                predictLayerNodes.add(predNode);
                // Remove any existing incoming edges to predictLayer from
                // non-tower nodes (e.g. sharedTensor, expert, gate) that were
                // created by wireImpliedFromInputs or the Default fallback.
                for (Map.Entry<String, GraphNode> entry : graph.adjList().entrySet()) {
                    if (entry.getKey().equals(towerLast)) continue;
                    entry.getValue().edges().removeIf(ge -> predNode.equals(ge.target()));
                }
                // Add the correct tower → predictLayer edge if not already present
                GraphNode tn = graph.adjList().get(towerLast);
                if (tn != null) {
                    boolean already = false;
                    for (GraphEdge ge : tn.edges()) {
                        if (predNode.equals(ge.target())) { already = true; break; }
                    }
                    if (!already) {
                        tn.addEdge(new GraphEdge(predNode, "", 0L, true));
                    }
                }
                // Also remove any wrong outgoing edges from towerLast (e.g.
                // towerLast → output added by sink-fan-out) since towerLast
                // should feed predictLayer, not output directly.
                GraphNode tNode = graph.adjList().get(towerLast);
                if (tNode != null) {
                    tNode.edges().removeIf(ge -> "output".equals(ge.target())
                            || ge.target().startsWith("output"));
                }
            }

            // Final pass: wire any child nodes that still have no outgoing
            // edges to the best available target (sharedTensor/embedOut/lastOut
            // source). This prevents EmbeddingImpl structural nodes from
            // becoming sinks that get incorrectly connected to model output.// Prefer sharedTensor/embedOut over lastOut — lastOut is the last
            // child's output which may be a parallel branch endpoint (expert,
            // gate, tower) that should NOT receive fan-in from siblings.
            String bestTarget = null;
            if (sharedTensor != null && !sharedTensor.isNull()) {
                bestTarget = tensorSource.get(TensorUtils.tensorKey(sharedTensor));
            }
            if (bestTarget == null && embedOut != null && !embedOut.isNull()) {
                bestTarget = tensorSource.get(TensorUtils.tensorKey(embedOut));
            }
            if (bestTarget == null && lastOut != null && !lastOut.isNull()) {
                bestTarget = tensorSource.get(TensorUtils.tensorKey(lastOut));
            }
            // Structural fallback: if all runtime forwards failed but a
            // structural cat(embed) node was created, use it as bestTarget.
            if (bestTarget == null && !embedNodeNames.isEmpty()) {
                bestTarget = embedNodeNames.get(embedNodeNames.size() - 1);
            }
            if (bestTarget != null) {
                for (String sname : allChildNodeNames) {
                    if (sname == null || sname.equals(bestTarget)) continue;
                    if (predictLayerNodes.contains(sname)) continue;
                    GraphNode sn = graph.adjList().get(sname);
                    if (sn == null) continue;
                    if (!sn.edges().isEmpty()) continue;
                    // Skip nodes that already have incoming edges — they are
                    // parallel branch endpoints (expert/gate/tower outputs)
                    // whose output is consumed downstream, not orphans.
                    boolean hasIncoming = false;
                    for (GraphNode gn : graph.adjList().values()) {
                        for (GraphEdge ge : gn.edges()) {
                            if (sname.equals(ge.target())) { hasIncoming = true; break; }
                        }
                        if (hasIncoming) break;
                    }
                    if (hasIncoming) continue;
                    sn.addEdge(new GraphEdge(bestTarget, "", 0L, false));
                }
            }

            // Structural cat(embed) fan-out: when runtime forward failed and
            // a structural cat node was created, wire it to downstream child
            // nodes (linear/fm/mlp/tower/expert) that have no incoming edges.
            // This ensures the graph shows: embedding → cat(embed) → downstream.
            if (!embedNodeNames.isEmpty()) {
                String catNode = embedNodeNames.get(embedNodeNames.size() - 1);
                GraphNode catGN = graph.adjList().get(catNode);
                if (catGN != null) {
                    String catDims = structuralCatDims.getOrDefault(catNode, "");
                    for (String sname : allChildNodeNames) {
                        if (sname == null || sname.equals(catNode)) continue;
                        // Skip embedding nodes — they feed cat, not vice-versa
                        if (embedNodeNames.contains(sname)) continue;
                        GraphNode sn = graph.adjList().get(sname);
                        if (sn == null) continue;
                        // Check if sname already has incoming edges
                        boolean hasIncoming = false;
                        for (GraphNode gn : graph.adjList().values()) {
                            for (GraphEdge ge : gn.edges()) {
                                if (sname.equals(ge.target())) { hasIncoming = true; break; }
                            }
                            if (hasIncoming) break;
                        }
                        if (hasIncoming) continue;
                        // Check if cat already has edge to sname
                        boolean hasEdge = false;
                        for (GraphEdge ge : catGN.edges()) {
                            if (sname.equals(ge.target())) { hasEdge = true; break; }
                        }
                        if (!hasEdge) {
                            catGN.addEdge(new GraphEdge(sname, catDims, 0L, false));
                        }
                    }
                }
            }
        } finally {
            if (frame != null) endContainerFrame(frame);
        }
        return lastOut;
    }

    /** Feature embedding tables (EmbeddingLayer) — NOT taskEmbedding_*. */
    private static boolean isFeatureEmbedKey(String keyL, Module cm) {
        if (keyL == null) keyL = "";
        if (isTaskEmbedKey(keyL)) return false;
        if (keyL.equals("embedding") || keyL.equals("embed") || keyL.startsWith("embed_")
                || keyL.endsWith("_embedding") || keyL.contains("embeddinglayer")
                || keyL.contains("user_embed") || keyL.contains("item_embed")
                || keyL.contains("sparse_embed")) {
            return true;
        }
        // encoder used as feature encoder (ESMM dual towers) — not transformer encoder
        if (keyL.contains("encoder") && !keyL.contains("transformer") && !keyL.contains("hstu")) {
            return acceptsMapForward(cm) || keyL.contains("embed");
        }
        return acceptsMapForward(cm);
    }

    private static boolean isTaskEmbedKey(String keyL) {
        if (keyL == null) return false;
        return keyL.contains("taskembed") || keyL.contains("task_embed")
                || keyL.startsWith("taskembedding") || keyL.contains("task_emb");
    }

    private static boolean looksLikeEmbeddingLayer(Module m) {
        if (m == null) return false;
        String n = ModuleDiscovery.simpleTypeName(m).toLowerCase();
        if (n.contains("embeddinglayer") || n.equals("embedding")) return true;
        // Has many EmbeddingImpl children and accepts Map
        if (!acceptsMapForward(m)) return false;
        int embKids = 0;
        for (ModuleChildren.NamedChild c : ModuleChildren.list(m)) {
            String cn = ModuleDiscovery.simpleTypeName(c.module).toLowerCase();
            if (cn.contains("embedding")) embKids++;
        }
        return embKids >= 1;
    }

    /**
     * Expand EmbeddingLayer non-invasively: each EmbeddingImpl table becomes a
     * graph node fed by its feature input; outputs are concatenated.
     */
    private Tensor expandEmbeddingLayer(Module embLayer, Map<?, ?> sparseFeats, int stackDepth) {
        Module m = ModuleDiscovery.concrete(embLayer);
        String frame = beginContainerFrame(m);
        List<Tensor> parts = new ArrayList<>();
        List<String> partNames = new ArrayList<>();
        List<String> structChildNames = new ArrayList<>();
        try {
            // Prefer real EmbeddingLayer.forward(Map) — one shot, correct dims.
            // Still expand children structurally so tables are visible.
            Tensor real = null;
            try {
                real = callForward(m, sparseFeats);
            } catch (Throwable ignored) {
                System.err.println("[DEBUG expandEmbeddingLayer] callForward FAILED: " + ignored);
            }
            System.err.println("[DEBUG expandEmbeddingLayer] module=" + ModuleDiscovery.typeName(m)
                    + " real=" + (real == null ? "null" : (real.isNull() ? "null(isNull)" : "non-null")));

            for (ModuleChildren.NamedChild child : ModuleChildren.list(m)) {
                Module cm = ModuleDiscovery.concrete(child.module);
                String key = child.key == null ? "" : child.key;
                String base = key.startsWith("embed_") ? key.substring("embed_".length()) : key;
                // Find matching feature tensor
                Tensor idx = null;
                if (sparseFeats != null) {
                    System.err.println("[DEBUG expandEmbeddingLayer] child key=" + key + " base=" + base
                            + " sparseFeats keys=" + sparseFeats.keySet());
                    Object v = sparseFeats.get(base);
                    if (v == null) {
                        // try raw key / without prefix
                        for (Map.Entry<?, ?> e : sparseFeats.entrySet()) {
                            String fk = String.valueOf(e.getKey());
                            if (fk.equals(base) || fk.endsWith(base) || base.endsWith(fk)
                                    || ("embed_" + fk).equals(key)) {
                                if (e.getValue() instanceof Tensor) {
                                    idx = (Tensor) e.getValue();
                                    break;
                                }
                            }
                        }
                    } else if (v instanceof Tensor) {
                        idx = (Tensor) v;
                    }
                }
                System.err.println("[DEBUG expandEmbeddingLayer] child key=" + key
                        + " idx=" + (idx == null ? "null" : "non-null")
                        + " cmType=" + ModuleDiscovery.simpleTypeName(cm)
                        + " isEmbedding=" + ModuleDiscovery.simpleTypeName(cm).toLowerCase().contains("embedding"));
                if (idx != null && ModuleDiscovery.simpleTypeName(cm).toLowerCase().contains("embedding")) {
                    try {
                        Tensor out = traceLeafModule(cm, idx, stackDepth + 1);
                        System.err.println("[DEBUG expandEmbeddingLayer] traceLeafModule for " + key
                                + " out=" + (out == null ? "null" : (out.isNull() ? "null(isNull)" : "non-null")));
                        if (out != null && !out.isNull()) {
                            parts.add(out);
                            String src = tensorSource.get(TensorUtils.tensorKey(out));
                            if (src != null) partNames.add(src);
                        }
                    } catch (Throwable ex) {
                        System.err.println("[DEBUG expandEmbeddingLayer] traceLeafModule THREW for " + key
                                + " err=" + ex.getClass().getSimpleName() + ": " + ex.getMessage());
                        // traceLeafModule failed (e.g. MPS placeholder storage issue
                        // after callForward moved tensors to device). Emit a
                        // structural node and record it for later wiring to the
                        // real output source (cat node).
                        if (!hasRecordedChild(cm)) {
                            String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                            if (sname != null) structChildNames.add(sname);
                        } else {
                            String sname = lastRecordedChildName(cm);
                            if (sname != null) structChildNames.add(sname);
                        }
                    }
                } else {
                    if (!hasRecordedChild(cm)) {
                        String sname = emitStructuralChildReturning(cm, key, stackDepth + 1);
                        if (sname != null) structChildNames.add(sname);
                    } else {
                        String sname = lastRecordedChildName(cm);
                        if (sname != null) structChildNames.add(sname);
                    }
                }
            }

            System.err.println("[DEBUG expandEmbeddingLayer] after loop: parts=" + parts.size()
                    + " partNames=" + partNames + " structChildNames=" + structChildNames);

            if (real != null && !real.isNull()) {
                // Prefer real forward output for downstream chaining.
                // Create a proper operation node so downstream modules can
                // trace edges from it (container frames are not in adjList).
                long key = TensorUtils.tensorKey(real);
                String realSource = tensorSource.get(key);
                System.err.println("[DEBUG expandEmbeddingLayer] real non-null, realSource=" + realSource);
                if (realSource == null) {
                    int totalCount = partNames.size() + structChildNames.size();
                    if (totalCount <= 1 && !partNames.isEmpty()) {
                        realSource = partNames.get(0);
                        tensorSource.put(key, realSource);
                    } else {
                        globalNodeCounter++;
                        String catName = "cat_" + globalNodeCounter;
                        graph.graphNodeNameToWithoutSuffix().put(catName, "cat");
                        graph.graphNodeDisplayNames().put(catName, "cat(embed)");
                        graph.nodeToModulePath().put(catName, "torch");
                        GraphNode catNode = GraphNode.of(NodeType.OPERATION);
                        graph.adjList().put(catName, catNode);
                        recordParentBookkeeping(catName);
                        graph.nodeToAncestors().put(catName, currentAncestors());
                        tensorSource.put(key, catName);
                        realSource = catName;
                        System.err.println("[DEBUG expandEmbeddingLayer] CREATED cat node: " + catName
                                + " totalCount=" + totalCount);
                    }
                }
                // Wire edges from each traced EmbeddingImpl part to the real
                // output source (cat node or last part). Without this, the
                // individual EmbeddingImpl nodes have no outgoing edges and
                // become "sinks" that the fan-out logic incorrectly connects
                // directly to the model output.
                if (realSource != null) {
                    for (int pi = 0; pi < parts.size(); pi++) {
                        String src = pi < partNames.size() ? partNames.get(pi)
                                : tensorSource.get(TensorUtils.tensorKey(parts.get(pi)));
                        if (src == null || src.equals(realSource)) continue;
                        GraphNode sn = graph.adjList().get(src);
                        if (sn == null) continue;
                        boolean hasEdge = false;
                        for (GraphEdge ge : sn.edges()) {
                            if (realSource.equals(ge.target())) { hasEdge = true; break; }
                        }
                        if (!hasEdge) {
                            sn.addEdge(new GraphEdge(realSource,
                                    TensorUtils.formatDims(parts.get(pi)),
                                    TensorUtils.tensorKey(parts.get(pi)), false));
                        }
                    }
                    // Also wire structural child nodes (from failed traceLeafModule)
                    // to the real output source so they don't become sinks.
                    for (String sname : structChildNames) {
                        if (sname == null || sname.equals(realSource)) continue;
                        GraphNode sn = graph.adjList().get(sname);
                        if (sn == null) continue;
                        boolean hasEdge = false;
                        for (GraphEdge ge : sn.edges()) {
                            if (realSource.equals(ge.target())) { hasEdge = true; break; }
                        }
                        if (!hasEdge) {
                            sn.addEdge(new GraphEdge(realSource, "", 0L, false));
                        }
                    }
                }
                return real;
            }
            if (parts.size() == 1) return parts.get(0);
            if (parts.size() >= 2) {
                try {
                    return catAlongDim1(parts, partNames);
                } catch (Throwable e) {
                    return parts.get(parts.size() - 1);
                }
            }
            // Structural fallback: real forward failed (e.g. MPS device issue)
            // and no runtime tensors were captured, but we have multiple
            // structural embedding children. Create a structural cat(embed)
            // node so the concatenation appears in the graph and wire the
            // structural children to it.
            if (structChildNames.size() >= 2) {
                // Compute batch size from sparse features
                long batchSize = -1;
                if (sparseFeats != null) {
                    for (Object v : sparseFeats.values()) {
                        if (v instanceof Tensor) {
                            Tensor t = (Tensor) v;
                            if (!t.isNull() && t.dim() > 0) {
                                batchSize = t.size(0);
                            }
                            break;
                        }
                    }
                }
                // Compute embedding dims from structural children
                Map<String, Long> childEmbDims = new HashMap<>();
                long totalEmbDim = 0;
                for (String sname : structChildNames) {
                    long embDim = inferEmbeddingDim(sname);
                    childEmbDims.put(sname, embDim);
                    if (embDim > 0) totalEmbDim += embDim;
                }
                globalNodeCounter++;
                String catName = "cat_" + globalNodeCounter;
                graph.graphNodeNameToWithoutSuffix().put(catName, "cat");
                graph.graphNodeDisplayNames().put(catName, "cat(embed)");
                graph.nodeToModulePath().put(catName, "torch");
                GraphNode catNode = GraphNode.of(NodeType.OPERATION);
                graph.adjList().put(catName, catNode);
                recordParentBookkeeping(catName);
                graph.nodeToAncestors().put(catName, currentAncestors());
                // Wire edges from structural children to cat node with inferred dims
                for (String sname : structChildNames) {
                    if (sname == null || sname.equals(catName)) continue;
                    GraphNode sn = graph.adjList().get(sname);
                    if (sn == null) continue;
                    boolean hasEdge = false;
                    for (GraphEdge ge : sn.edges()) {
                        if (catName.equals(ge.target())) { hasEdge = true; break; }
                    }
                    if (!hasEdge) {
                        long embDim = childEmbDims.getOrDefault(sname, -1L);
                        String dims = batchSize > 0 && embDim > 0
                                ? "(" + batchSize + "," + embDim + ")" : "";
                        sn.addEdge(new GraphEdge(catName, dims, 0L, false));
                    }
                }
                // Store inferred output dims for downstream wiring
                String catOutDims = batchSize > 0 && totalEmbDim > 0
                        ? "(" + batchSize + "," + totalEmbDim + ")" : "";
                if (!catOutDims.isEmpty()) {
                    structuralCatDims.put(catName, catOutDims);
                }
                // Register the cat node as the implied output source for this
                // embedding layer so downstream modules can reference it via
                // the wire-implied post-processing pass.
                structuralEmbeddingCatNodes.put(moduleId(m), catName);
            }
            return null;
        } finally {
            endContainerFrame(frame);
        }
    }

    /** Infer embedding dim from a node's module_info (EmbeddingImpl weight shape[1]). */
    private long inferEmbeddingDim(String nodeName) {
        ModuleInfo info = graph.moduleInfo().get(nodeName);
        if (info == null || info.parameters() == null) return -1;
        ModuleInfo.ParamInfo weight = info.parameters().get("weight");
        if (weight == null) return -1;
        long[] shape = weight.shape();
        if (shape.length < 2) return -1;
        return shape[1];
    }

    private Tensor catAlongDim1(List<Tensor> parts, List<String> partNames) {
        Tensor[] arr = parts.toArray(new Tensor[0]);
        org.bytedeco.pytorch.TensorVector tv = new org.bytedeco.pytorch.TensorVector(arr);
        Tensor catOut = org.bytedeco.pytorch.global.torch.cat(tv, 1L);
        globalNodeCounter++;
        String catName = "cat_" + globalNodeCounter;
        graph.graphNodeNameToWithoutSuffix().put(catName, "cat");
        graph.graphNodeDisplayNames().put(catName, "cat");
        graph.nodeToModulePath().put(catName, "torch");
        GraphNode catNode = GraphNode.of(NodeType.OPERATION);
        graph.adjList().put(catName, catNode);
        recordParentBookkeeping(catName);
        graph.nodeToAncestors().put(catName, currentAncestors());
        for (int i = 0; i < parts.size(); i++) {
            Tensor t = parts.get(i);
            String src = i < partNames.size() ? partNames.get(i)
                    : tensorSource.get(TensorUtils.tensorKey(t));
            if (src == null) continue;
            GraphNode sn = graph.adjList().get(src);
            if (sn == null) continue;
            sn.addEdge(new GraphEdge(catName, TensorUtils.formatDims(t),
                    TensorUtils.tensorKey(t), false));
        }
        tensorSource.put(TensorUtils.tensorKey(catOut), catName);
        return catOut;
    }

    /**
     * Like {@link #runForward} but never throws — on failure emits structural
     * children when nothing was recorded, returns null.
     */
    private Tensor runForwardSafe(Module module, Object inputs, int stackDepth) {
        try {
            return runForward(module, inputs, stackDepth);
        } catch (Throwable e) {
            noteChildException(e);
            Module cm = ModuleDiscovery.concrete(module);
            if (!hasRecordedChild(cm)) {
                // If it's a sequential/MLP, expand structure fully
                if (ModuleDiscovery.isSequential(cm)
                        || ModuleDiscovery.canChainChildrenAsSequential(cm)
                        || ModuleChildren.hasChildren(cm)) {
                    emitStructuralChild(cm, moduleToAttrName.get(moduleId(cm)), stackDepth);
                } else {
                    // Try one more time as leaf to at least get a failed node with type info
                    structuralFallback = true;
                    try {
                        return traceLeafModule(cm, inputs, stackDepth);
                    } catch (Throwable e2) {
                        // preTraceOp already added failed node inside traceLeafModule
                        // if it got that far; otherwise structural
                        if (!hasRecordedChild(cm)) {
                            emitStructuralChild(cm, moduleToAttrName.get(moduleId(cm)), stackDepth);
                        }
                    } finally {
                        structuralFallback = false;
                    }
                }
            }
            return null;
        }
    }

    private void noteChildException(Throwable childEx) {
        if (childEx == null || graph.exception() != null) return;
        String mmsg = String.valueOf(childEx.getMessage());
        if (mmsg.contains("forward_tensor")
                || mmsg.contains("refusing to explode")
                || mmsg.contains("No matching forward")) {
            return;
        }
        graph.setException(childEx);
    }

    private static Tensor firstNonNull(Tensor a, Tensor b, Tensor c) {
        if (a != null && !a.isNull()) return a;
        if (b != null && !b.isNull()) return b;
        if (c != null && !c.isNull()) return c;
        return null;
    }

    private static Tensor firstNonNull(Tensor a, Tensor b) {
        return firstNonNull(a, b, null);
    }

    /** True when tensor is a floating activation (safe for Linear/Norm), not Long indices. */
    private static boolean isFloatTensor(Tensor t) {
        if (t == null || t.isNull()) return false;
        try {
            org.bytedeco.pytorch.global.torch.ScalarType st = t.scalar_type().intern();
            return st == org.bytedeco.pytorch.global.torch.ScalarType.Float
                    || st == org.bytedeco.pytorch.global.torch.ScalarType.Double
                    || st == org.bytedeco.pytorch.global.torch.ScalarType.Half
                    || st == org.bytedeco.pytorch.global.torch.ScalarType.BFloat16;
        } catch (Throwable e) {
            return false;
        }
    }

    /**
     * Build cat(taskEmb, pooled) for MetaHeac-style gates. Returns null when
     * ingredients are missing (caller falls back to structural emit).
     */
    private Object synthesizeGateInput(Map<String, Tensor> taggedOutputs, String pairIdx,
                                       Tensor sharedTensor, Tensor embedOut) {
        Tensor taskEmb = null;
        for (Map.Entry<String, Tensor> e : taggedOutputs.entrySet()) {
            String tk = e.getKey();
            if (tk.contains("taskembed") || tk.contains("task_emb") || tk.contains("taskemb")) {
                if (pairIdx == null || tk.endsWith("_" + pairIdx) || tk.endsWith(pairIdx)) {
                    taskEmb = e.getValue();
                    if (pairIdx != null) break;
                }
                if (taskEmb == null) taskEmb = e.getValue();
            }
        }
        Tensor embSrc = firstNonNull(sharedTensor, embedOut);
        Tensor pooled = null;
        if (embSrc != null && !embSrc.isNull()) {
            try {
                if (taskEmb != null && embSrc.dim() == 2 && taskEmb.dim() >= 1) {
                    long B = embSrc.size(0);
                    long flat = embSrc.size(1);
                    long D = taskEmb.dim() == 1 ? taskEmb.size(0) : taskEmb.size(taskEmb.dim() - 1);
                    if (D > 0 && flat % D == 0) {
                        long F = flat / D;
                        pooled = embSrc.view(B, F, D).mean(1);
                    }
                }
                if (pooled == null) pooled = embSrc;
            } catch (Throwable ignored) {
                pooled = embSrc;
            }
        }
        if (taskEmb != null && pooled != null) {
            try {
                org.bytedeco.pytorch.TensorVector tv =
                        new org.bytedeco.pytorch.TensorVector(taskEmb, pooled);
                Tensor gateIn = org.bytedeco.pytorch.global.torch.cat(tv, 1L);
                globalNodeCounter++;
                String catName = "cat_" + globalNodeCounter;
                graph.graphNodeNameToWithoutSuffix().put(catName, "cat");
                graph.graphNodeDisplayNames().put(catName, "cat(task,pooled)");
                graph.nodeToModulePath().put(catName, "torch");
                GraphNode catNode = GraphNode.of(NodeType.OPERATION);
                graph.adjList().put(catName, catNode);
                recordParentBookkeeping(catName);
                graph.nodeToAncestors().put(catName, currentAncestors());
                String tSrc = tensorSource.get(TensorUtils.tensorKey(taskEmb));
                String pSrc = tensorSource.get(TensorUtils.tensorKey(pooled));
                if (pSrc == null && embSrc != null) pSrc = tensorSource.get(TensorUtils.tensorKey(embSrc));
                if (tSrc != null) {
                    GraphNode sn = graph.adjList().get(tSrc);
                    if (sn != null) sn.addEdge(new GraphEdge(catName,
                            TensorUtils.formatDims(taskEmb), TensorUtils.tensorKey(taskEmb), false));
                }
                if (pSrc != null) {
                    GraphNode sn = graph.adjList().get(pSrc);
                    if (sn != null) sn.addEdge(new GraphEdge(catName,
                            TensorUtils.formatDims(pooled), TensorUtils.tensorKey(pooled), false));
                }
                tensorSource.put(TensorUtils.tensorKey(gateIn), catName);
                return gateIn;
            } catch (Throwable catEx) {
                return taskEmb;
            }
        }
        if (taskEmb != null) return taskEmb;
        return firstNonNull(sharedTensor, embedOut);
    }

    /**
     * After a structural emit, attach implied edges from the matching multi-arg
     * input(s) so the UI always shows flowing dashed arrows into the child.
     *
     * <p>Pos embeds: prefer an explicit positions slot (arg[1]) when present;
     * otherwise wire from the token-id input (models synthesize arange positions
     * from seq_len). Never invent a float activation source for EmbeddingImpl.
     */
    private void wireImpliedFromInputs(Module cm, String keyL, boolean isPosEmbed,
                                       boolean isTokenEmbed, boolean isNorm,
                                       List<Tensor> tensorArgs) {
        if (tensorArgs == null || tensorArgs.isEmpty()) return;
        // Resolve the node name(s) just recorded for this module
        List<String> names = moduleToNodeNames.get(moduleId(ModuleDiscovery.concrete(cm)));
        if (names == null || names.isEmpty()) return;
        String target = names.get(names.size() - 1);
        if (!graph.adjList().containsKey(target)) {
            // Container frame — prefer first leaf under it
            List<String> kids = graph.parentModuleToNodes().get(target);
            if (kids != null && !kids.isEmpty()) target = kids.get(0);
            else return;
        }
        Tensor srcT = null;
        if (isPosEmbed) {
            // Prefer dedicated positions tensor; else token ids (seq_len source).
            if (tensorArgs.size() > 1) {
                Tensor cand = tensorArgs.get(1);
                if (cand != null && !cand.isNull() && isIndexTensor(cand)) srcT = cand;
            }
            if (srcT == null) srcT = firstIndexTensor(tensorArgs);
            if (srcT == null) srcT = tensorArgs.get(0);
        } else if (isTokenEmbed) {
            srcT = firstIndexTensor(tensorArgs);
            if (srcT == null) srcT = tensorArgs.get(0);
        } else if (isNorm && tensorArgs.size() >= 1) {
            srcT = tensorArgs.get(0);
        } else if (tensorArgs.size() > 1 && (keyL.contains("time") || keyL.contains("pos"))) {
            srcT = tensorArgs.get(1);
        } else {
            srcT = tensorArgs.get(0);
        }
        if (srcT == null || srcT.isNull()) return;
        String inName = tensorSource.get(TensorUtils.tensorKey(srcT));
        if (inName == null) return;
        GraphNode inNode = graph.adjList().get(inName);
        if (inNode == null) return;
        // Avoid duplicate
        for (GraphEdge e : inNode.edges()) {
            if (target.equals(e.target())) return;
        }
        inNode.addEdge(new GraphEdge(target, TensorUtils.formatDims(srcT),
                TensorUtils.tensorKey(srcT), true));
    }

    /** Keys that name a positional Embedding table (not a token / codebook table). */
    private static boolean isPosEmbedKey(String keyL) {
        if (keyL == null || keyL.isEmpty()) return false;
        return keyL.contains("position") || keyL.contains("pos_emb")
                || keyL.contains("posemb") || keyL.contains("pos_embedding")
                || keyL.equals("pos") || keyL.equals("positionalembedding")
                || keyL.equals("positional_embedding")
                || (keyL.startsWith("pos") && keyL.contains("emb"));
    }

    /** True when tensor holds discrete indices suitable for EmbeddingImpl. */
    private static boolean isIndexTensor(Tensor t) {
        if (t == null || t.isNull()) return false;
        try {
            org.bytedeco.pytorch.global.torch.ScalarType st = t.scalar_type().intern();
            return st == org.bytedeco.pytorch.global.torch.ScalarType.Long
                    || st == org.bytedeco.pytorch.global.torch.ScalarType.Int
                    || st == org.bytedeco.pytorch.global.torch.ScalarType.Short
                    || st == org.bytedeco.pytorch.global.torch.ScalarType.Byte
                    || st == org.bytedeco.pytorch.global.torch.ScalarType.Char;
        } catch (Throwable e) {
            return false;
        }
    }

    private static Tensor firstIndexTensor(List<Tensor> tensorArgs) {
        if (tensorArgs == null) return null;
        for (Tensor t : tensorArgs) {
            if (t != null && !t.isNull() && isIndexTensor(t)) return t;
        }
        return null;
    }

    private static void markTensorArgUsed(List<Tensor> tensorArgs, boolean[] used, Tensor t) {
        if (tensorArgs == null || used == null || t == null) return;
        long key = TensorUtils.tensorKey(t);
        for (int i = 0; i < tensorArgs.size() && i < used.length; i++) {
            Tensor cand = tensorArgs.get(i);
            if (cand != null && !cand.isNull() && TensorUtils.tensorKey(cand) == key) {
                used[i] = true;
                return;
            }
        }
    }

    /**
     * Resolve input for a positional EmbeddingImpl.
     *
     * <ol>
     *   <li>Explicit positions slot (arg[1]) when multi-arg and index-typed (LLM4Rec).</li>
     *   <li>Otherwise synthesize {@code arange(T)} (or {@code arange(T)} expanded to
     *       {@code [B,T]}) from the token-id tensor, clamped to the table size —
     *       mirrors OneRec / HSTU / HLLM / OpenOneRec internal position construction.</li>
     * </ol>
     */
    private Tensor resolvePosEmbedInput(Module posEmb, List<Tensor> tensorArgs,
                                        Tensor rootTensor, boolean[] tensorArgUsed) {
        // 1) Explicit positions argument
        if (tensorArgs != null && tensorArgs.size() > 1) {
            Tensor cand = tensorArgs.get(1);
            if (cand != null && !cand.isNull() && isIndexTensor(cand)) {
                if (tensorArgUsed != null && tensorArgUsed.length > 1) tensorArgUsed[1] = true;
                return cand;
            }
            // Named multi-arg may put positions elsewhere — scan for a second index tensor
            Tensor firstIdx = null;
            for (int i = 0; i < tensorArgs.size(); i++) {
                Tensor t = tensorArgs.get(i);
                if (t == null || t.isNull() || !isIndexTensor(t)) continue;
                if (firstIdx == null) {
                    firstIdx = t;
                    continue;
                }
                // second index tensor → treat as positions
                if (tensorArgUsed != null && i < tensorArgUsed.length) tensorArgUsed[i] = true;
                return t;
            }
        }
        // 2) Synthesize from token sequence length
        Tensor tokens = firstIndexTensor(tensorArgs);
        if (tokens == null) tokens = (rootTensor != null && isIndexTensor(rootTensor)) ? rootTensor : null;
        if (tokens == null || tokens.isNull()) return null;
        if (tensorArgUsed != null) markTensorArgUsed(tensorArgs, tensorArgUsed, tokens);
        return synthesizePositionIndices(posEmb, tokens);
    }

    /**
     * Build Long position indices matching generative-model convention:
     * {@code arange(0, T)} broadcast / expanded to token batch shape when needed,
     * values clamped into {@code [0, num_embeddings)}.
     */
    private Tensor synthesizePositionIndices(Module posEmb, Tensor tokens) {
        try {
            long T;
            long B = 1L;
            int dim = (int) tokens.dim();
            if (dim >= 2) {
                B = tokens.size(0);
                T = tokens.size(1);
            } else if (dim == 1) {
                T = tokens.size(0);
            } else {
                T = 1L;
            }
            if (T <= 0) T = 1L;

            long vocab = embeddingNumEmbeddings(posEmb);
            if (vocab > 0 && T > vocab) T = vocab; // avoid OOB; still show the node

            org.bytedeco.pytorch.TensorOptions opts = new org.bytedeco.pytorch.TensorOptions()
                    .dtype(new org.bytedeco.pytorch.ScalarTypeOptional(
                            org.bytedeco.pytorch.global.torch.ScalarType.Long));
            try {
                opts = opts.device(new org.bytedeco.pytorch.DeviceOptional(tokens.device()));
            } catch (Throwable ignored) {}

            Tensor positions = org.bytedeco.pytorch.global.torch.arange(
                    new org.bytedeco.pytorch.Scalar(0),
                    new org.bytedeco.pytorch.Scalar((double) T),
                    new org.bytedeco.pytorch.Scalar(1),
                    opts);
            // OneRec / OpenOneRec expand to [B, T]; HSTU / HLLM use [T] + unsqueeze in model.
            // EmbeddingImpl accepts both; prefer [B,T] so output shape matches token emb.
            if (dim >= 2 && B > 0) {
                positions = positions.unsqueeze(0).expand(new long[]{B, T});
            }
            // Tag as synthetic so graph can still link from token input if needed
            try {
                long key = TensorUtils.tensorKey(positions);
                String tokSrc = tensorSource.get(TensorUtils.tensorKey(tokens));
                if (tokSrc != null && !tensorSource.containsKey(key)) {
                    // Keep provenance on the token input node (positions are derived)
                    tensorSource.put(key, tokSrc);
                    tensorImplied.put(key, Boolean.TRUE);
                }
            } catch (Throwable ignored) {}
            return positions;
        } catch (Throwable e) {
            return null;
        }
    }

    /** Best-effort {@code EmbeddingImpl.options().num_embeddings()} / weight rows. */
    private static long embeddingNumEmbeddings(Module m) {
        if (m == null) return -1L;
        try {
            Module cm = ModuleDiscovery.concrete(m);
            if (cm instanceof org.bytedeco.pytorch.nn.modules.EmbeddingImpl) {
                org.bytedeco.pytorch.nn.modules.EmbeddingImpl emb =
                        (org.bytedeco.pytorch.nn.modules.EmbeddingImpl) cm;
                try {
                    long n = emb.options().num_embeddings().get();
                    if (n > 0) return n;
                } catch (Throwable ignored) {}
                try {
                    Tensor w = emb.weight();
                    if (w != null && !w.isNull() && w.dim() >= 1) return w.size(0);
                } catch (Throwable ignored) {}
            }
        } catch (Throwable ignored) {}
        // Reflective fallback for wrapped / as() modules
        try {
            Object opts = m.getClass().getMethod("options").invoke(m);
            if (opts != null) {
                Object ne = opts.getClass().getMethod("num_embeddings").invoke(opts);
                if (ne instanceof org.bytedeco.javacpp.LongPointer) {
                    long n = ((org.bytedeco.javacpp.LongPointer) ne).get();
                    if (n > 0) return n;
                } else if (ne instanceof Number) {
                    long n = ((Number) ne).longValue();
                    if (n > 0) return n;
                }
            }
        } catch (Throwable ignored) {}
        try {
            Object w = m.getClass().getMethod("weight").invoke(m);
            if (w instanceof Tensor) {
                Tensor wt = (Tensor) w;
                if (wt != null && !wt.isNull() && wt.dim() >= 1) return wt.size(0);
            }
        } catch (Throwable ignored) {}
        return -1L;
    }

    private static boolean acceptsMapForward(Module m) {
        java.lang.reflect.Method fm = ModuleDiscovery.findForwardMethod(m, new HashMap<String, Tensor>());
        if (fm == null) return false;
        return Map.class.isAssignableFrom(fm.getParameterTypes()[0]);
    }

    /** {@code tower_12} → {@code "12"}; no trailing digits → null. */
    private static String trailingIndex(String key) {
        if (key == null || key.isEmpty()) return null;
        int i = key.length() - 1;
        while (i >= 0 && Character.isDigit(key.charAt(i))) i--;
        if (i == key.length() - 1) return null;
        // require separator before digits: tower_0 or tower0
        return key.substring(i + 1);
    }

    private static Tensor findTagged(Map<String, Tensor> tagged, String idx, String... prefixes) {
        for (String p : prefixes) {
            Tensor t = tagged.get(p + "_" + idx);
            if (t != null) return t;
            t = tagged.get(p + idx);
            if (t != null) return t;
        }
        // fuzzy: any key ending with _idx that contains a prefix
        for (Map.Entry<String, Tensor> e : tagged.entrySet()) {
            String k = e.getKey();
            if (k.endsWith("_" + idx) || k.endsWith(idx)) {
                for (String p : prefixes) {
                    if (k.contains(p)) return e.getValue();
                }
            }
        }
        return null;
    }

    private static Tensor findAnyTagged(Map<String, Tensor> tagged, String prefix) {
        for (Map.Entry<String, Tensor> e : tagged.entrySet()) {
            if (e.getKey().contains(prefix)) return e.getValue();
        }
        return null;
    }

    /** True if a node name represents an embedding source (EmbeddingImpl or cat(embed)). */
    private static boolean isEmbeddingSourceName(String name) {
        if (name == null) return false;
        String lower = name.toLowerCase();
        return lower.contains("embeddingimpl") || lower.contains("embedding")
                || lower.contains("cat(embed)") || lower.startsWith("cat_");
    }

    /** Register a child as a Module node without running forward (structure-only). */
    private void emitStructuralChild(Module child, String attrName, int stackDepth) {
        emitStructuralChildReturning(child, attrName, stackDepth);
    }

    /** @return last leaf node name emitted (for implied-edge chaining), or null */
    private String emitStructuralChildReturning(Module child, String attrName, int stackDepth) {
        Module m = ModuleDiscovery.concrete(child);

        // 1) Sequential-like containers: chain children in-order with implied edges
        if ((ModuleDiscovery.isSequential(m) || ModuleDiscovery.canChainChildrenAsSequential(m))
                && ModuleChildren.hasChildren(m)
                && !ModuleDiscovery.isTracedLeaf(m, stackDepth, options.forcedModuleTracingDepth())) {
            String frame = beginContainerFrame(m);
            String prevNode = null;
            String lastNode = null;
            try {
                for (ModuleChildren.NamedChild c : ModuleChildren.list(m)) {
                    String leaf = emitStructuralChildReturning(c.module, c.key, stackDepth + 1);
                    if (leaf != null) {
                        // Prefer to link from previous node to the first node emitted for this child
                        Long cid = moduleId(ModuleDiscovery.concrete(c.module));
                        java.util.List<String> childNames = moduleToNodeNames.get(cid);
                        String firstChild = (childNames != null && !childNames.isEmpty()) ? childNames.get(0) : leaf;
                        String lastChild = (childNames != null && !childNames.isEmpty()) ? childNames.get(childNames.size() - 1) : leaf;
                        if (prevNode != null && firstChild != null) linkImplied(prevNode, firstChild);
                        prevNode = lastChild;
                        lastNode = lastChild;
                    }
                }
            } finally {
                endContainerFrame(frame);
            }
            return lastNode;
        }

        // 2) Non-builtin composite with children: prefer chaining unless it's a
        //    list/dict/parameter collection which should remain parallel/transparent.
        if (ModuleChildren.hasChildren(m)
                && !ModuleDiscovery.isBuiltinLeaf(m)
                && !ModuleDiscovery.isTracedLeaf(m, stackDepth, options.forcedModuleTracingDepth())) {
            boolean forceParallel = ModuleDiscovery.isModuleListLike(m)
                    || ModuleDiscovery.isModuleDictLike(m)
                    || ModuleDiscovery.isParameterListLike(m)
                    || ModuleDiscovery.isParameterDictLike(m);

            String frame = beginContainerFrame(m);
            String lastNode = null;
            try {
                if (forceParallel) {
                    // Emit children structurally in parallel (no implied chaining)
                    for (ModuleChildren.NamedChild c : ModuleChildren.list(m)) {
                        String leaf = emitStructuralChildReturning(c.module, c.key, stackDepth + 1);
                        if (leaf != null) lastNode = leaf;
                    }
                } else {
                    // Prefer chaining for composites that are logically sequential
                    String prevNode = null;
                    for (ModuleChildren.NamedChild c : ModuleChildren.list(m)) {
                        String leaf = emitStructuralChildReturning(c.module, c.key, stackDepth + 1);
                        if (leaf != null) {
                            Long cid = moduleId(ModuleDiscovery.concrete(c.module));
                            java.util.List<String> childNames = moduleToNodeNames.get(cid);
                            String firstChild = (childNames != null && !childNames.isEmpty()) ? childNames.get(0) : leaf;
                            String lastChild = (childNames != null && !childNames.isEmpty()) ? childNames.get(childNames.size() - 1) : leaf;
                            if (prevNode != null && firstChild != null) linkImplied(prevNode, firstChild);
                            prevNode = lastChild;
                            lastNode = lastChild;
                        }
                    }
                }
            } finally {
                endContainerFrame(frame);
            }
            return lastNode;
        }

        // Leaf node
        return emitStructuralLeaf(m, attrName);
    }

    private String emitStructuralLeaf(Module m, String attrName) {
        String typeSimple = ModuleDiscovery.simpleTypeName(m);
        long id = moduleId(m);
        // If we've already emitted nodes for this module (via traceLeafModule or
        // a prior structural emit), reuse the last one to avoid duplicates.
        java.util.List<String> existing = moduleToNodeNames.get(id);
        if (existing != null && !existing.isEmpty()) {
            for (int i = existing.size() - 1; i >= 0; i--) {
                String existingName = existing.get(i);
                if (graph.adjList().containsKey(existingName)) {
                    // Reuse: update display metadata but keep the existing node
                    // (its edges and type are already correct).
                    String indexed = moduleToAttrName.get(id);
                    String useAttr = null;
                    if (indexed != null && !indexed.isEmpty()) useAttr = indexed;
                    else if (attrName != null && !attrName.isEmpty()) useAttr = attrName;
                    if (useAttr != null) graph.nodeToAttrName().put(existingName, useAttr);
                    graph.graphNodeDisplayNames().put(existingName, displayNameFor(m, typeSimple));
                    graph.moduleInfo().put(existingName, ModuleInfoCollector.collect(m));
                    // Clear failed flag: structural nodes are placeholders, not
                    // runtime-traced nodes. A prior traceLeafModule attempt may
                    // have set failed=true, but the structural emission treats
                    // this as a known-existing module, not a failed trace.
                    graph.adjList().get(existingName).setFailed(false);
                    return existingName;
                }
            }
        }

        String nodeName = nextModuleNodeName(typeSimple, m);
        graph.graphNodeNameToWithoutSuffix().put(nodeName, typeSimple);
        graph.nodeToModulePath().put(nodeName, packagePathFor(m));
        // Prefer pre-indexed dotted path (tower_0.sequential.1) over bare Sequential
        // child keys ("0"/"1"/"2") so the inspector stays readable after structural emit.
        String indexed = moduleToAttrName.get(moduleId(m));
        String useAttr = null;
        if (indexed != null && !indexed.isEmpty()) {
            useAttr = indexed;
        } else if (attrName != null && !attrName.isEmpty()) {
            useAttr = attrName;
        }
        if (useAttr != null) graph.nodeToAttrName().put(nodeName, useAttr);
        // displayNameFor also reads moduleToAttrName when showModuleAttrNames
        graph.graphNodeDisplayNames().put(nodeName, displayNameFor(m, typeSimple));
        graph.moduleInfo().put(nodeName, ModuleInfoCollector.collect(m));
        graph.adjList().put(nodeName, GraphNode.of(NodeType.MODULE));
        recordParentBookkeeping(nodeName);
        graph.nodeToAncestors().put(nodeName, currentAncestors());
        return nodeName;
    }

    /**
     * Recursively find the last leaf node (present in adjList) for a module by
     * walking its children. Container frames are registered in
     * {@code moduleToNodeNames} but are NOT added to {@code adjList}, so we must
     * descend into children to find the actual graph node representing the
     * module's output.
     */
    private String findLastLeafInGraph(Module m) {
        Module concrete = ModuleDiscovery.concrete(m);
        List<String> nodes = moduleToNodeNames.get(moduleId(concrete));
        if (nodes != null) {
            for (int i = nodes.size() - 1; i >= 0; i--) {
                String n = nodes.get(i);
                if (graph.adjList().containsKey(n)) return n;
            }
        }
        List<ModuleChildren.NamedChild> kids = ModuleChildren.list(concrete);
        for (int i = kids.size() - 1; i >= 0; i--) {
            String leaf = findLastLeafInGraph(kids.get(i).module);
            if (leaf != null) return leaf;
        }
        return null;
    }

    private void linkImplied(String from, String to) {
        if (from == null || to == null || from.equals(to)) return;
        GraphNode src = graph.adjList().get(from);
        GraphNode dst = graph.adjList().get(to);
        if (src == null || dst == null) return;
        // prefer linking to the first leaf under dst if dst is a container frame
        if (dst.nodeType() != NodeType.MODULE && graph.parentModuleToNodes().containsKey(to)) {
            java.util.List<String> kids = graph.parentModuleToNodes().get(to);
            if (kids != null && !kids.isEmpty()) {
                to = kids.get(0);
            }
        }
        for (GraphEdge e : src.edges()) {
            if (to.equals(e.target())) return;
        }
        src.addEdge(new GraphEdge(to, "", null, true));
    }

    /** Extract the numeric index from a role-tagged attr name (e.g.
     *  "gate_0.layer_1" with role "gate" → "0"; "tower_1.layer_0" → "1").
     *  Returns "0" when no index is found (single-instance fallback). */
    private static String extractRoleIndex(String attrL, String role) {
        if (attrL == null || attrL.isEmpty()) return "0";
        int idx = attrL.indexOf(role);
        if (idx < 0) return "0";
        int start = idx + role.length();
        if (start < attrL.length() && attrL.charAt(start) == '_') start++;
        int end = start;
        while (end < attrL.length() && Character.isDigit(attrL.charAt(end))) end++;
        if (end > start) return attrL.substring(start, end);
        return "0";
    }

    private Tensor traceLeafModule(Module module, Object inputs, int stackDepth) {
        Module m = ModuleDiscovery.concrete(module);
        String typeSimple = ModuleDiscovery.simpleTypeName(m);
        NodeType nodeType = NodeType.MODULE;

        // Reuse an existing structural node if present to avoid duplicates
        long id = moduleId(m);
        java.util.List<String> existing = moduleToNodeNames.get(id);
        String nodeName;
        if (existing != null && !existing.isEmpty()) {
            nodeName = existing.get(existing.size() - 1);
        } else {
            nodeName = nextModuleNodeName(typeSimple, m);
        }

        graph.graphNodeNameToWithoutSuffix().put(nodeName, typeSimple);
        graph.graphNodeDisplayNames().put(nodeName, displayNameFor(m, typeSimple));
        graph.nodeToModulePath().put(nodeName, packagePathFor(m));
        String attr = moduleToAttrName.get(moduleId(m));
        if (attr != null) {
            graph.nodeToAttrName().put(nodeName, attr);
        }
        graph.moduleInfo().put(nodeName, ModuleInfoCollector.collect(m));

        // pre_trace_op
        preTraceOp(nodeName, nodeType, inputs);

        moduleStack.add(nodeName);
        insideLeafModuleDepth++;
        Tensor output = null;
        try {
            output = callForward(m, inputs);
            // trace_op
            traceOp(nodeName, output, false);
        } catch (Throwable e) {
            // Clear failed flag: the module exists in the model structure,
            // we just couldn't trace its runtime behavior. It should appear
            // as a normal structural node, not a "FAILED" node.
            GraphNode failedNode = graph.adjList().get(nodeName);
            if (failedNode != null) {
                failedNode.setFailed(false);
            }
            throw e instanceof RuntimeException
                    ? (RuntimeException) e
                    : new RuntimeException(e);
        } finally {
            insideLeafModuleDepth = Math.max(0, insideLeafModuleDepth - 1);
            if (!moduleStack.isEmpty()
                    && moduleStack.get(moduleStack.size() - 1).equals(nodeName)) {
                moduleStack.remove(moduleStack.size() - 1);
            }
        }
        return output;
    }

    // =========================================================================
    // Container frame (ancestor only — no adj_list entry until nested transform)
    // =========================================================================

    private String beginContainerFrame(Module m) {
        String typeSimple = ModuleDiscovery.simpleTypeName(m);
        String nodeName = nextModuleNodeName(typeSimple, m);
        graph.graphNodeNameToWithoutSuffix().put(nodeName, typeSimple);
        graph.graphNodeDisplayNames().put(nodeName, displayNameFor(m, typeSimple));
        graph.nodeToModulePath().put(nodeName, packagePathFor(m));
        String attr = moduleToAttrName.get(moduleId(m));
        if (attr != null) {
            graph.nodeToAttrName().put(nodeName, attr);
        }
        // Containers are not leaves in adj_list during flat tracing; they appear
        // via node_to_ancestors and parent_module_to_nodes for collapsible UI
        // once children call preTraceOp. Do NOT recordParentBookkeeping here —
        // that would self-register the container under itself.
        graph.moduleInfo().put(nodeName, ModuleInfoCollector.collect(m));
        moduleStack.add(nodeName);
        return nodeName;
    }

    private void endContainerFrame(String nodeName) {
        if (!moduleStack.isEmpty()
                && moduleStack.get(moduleStack.size() - 1).equals(nodeName)) {
            moduleStack.remove(moduleStack.size() - 1);
        }
    }

    // =========================================================================
    // pre_trace_op / trace_op (torchvista engine equivalents)
    // =========================================================================

    private void preTraceOp(String opName, NodeType nodeType, Object inputs) {
        List<Tensor> inputTensors = TensorUtils.extractTensors(inputs);

        GraphNode node;
        if (graph.adjList().containsKey(opName)) {
            // Reuse existing structural node — do NOT override failed status.
            // A node that was already successfully traced (failed=false) must
            // stay successful; only fresh nodes start as failed until traceOp
            // proves them successful.
            node = graph.adjList().get(opName);
        } else {
            node = GraphNode.failed(nodeType); // failed until trace_op succeeds
            graph.adjList().put(opName, node);
        }

        for (Tensor inp : inputTensors) {
            long key = TensorUtils.tensorKey(inp);
            String sourceName = tensorSource.get(key);
            if (sourceName != null) {
                Long edgeDataId = key;
                String edgeKey = sourceName + "->" + opName + "#" + edgeDataId;
                if (seenEdges.contains(edgeKey)) continue;
                seenEdges.add(edgeKey);

                String dims = TensorUtils.formatDims(inp);
                boolean implied = Boolean.TRUE.equals(tensorImplied.get(key));
                GraphNode src = graph.adjList().get(sourceName);
                if (src != null) {
                    src.addEdge(new GraphEdge(opName, dims, edgeDataId, implied));
                }
            } else if (options.showNonGradientNodes() && !structuralFallback) {
                // Untagged tensor → Constant node (torchvista non-gradient path).
                // Skip tensors that are likely parameters (requires_grad) or
                // intermediate computation results (not real constants).
                // Only create a constant node for truly standalone tensor literals.
                boolean isParam = false;
                try {
                    isParam = inp.requires_grad();
                } catch (Throwable ignored) {}
                if (!isParam) {
                    globalNodeCounter++;
                    String tensorNodeName = "tensor_" + globalNodeCounter;
                    GraphNode c = GraphNode.of(NodeType.CONSTANT);
                    graph.adjList().put(tensorNodeName, c);
                    String dims = TensorUtils.formatDims(inp);
                    c.addEdge(new GraphEdge(opName, dims, key, false));
                    graph.nodeToAncestors().put(tensorNodeName, currentAncestors());
                    constantNodeNames.add(tensorNodeName);
                    graph.graphNodeDisplayNames().put(tensorNodeName, "tensor");
                    graph.graphNodeNameToWithoutSuffix().put(tensorNodeName, "tensor");
                    tensorSource.put(key, tensorNodeName);
                }
            }
        }

        // Record func/module call args for the info popup
        recordOpParameters(opName, inputs);

        currentOp = opName;
        recordParentBookkeeping(opName);
        graph.nodeToAncestors().put(opName, currentAncestors());
    }

    private void traceOp(String opName, Object output, boolean impliedEdge) {
        GraphNode node = graph.adjList().get(opName);
        if (node == null) return;

        List<Tensor> outputTensors = TensorUtils.extractTensors(output);
        if (outputTensors.isEmpty()) {
            nodesToDelete.add(opName);
            return;
        }
        node.setFailed(false);
        lastSuccessfulOp = opName;
        currentOp = null;

        for (Tensor t : outputTensors) {
            long key = TensorUtils.tensorKey(t);
            // Prefer earlier source mapping — do not overwrite existing mapping which
            // may represent the true origin of this tensor (prevents later ops
            // from stealing index/embedding sources).
            if (!tensorSource.containsKey(key)) {
                tensorSource.put(key, opName);
            }
            if (impliedEdge) {
                tensorImplied.put(key, true);
            } else {
                tensorImplied.remove(key);
            }
        }
    }

    /**
     * Record a free functional / tensor op as an {@link NodeType#OPERATION} node.
     * Called by {@link VistaOps} while this engine is bound on the thread.
     *
     * <p>Free ops inside an opaque built-in leaf ({@code insideLeafModuleDepth > 0})
     * are suppressed — the leaf already represents that work. Free ops inside an
     * open custom module frame are recorded, matching torchvista.
     */
    public Tensor traceFreeOp(String opName, String namespace, Object[] inputs,
                              VistaOps.TensorSupplier body) {
        if (insideLeafModuleDepth > 0) {
            return body.get();
        }
        if (opName == null || opName.isEmpty()) opName = "op";
        if (namespace == null) namespace = "torch";

        // Skip zero-tensor ops (pure scalar bookkeeping) — same as torchvista
        List<Tensor> inTensors = TensorUtils.extractTensors(inputs);
        if (inTensors.isEmpty()) {
            return body.get();
        }

        globalNodeCounter++;
        String nodeName = opName + "_" + globalNodeCounter;
        graph.graphNodeNameToWithoutSuffix().put(nodeName, opName);
        graph.graphNodeDisplayNames().put(nodeName, opName);
        graph.nodeToModulePath().put(nodeName, namespace);

        preTraceOp(nodeName, NodeType.OPERATION, inputs == null ? new Object[0] : inputs);
        Tensor out;
        try {
            out = body.get();
            traceOp(nodeName, out, false);
        } catch (RuntimeException e) {
            // leave failed=true
            throw e;
        } catch (Exception e) {
            throw new RuntimeException("free op " + opName + " failed: " + e.getMessage(), e);
        }
        return out;
    }

    /**
     * Run a nested module call from inside an open custom {@code Module.forward}
     * (via {@link VistaOps#module}). Re-enters {@link #runForward} at
     * {@code moduleStack.size()} depth so the child becomes a proper graph node.
     */
    public Tensor traceNestedModule(Module child, Object inputs) {
        if (insideLeafModuleDepth > 0) {
            // Shouldn't normally happen — nested module from inside builtin leaf
            return callForward(child, inputs);
        }
        int depth = moduleStack.size();
        return runForward(child, inputs, depth);
    }

    private void recordOpParameters(String opName, Object inputs) {
        Map<String, Object> info = new LinkedHashMap<>();
        if (inputs instanceof Object[]) {
            info.put("positional_args", TensorUtils.formatArgs((Object[]) inputs));
            info.put("keyword_args", Collections.emptyMap());
        } else if (inputs instanceof List) {
            List<?> list = (List<?>) inputs;
            info.put("positional_args", TensorUtils.formatArgs(list.toArray()));
            info.put("keyword_args", Collections.emptyMap());
        } else if (inputs instanceof Map) {
            info.put("positional_args", Collections.emptyList());
            Map<String, Object> kwargs = new LinkedHashMap<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) inputs).entrySet()) {
                kwargs.put(String.valueOf(e.getKey()), TensorUtils.formatArg(e.getValue()));
            }
            info.put("keyword_args", kwargs);
        } else if (inputs != null) {
            info.put("positional_args", TensorUtils.formatArgs(inputs));
            info.put("keyword_args", Collections.emptyMap());
        } else {
            info.put("positional_args", Collections.emptyList());
            info.put("keyword_args", Collections.emptyMap());
        }
        graph.funcInfo().put(opName, info);
    }

    private void recordParentBookkeeping(String opName) {
        int depth = 1;
        // module_stack is root→leaf; torchvista iterates module_stack[::-1]
        for (int i = moduleStack.size() - 1; i >= 0; i--) {
            String parent = moduleStack.get(i);
            graph.parentModuleToNodes()
                    .computeIfAbsent(parent, k -> new ArrayList<>())
                    .add(opName);
            Integer prev = graph.parentModuleToDepth().get(parent);
            graph.parentModuleToDepth().put(parent, Math.max(depth, prev == null ? 0 : prev));
            depth++;
        }
    }

    private List<String> currentAncestors() {
        // torchvista: module_stack[::-1] — immediate parent first
        List<String> a = new ArrayList<>(moduleStack.size());
        for (int i = moduleStack.size() - 1; i >= 0; i--) {
            a.add(moduleStack.get(i));
        }
        return a;
    }

    // =========================================================================
    // Inputs / outputs
    // =========================================================================

    private List<Tensor> tagInputs(Object inputs) {
        List<Tensor> tensors = TensorUtils.extractTensors(inputs);

        // Multi-arg Object[]{Map features, Tensor taskIdx, …} — tag the Map by
        // feature keys and remaining bare tensors as input_task / input_N.
        if (inputs instanceof Object[] && !(inputs instanceof Tensor[])) {
            Object[] arr = (Object[]) inputs;
            int bare = 0;
            for (Object o : arr) {
                if (o instanceof Map) {
                    for (Map.Entry<?, ?> e : ((Map<?, ?>) o).entrySet()) {
                        for (Tensor t : TensorUtils.extractTensors(e.getValue())) {
                            String path = String.valueOf(e.getKey());
                            registerInputNode("input_" + path, path, t);
                        }
                    }
                } else {
                    for (Tensor t : TensorUtils.extractTensors(o)) {
                        long key = TensorUtils.tensorKey(t);
                        if (tensorSource.containsKey(key)) continue;
                        String name = bare == 0 ? "input_task" : ("input_arg_" + bare);
                        registerInputNode(name, name, t);
                        bare++;
                    }
                }
            }
            for (Tensor t : tensors) {
                long key = TensorUtils.tensorKey(t);
                if (!tensorSource.containsKey(key)) {
                    registerInputNode("input_" + (globalNodeCounter++), "input", t);
                }
            }
            return tensors;
        }

        if (inputs instanceof Map) {
            for (Map.Entry<?, ?> e : ((Map<?, ?>) inputs).entrySet()) {
                List<Tensor> ts = TensorUtils.extractTensors(e.getValue());
                for (Tensor t : ts) {
                    String path = String.valueOf(e.getKey());
                    registerInputNode("input_" + path, path, t);
                }
            }
            for (Tensor t : tensors) {
                long key = TensorUtils.tensorKey(t);
                if (!tensorSource.containsKey(key)) {
                    registerInputNode("input_" + (globalNodeCounter++), "input", t);
                }
            }
        } else if (inputs instanceof Tensor[] || (inputs instanceof List && !tensors.isEmpty()
                && tensors.size() == ((List<?>) inputs).size())) {
            // Multi-tensor forward(tokens, positions, …) — give each slot a stable name
            // so expandCustomFromChildren can route input_0→tokenEmb, input_1→posEmb
            // and BOTH inputs always get flowing dashed edges.
            String[] slotNames = multiTensorSlotNames(tensors.size());
            for (int i = 0; i < tensors.size(); i++) {
                String slot = slotNames[i];
                registerInputNode("input_" + i, slot, tensors.get(i));
            }
        } else {
            for (int i = 0; i < tensors.size(); i++) {
                registerInputNode("input_" + i, "input_" + i, tensors.get(i));
            }
        }
        return tensors;
    }

    /** Friendly slot labels for multi-tensor model inputs (LLM4Rec, HLLM, …). */
    private static String[] multiTensorSlotNames(int n) {
        String[] names = new String[n];
        String[] defaults = {"tokens", "positions", "time_diffs", "mask", "labels", "arg5", "arg6"};
        for (int i = 0; i < n; i++) {
            names[i] = i < defaults.length ? defaults[i] : ("arg_" + i);
        }
        // 2-arg common patterns
        if (n == 2) {
            names[0] = "tokens";
            names[1] = "positions";
        }
        return names;
    }

    private void registerInputNode(String inputName, String display, Tensor t) {
        long key = TensorUtils.tensorKey(t);
        tensorSource.put(key, inputName);
        graph.graphNodeNameToWithoutSuffix().put(inputName, display);
        graph.graphNodeDisplayNames().put(inputName, display);
        graph.adjList().put(inputName, GraphNode.of(NodeType.INPUT));
        graph.nodeToAncestors().put(inputName, new ArrayList<>());
        // Annotate with feature catalog (sparse/dense/sequence) + live tensor shape
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("kind", "input");
        meta.put("name", display);
        String shape = TensorUtils.formatDims(t);
        meta.put("shape", shape);
        meta.put("dtype", TensorUtils.safeDtype(t));
        // Match catalog by bare feature name
        Map<String, Object> cat = lookupFeatureCatalog(display);
        if (cat != null) {
            meta.putAll(cat);
            // Prefer catalog name if richer
            if (cat.get("name") != null) {
                graph.graphNodeDisplayNames().put(inputName, String.valueOf(cat.get("name")));
            }
            // Subtitle for renderer: "SPARSE · vocab=500 · emb=8"
            String ft = String.valueOf(cat.getOrDefault("feature_type", "feature"));
            graph.graphNodeNameToWithoutSuffix().put(inputName, ft.toUpperCase() + " " + display);
        } else {
            // Heuristic when no Feature object is available
            String guess = guessFeatureType(display, t);
            meta.put("feature_type", guess);
            graph.graphNodeNameToWithoutSuffix().put(inputName, guess.toUpperCase() + " " + display);
        }
        graph.nodeMeta().put(inputName, meta);
    }

    private void tagOutputs(Object output) {
        List<Tensor> outputTensors = TensorUtils.extractTensors(output);

        // Named multi-task Map outputs → one Output node per task/label, preserving order
        List<String> namedKeys = null;
        List<Tensor> orderedTensors = outputTensors;
        if (output instanceof Map) {
            namedKeys = new ArrayList<>();
            orderedTensors = new ArrayList<>();
            for (Map.Entry<?, ?> e : ((Map<?, ?>) output).entrySet()) {
                namedKeys.add(String.valueOf(e.getKey()));
                List<Tensor> ts = TensorUtils.extractTensors(e.getValue());
                orderedTensors.add(ts.isEmpty() ? null : ts.get(0));
            }
        }

        Map<Long, String> seen = new HashMap<>();
        List<String> createdOuts = new ArrayList<>();
        for (int i = 0; i < orderedTensors.size(); i++) {
            Tensor t = orderedTensors.get(i);
            String labelName = null;
            if (namedKeys != null && i < namedKeys.size()) {
                labelName = namedKeys.get(i);
            } else if (taskLabelNames.size() > i) {
                labelName = taskLabelNames.get(i);
            }

            String outputNodeName;
            if (labelName != null) {
                outputNodeName = orderedTensors.size() <= 1 && namedKeys == null
                        ? "output" : ("output_" + labelName);
            } else {
                outputNodeName = orderedTensors.size() <= 1 ? "output" : ("output_" + i);
            }
            if (graph.adjList().containsKey(outputNodeName)) {
                outputNodeName = outputNodeName + "_" + globalNodeCounter++;
            }

            // Dedup by tensor identity when available
            if (t != null && !t.isNull()) {
                long tid = TensorUtils.tensorKey(t);
                String existing = seen.get(tid);
                if (existing != null) {
                    outputNodeName = existing;
                } else {
                    seen.put(tid, outputNodeName);
                }
            }

            if (!graph.adjList().containsKey(outputNodeName)) {
                String display = labelName != null ? labelName : outputNodeName;
                graph.graphNodeNameToWithoutSuffix().put(outputNodeName, "LABEL " + display);
                graph.graphNodeDisplayNames().put(outputNodeName, display);
                graph.adjList().put(outputNodeName, GraphNode.of(NodeType.OUTPUT));
                outputNodeSet.add(outputNodeName);
                createdOuts.add(outputNodeName);

                Map<String, Object> meta = new LinkedHashMap<>();
                meta.put("kind", "output");
                meta.put("name", display);
                meta.put("label", display);
                meta.put("feature_type", "label");
                if (labelName != null) meta.put("task", labelName);
                if (t != null && !t.isNull()) {
                    meta.put("shape", TensorUtils.formatDims(t));
                    meta.put("dtype", TensorUtils.safeDtype(t));
                }
                graph.nodeMeta().put(outputNodeName, meta);
            } else {
                createdOuts.add(outputNodeName);
            }

            if (t != null && !t.isNull()) {
                long tid = TensorUtils.tensorKey(t);
                String source = tensorSource.get(tid);
                if (source != null && !isEmbeddingSourceName(source)) {
                    GraphNode src = graph.adjList().get(source);
                    if (src != null) {
                        String dims = TensorUtils.formatDims(t);
                        boolean implied = Boolean.TRUE.equals(tensorImplied.get(tid));
                        src.addEdge(new GraphEdge(outputNodeName, dims, tid, implied));
                    }
                }
            }
        }

        // Multi-task / fan-out: connect every Module/Operation sink that still has
        // no outgoing edge to the output node(s).
        if (outputNodeSet.isEmpty()) {
            String outputNodeName = "output";
            graph.graphNodeNameToWithoutSuffix().put(outputNodeName, "LABEL output");
            graph.graphNodeDisplayNames().put(outputNodeName, "output");
            graph.adjList().put(outputNodeName, GraphNode.of(NodeType.OUTPUT));
            outputNodeSet.add(outputNodeName);
            Map<String, Object> meta = new LinkedHashMap<>();
            meta.put("kind", "output");
            meta.put("name", "output");
            meta.put("label", "output");
            meta.put("feature_type", "label");
            graph.nodeMeta().put(outputNodeName, meta);
        }

        List<String> outs = new ArrayList<>(outputNodeSet);
        Map<String, String> inDimHint = new HashMap<>();
        for (GraphNode n : graph.adjList().values()) {
            for (GraphEdge e : n.edges()) {
                if (e.dims() != null && !e.dims().isEmpty()) {
                    inDimHint.putIfAbsent(e.target(), e.dims());
                }
            }
        }
        // Compute the downstream target of the first EmbeddingImpl that has
        // edges (from real tracing). This is used to wire orphaned EmbeddingImpl
        // siblings — they model the concatenation that happens inside
        // feature-embedding modules whose forward expects a Map input (which
        // we cannot trace with Tensor[]).
        String embedDownstream = null;
        for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
            String name = e.getKey();
            if (!name.contains("EmbeddingImpl")) continue;
            GraphNode node = e.getValue();
            if (!node.edges().isEmpty()) {
                for (GraphEdge edge : node.edges()) {
                    String tgt = edge.target();
                    if (!tgt.equals("output") && !tgt.contains("output")
                            && graph.adjList().containsKey(tgt)) {
                        if (embedDownstream == null) embedDownstream = tgt;
                    }
                }
            }
        }

        // Infer missing dims on structural edges: when runtime forward
        // failed (e.g. MPS device), edges created structurally have empty
        // dims. Try to infer them from module_info (LinearImpl weight
        // shape[0] = out_features) and the batch size from input edges.
        {
            long inferredBatch = -1;
            for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
                if (!e.getKey().startsWith("input_")) continue;
                for (GraphEdge ge : e.getValue().edges()) {
                    if (ge.dims() != null && !ge.dims().isEmpty()) {
                        String d = ge.dims().replaceAll("[()]", "");
                        String[] parts = d.split(",");
                        if (parts.length > 0) {
                            try { inferredBatch = Long.parseLong(parts[0].trim()); } catch (NumberFormatException ex) {}
                        }
                    }
                    if (inferredBatch > 0) break;
                }
                if (inferredBatch > 0) break;
            }
            // Pass 1: infer from LinearImpl weight shape
            if (inferredBatch > 0) {
                for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
                    String srcName = e.getKey();
                    GraphNode srcNode = e.getValue();
                    List<GraphEdge> edges = srcNode.edges();
                    for (int i = 0; i < edges.size(); i++) {
                        GraphEdge ge = edges.get(i);
                        if (ge.dims() != null && !ge.dims().isEmpty()) continue;
                        ModuleInfo info = graph.moduleInfo().get(srcName);
                        if (info == null || info.parameters() == null) continue;
                        ModuleInfo.ParamInfo weight = info.parameters().get("weight");
                        if (weight == null) continue;
                        long[] shape = weight.shape();
                        if (shape.length < 2) continue;
                        long outFeatures = shape[0];
                        String inferred = "(" + inferredBatch + "," + outFeatures + ")";
                        edges.set(i, new GraphEdge(ge.target(), inferred, ge.edgeDataId(), ge.implied()));
                    }
                }
            }
            // Pass 2: propagate dims through pass-through ops (ReLU, Dropout, etc.)
            // Iterate until no changes (handles chains like Linear→ReLU→Dropout→Linear)
            for (int iter = 0; iter < 10; iter++) {
                boolean changed = false;
                // Build incoming dims map
                Map<String, String> incomingDims = new HashMap<>();
                for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
                    for (GraphEdge ge : e.getValue().edges()) {
                        if (ge.dims() != null && !ge.dims().isEmpty()) {
                            incomingDims.putIfAbsent(ge.target(), ge.dims());
                        }
                    }
                }
                // Propagate to empty edges
                for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
                    String srcName = e.getKey();
                    GraphNode srcNode = e.getValue();
                    List<GraphEdge> edges = srcNode.edges();
                    String srcIncoming = incomingDims.get(srcName);
                    if (srcIncoming == null) continue;
                    for (int i = 0; i < edges.size(); i++) {
                        GraphEdge ge = edges.get(i);
                        if (ge.dims() != null && !ge.dims().isEmpty()) continue;
                        edges.set(i, new GraphEdge(ge.target(), srcIncoming, ge.edgeDataId(), ge.implied()));
                        changed = true;
                    }
                }
                if (!changed) break;
            }
        }

        // Structural cat(embed) global fan-out: wire cat(embed) nodes to
        // downstream modules (linear/fm/mlp/tower/expert) that have no
        // incoming edges. This handles the case where runtime forward failed
        // (e.g. MPS device) and the embedding concatenation node was created
        // structurally but downstream modules inside nested containers (MLP)
        // were not wired to it.
        for (Map.Entry<Long, String> entry : structuralEmbeddingCatNodes.entrySet()) {
            String catName = entry.getValue();
            GraphNode catGN = graph.adjList().get(catName);
            if (catGN == null) continue;
            // Find the parent module of the cat node
            String catParent = null;
            for (Map.Entry<String, List<String>> pe : graph.parentModuleToNodes().entrySet()) {
                if (pe.getValue().contains(catName)) {
                    catParent = pe.getKey();
                    break;
                }
            }
            // Find sibling nodes (same parent) with no incoming edges
            if (catParent != null) {
                List<String> siblings = graph.parentModuleToNodes().get(catParent);
                if (siblings != null) {
                    for (String sib : siblings) {
                        if (sib == null || sib.equals(catName)) continue;
                        GraphNode sn = graph.adjList().get(sib);
                        if (sn == null) continue;
                        // Skip embedding nodes and cat nodes
                        String sibDisp = graph.graphNodeDisplayNames().get(sib);
                        if (sibDisp != null && (sibDisp.contains("cat") || sibDisp.contains("EmbeddingImpl")
                                || sibDisp.contains("embedding"))) continue;
                        // Check if sib already has incoming edges
                        boolean hasIncoming = false;
                        for (GraphNode gn : graph.adjList().values()) {
                            for (GraphEdge ge : gn.edges()) {
                                if (sib.equals(ge.target())) { hasIncoming = true; break; }
                            }
                            if (hasIncoming) break;
                        }
                        if (hasIncoming) continue;
                        // Wire cat → sib
                        boolean hasEdge = false;
                        for (GraphEdge ge : catGN.edges()) {
                            if (sib.equals(ge.target())) { hasEdge = true; break; }
                        }
                        if (!hasEdge) {
                            String dims = structuralCatDims.getOrDefault(catName, "");
                            catGN.addEdge(new GraphEdge(sib, dims, 0L, false));
                        }
                    }
                }
            }
        }

        // Also wire cat(embed) to first layers of nested containers (MLP etc.)
        // that are NOT siblings but descendants of the same model.
        for (Map.Entry<Long, String> entry : structuralEmbeddingCatNodes.entrySet()) {
            String catName = entry.getValue();
            GraphNode catGN = graph.adjList().get(catName);
            if (catGN == null) continue;
            String catDims = structuralCatDims.getOrDefault(catName, "");
            // Find all nodes with no incoming edges that are Linear/FM/MLP-like
            for (Map.Entry<String, GraphNode> ne : graph.adjList().entrySet()) {
                String name = ne.getKey();
                if (name.equals(catName)) continue;
                GraphNode node = ne.getValue();
                if (node.nodeType() == NodeType.INPUT || node.nodeType() == NodeType.OUTPUT) continue;
                String disp = graph.graphNodeDisplayNames().get(name);
                if (disp == null) continue;
                // Only wire to first-layer linear/fm/mlp nodes
                boolean isFirstLayer = disp.contains("layer_0") || disp.equals("linear")
                        || disp.equals("fm") || disp.contains(".layer_0");
                if (!isFirstLayer) continue;
                // Skip if already has incoming edges
                boolean hasIncoming = false;
                for (GraphNode gn : graph.adjList().values()) {
                    for (GraphEdge ge : gn.edges()) {
                        if (name.equals(ge.target())) { hasIncoming = true; break; }
                    }
                    if (hasIncoming) break;
                }
                if (hasIncoming) continue;
                // Wire cat → node
                boolean hasEdge = false;
                for (GraphEdge ge : catGN.edges()) {
                    if (name.equals(ge.target())) { hasEdge = true; break; }
                }
                if (!hasEdge) {
                    catGN.addEdge(new GraphEdge(name, catDims, 0L, false));
                }
            }
        }

        // Collect sinks (modules with no outgoing edges to existing nodes)
        // Also classify sinks by their attr-name role for combining-op detection.
        List<String> sinks = new ArrayList<>();
        List<String> expertSinks = new ArrayList<>();
        List<String> towerSinks = new ArrayList<>();
        List<String> gateSinks = new ArrayList<>();
        List<String> predictLayerSinks = new ArrayList<>();
        List<String> aitSinks = new ArrayList<>();
        for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
            String name = e.getKey();
            GraphNode node = e.getValue();
            if (node.nodeType() == NodeType.INPUT
                    || node.nodeType() == NodeType.OUTPUT
                    || node.nodeType() == NodeType.CONSTANT
                    || node.nodeType() == NodeType.PARAMETER) {
                continue;
            }
            // Only count edges that point to nodes actually in the graph —
            // edges to removed container frames don't count.
            boolean hasRealEdge = false;
            for (GraphEdge edge : node.edges()) {
                if (graph.adjList().containsKey(edge.target())) {
                    hasRealEdge = true;
                    break;
                }
            }
            if (hasRealEdge) continue;
            // When we have a real downstream target for EmbeddingImpl nodes,
            // skip them from sink-fan-out — they are feature extractors whose
            // outputs are consumed downstream, not final outputs. But when no
            // downstream target was found (e.g. EmbeddingImpl IS the last
            // module), let them participate normally.
            if (embedDownstream != null && name.contains("EmbeddingImpl")) {
                continue;
            }
            // Classify by attr name for combining-op detection
            String attrName = graph.nodeToAttrName().get(name);
            String attrL = attrName == null ? "" : attrName.toLowerCase();
            if (attrL.contains("expert")) {
                expertSinks.add(name);
                continue; // experts feed towers/gates, not output
            }
            if (attrL.contains("tower")) {
                towerSinks.add(name);
            }
            if (attrL.contains("gate")) {
                gateSinks.add(name);
                continue; // gates classified separately — added to sinks
                          // only when no towers exist (see below)
            }
            // AITM: collect bottom / infoGate / ait sinks separately so they
            // can be wired to their real downstream consumers (ait → tower)
            // instead of being dropped as "intermediate" with no edges.
            if (attrL.contains("ait")) {
                aitSinks.add(name);
                continue;
            }
            if (attrL.contains("infogate") || attrL.contains("bottom")) {
                continue; // wired via AITM section below
            }
            if (attrL.contains("predict") || attrL.startsWith("pred")
                    || attrL.contains("output_layer")
                    || (attrL.contains("head") && !attrL.contains("gate"))) {
                predictLayerSinks.add(name);
            }
            // Note: embedding-attr nodes are NOT excluded here. Codebook
            // embeddings (VectorQuantizer, RQVAE) have no downstream consumer
            // and ARE the final output. Feature embeddings with downstream
            // targets are already skipped by the embedDownstream check above.
            sinks.add(name);
        }

        // MoE / MMOE / OMoE: when experts and towers coexist, insert a
        // combine operation that represents the gate-weighted sum of expert
        // outputs. The gate produces per-expert weights, experts produce
        // outputs, and the combine op weights and sums them before feeding
        // the result into tower INPUT nodes (heads), not tower tail nodes.

        // Find tower head nodes: tower-attributed nodes that have an
        // incoming edge from a non-tower node (the tower entry point).
        Set<String> towerHeads = new LinkedHashSet<>();
        for (Map.Entry<String, GraphNode> te : graph.adjList().entrySet()) {
            String tName = te.getKey();
            String tAttr = graph.nodeToAttrName().get(tName);
            String tL = tAttr == null ? "" : tAttr.toLowerCase();
            if (!tL.contains("tower")) continue;
            for (Map.Entry<String, GraphNode> se : graph.adjList().entrySet()) {
                if (se.getKey().equals(tName)) continue;
                String sAttr = graph.nodeToAttrName().get(se.getKey());
                String sL = sAttr == null ? "" : sAttr.toLowerCase();
                if (sL.contains("tower")) continue; // skip tower-internal edges
                for (GraphEdge edge : se.getValue().edges()) {
                    if (tName.equals(edge.target())) {
                        towerHeads.add(tName);
                        break;
                    }
                }
                if (towerHeads.contains(tName)) break;
            }
        }

        if (!expertSinks.isEmpty() && !towerHeads.isEmpty()) {
            // Group gate sinks by index (gate_0, gate_1, …) for per-gate
            // combine nodes (MMOE). OMoE has a single gate → one combine.
            Map<String, String> gateByIndex = new LinkedHashMap<>();
            for (String gate : gateSinks) {
                String attr = graph.nodeToAttrName().get(gate);
                String aL = attr == null ? "" : attr.toLowerCase();
                String idx = extractRoleIndex(aL, "gate");
                gateByIndex.put(idx, gate);
            }
            // Group tower heads by index (tower_0, tower_1, …)
            Map<String, String> towerHeadByIndex = new LinkedHashMap<>();
            for (String head : towerHeads) {
                String attr = graph.nodeToAttrName().get(head);
                String aL = attr == null ? "" : attr.toLowerCase();
                String idx = extractRoleIndex(aL, "tower");
                towerHeadByIndex.put(idx, head);
            }

            boolean singleGate = gateByIndex.size() <= 1;
            if (singleGate) {
                // OMoE: one combine for all experts + the single gate → all towers
                globalNodeCounter++;
                String moeCombineName = "moe_combine_" + globalNodeCounter;
                graph.graphNodeNameToWithoutSuffix().put(moeCombineName, "moe_combine");
                graph.graphNodeDisplayNames().put(moeCombineName, "moe_combine");
                graph.nodeToModulePath().put(moeCombineName, "torch");
                GraphNode moeCombineNode = GraphNode.of(NodeType.OPERATION);
                graph.adjList().put(moeCombineName, moeCombineNode);
                for (String gate : gateSinks) {
                    GraphNode gn = graph.adjList().get(gate);
                    if (gn != null) gn.addEdge(new GraphEdge(moeCombineName, "", 0L, true));
                }
                for (String expert : expertSinks) {
                    GraphNode en = graph.adjList().get(expert);
                    if (en != null) en.addEdge(new GraphEdge(moeCombineName, "", 0L, true));
                }
                for (String head : towerHeads) {
                    final String hd = head;
                    for (Map.Entry<String, GraphNode> se : graph.adjList().entrySet()) {
                        if (se.getKey().equals(hd) || se.getKey().equals(moeCombineName)) continue;
                        String sAttr = graph.nodeToAttrName().get(se.getKey());
                        String sL = sAttr == null ? "" : sAttr.toLowerCase();
                        if (sL.contains("tower")) continue;
                        se.getValue().edges().removeIf(edge -> hd.equals(edge.target()));
                    }
                    moeCombineNode.addEdge(new GraphEdge(head, "", 0L, true));
                }
            } else {
                // MMOE: one combine per gate, each feeding the matching tower
                for (Map.Entry<String, String> ge : gateByIndex.entrySet()) {
                    String gIdx = ge.getKey();
                    String gate = ge.getValue();
                    globalNodeCounter++;
                    String mcName = "moe_combine_" + globalNodeCounter;
                    graph.graphNodeNameToWithoutSuffix().put(mcName, "moe_combine");
                    graph.graphNodeDisplayNames().put(mcName, "moe_combine");
                    graph.nodeToModulePath().put(mcName, "torch");
                    GraphNode mcNode = GraphNode.of(NodeType.OPERATION);
                    graph.adjList().put(mcName, mcNode);
                    GraphNode gn = graph.adjList().get(gate);
                    if (gn != null) gn.addEdge(new GraphEdge(mcName, "", 0L, true));
                    for (String expert : expertSinks) {
                        GraphNode en = graph.adjList().get(expert);
                        if (en != null) en.addEdge(new GraphEdge(mcName, "", 0L, true));
                    }
                    String head = towerHeadByIndex.get(gIdx);
                    if (head != null) {
                        final String hd = head;
                        for (Map.Entry<String, GraphNode> se : graph.adjList().entrySet()) {
                            if (se.getKey().equals(hd) || se.getKey().equals(mcName)) continue;
                            String sAttr = graph.nodeToAttrName().get(se.getKey());
                            String sL = sAttr == null ? "" : sAttr.toLowerCase();
                            if (sL.contains("tower")) continue;
                            se.getValue().edges().removeIf(edge -> hd.equals(edge.target()));
                        }
                        mcNode.addEdge(new GraphEdge(head, "", 0L, true));
                    } else {
                        // No matching tower — feed all tower heads
                        for (String h : towerHeads) {
                            mcNode.addEdge(new GraphEdge(h, "", 0L, true));
                        }
                    }
                }
            }

            // Gate is no longer a sink — it feeds moe_combine
            gateSinks.clear();

            // Re-collect sinks after wiring experts/gate → moe_combine → towers
            sinks.removeIf(s -> {
                GraphNode n = graph.adjList().get(s);
                if (n == null) return true;
                for (GraphEdge edge : n.edges()) {
                    if (graph.adjList().containsKey(edge.target())) return true;
                }
                return false;
            });
        } else if (!expertSinks.isEmpty() && !gateSinks.isEmpty()
                && towerHeads.isEmpty() && predictLayerSinks.isEmpty()) {
            // CGC-like: no towers, no predictLayers, experts feed gates, gate
            // outputs are final. Wire experts → gates (all-to-all) and let
            // gates be sinks.
            for (String expert : expertSinks) {
                for (String gate : gateSinks) {
                    GraphNode en = graph.adjList().get(expert);
                    if (en != null) {
                        en.addEdge(new GraphEdge(gate, "", 0L, true));
                    }
                }
            }
            // Gates become sinks now
            sinks.addAll(gateSinks);
        } else if (!aitSinks.isEmpty()) {
            // AITM: AttentionLayer (ait_k) q/k/v children are sinks because the
            // attention computation (Q*K→softmax→*V→sum) can't be traced. Group
            // ait sinks by their ait index, create an "ait_out" operation node
            // per group, wire q/k/v → ait_out, then ait_out → tower_(k+1) head.
            // Also remove the spurious bottom_(k+1) → tower_(k+1) edge since
            // tower_(k+1) should receive ait_k's output, not bottom_(k+1)'s.
            Map<String, List<String>> aitGroups = new LinkedHashMap<>();
            for (String s : aitSinks) {
                String attr = graph.nodeToAttrName().get(s);
                String aL = attr == null ? "" : attr.toLowerCase();
                // Extract ait index: ait_0.q_layer → "0"
                int idx = aL.indexOf("ait_");
                String grp = "0";
                if (idx >= 0) {
                    int start = idx + 4;
                    int end = start;
                    while (end < aL.length() && Character.isDigit(aL.charAt(end))) end++;
                    if (end > start) grp = aL.substring(start, end);
                }
                aitGroups.computeIfAbsent(grp, k -> new ArrayList<>()).add(s);
            }
            for (Map.Entry<String, List<String>> ag : aitGroups.entrySet()) {
                String kIdx = ag.getKey();
                List<String> members = ag.getValue();
                // Create ait_out operation node
                globalNodeCounter++;
                String aitOutName = "ait_out_" + globalNodeCounter;
                graph.graphNodeNameToWithoutSuffix().put(aitOutName, "ait_out");
                graph.graphNodeDisplayNames().put(aitOutName, "ait_out");
                graph.nodeToModulePath().put(aitOutName, "torch");
                GraphNode aitOutNode = GraphNode.of(NodeType.OPERATION);
                graph.adjList().put(aitOutName, aitOutNode);
                // Wire q/k/v → ait_out
                for (String m : members) {
                    GraphNode mn = graph.adjList().get(m);
                    if (mn != null) {
                        mn.addEdge(new GraphEdge(aitOutName, "", 0L, true));
                    }
                }
                // Find tower_(k+1) head: tower-attributed node whose attr
                // contains tower_(k+1) and has an incoming edge from a
                // non-tower node.
                int nextIdx = Integer.parseInt(kIdx) + 1;
                String towerKey = "tower_" + nextIdx;
                String towerHead = null;
                for (Map.Entry<String, GraphNode> te : graph.adjList().entrySet()) {
                    String tName = te.getKey();
                    String tAttr = graph.nodeToAttrName().get(tName);
                    String tL = tAttr == null ? "" : tAttr.toLowerCase();
                    if (!tL.contains(towerKey)) continue;
                    // Check if this node has an incoming edge from a non-tower node
                    for (Map.Entry<String, GraphNode> se : graph.adjList().entrySet()) {
                        if (se.getKey().equals(tName)) continue;
                        String sAttr = graph.nodeToAttrName().get(se.getKey());
                        String sL = sAttr == null ? "" : sAttr.toLowerCase();
                        if (sL.contains("tower")) continue;
                        for (GraphEdge edge : se.getValue().edges()) {
                            if (tName.equals(edge.target())) {
                                towerHead = tName;
                                break;
                            }
                        }
                        if (towerHead != null) break;
                    }
                    if (towerHead != null) break;
                }
                if (towerHead != null) {
                    // Remove spurious non-tower edges to towerHead (e.g. bottom_(k+1))
                    final String th = towerHead;
                    for (Map.Entry<String, GraphNode> se : graph.adjList().entrySet()) {
                        if (se.getKey().equals(th) || se.getKey().equals(aitOutName)) continue;
                        String sAttr = graph.nodeToAttrName().get(se.getKey());
                        String sL = sAttr == null ? "" : sAttr.toLowerCase();
                        if (sL.contains("tower")) continue;
                        se.getValue().edges().removeIf(edge -> th.equals(edge.target()));
                    }
                    // Wire ait_out → tower_(k+1) head
                    aitOutNode.addEdge(new GraphEdge(towerHead, "", 0L, true));
                }
            }
        } else {
            // No expert-tower or expert-gate pairing. Gates are final outputs
            // only when no towers AND no predictLayers exist; otherwise gates
            // are intermediate (they produce weights, not final outputs).
            if (towerSinks.isEmpty() && predictLayerSinks.isEmpty()) {
                sinks.addAll(gateSinks);
            }
        }

        // Determine the combining operation when multiple sinks feed a single
        // output. Multi-task models (predictLayer/head/tower sinks) use
        // torch.cat; DeepFM-like models (different branch types like
        // linear/fm/mlp) use element-wise add.
        boolean needCombineOp = sinks.size() > 1 && outs.size() == 1;
        String combineOpName = null;
        if (needCombineOp) {
            // cat: multi-task outputs (predictLayer, head, tower) are concatenated
            // add: ensemble outputs (linear, fm, mlp, etc.) are summed
            boolean useCat = !predictLayerSinks.isEmpty()
                    || !towerSinks.isEmpty();
            globalNodeCounter++;
            combineOpName = (useCat ? "cat_" : "add_") + globalNodeCounter;
            String opLabel = useCat ? "cat" : "add";
            String opDisplay = useCat ? "cat(tasks)" : "add";
            graph.graphNodeNameToWithoutSuffix().put(combineOpName, opLabel);
            graph.graphNodeDisplayNames().put(combineOpName, opDisplay);
            graph.nodeToModulePath().put(combineOpName, "torch");
            GraphNode combineNode = GraphNode.of(NodeType.OPERATION);
            graph.adjList().put(combineOpName, combineNode);
            // Wire all sinks → combine op → output
            String primaryOut = outs.get(0);
            for (String sink : sinks) {
                GraphNode node = graph.adjList().get(sink);
                String dims = inDimHint.getOrDefault(sink, "(out)");
                node.addEdge(new GraphEdge(combineOpName, dims));
            }
            combineNode.addEdge(new GraphEdge(primaryOut, "(out)"));
        } else if (!sinks.isEmpty()) {
            // Pair sinks to outputs by index when counts match; else fan-in all sinks → all outs
            if (sinks.size() == outs.size()) {
                for (int i = 0; i < sinks.size(); i++) {
                    GraphNode node = graph.adjList().get(sinks.get(i));
                    String dims = inDimHint.getOrDefault(sinks.get(i),
                            metaShape(outs.get(i), "(out)"));
                    node.addEdge(new GraphEdge(outs.get(i), dims));
                }
            } else if (outs.size() == 1) {
                String primaryOut = outs.get(0);
                for (String sink : sinks) {
                    GraphNode node = graph.adjList().get(sink);
                    String dims = inDimHint.getOrDefault(sink, "(out)");
                    node.addEdge(new GraphEdge(primaryOut, dims));
                }
            } else {
                // Multiple outs, mismatched sink count: round-robin
                for (int i = 0; i < sinks.size(); i++) {
                    GraphNode node = graph.adjList().get(sinks.get(i));
                    String target = outs.get(i % outs.size());
                    String dims = inDimHint.getOrDefault(sinks.get(i), metaShape(target, "(out)"));
                    node.addEdge(new GraphEdge(target, dims));
                }
            }
        }

        // Wire orphaned EmbeddingImpl nodes to the downstream target found
        // earlier. This connects sibling EmbeddingImpl nodes that couldn't be
        // traced (Map-input forwards) to the same consumer as the one that was.
        if (embedDownstream != null) {
            for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
                String name = e.getKey();
                if (!name.contains("EmbeddingImpl")) continue;
                GraphNode node = e.getValue();
                boolean hasRealEdge = false;
                for (GraphEdge edge : node.edges()) {
                    if (graph.adjList().containsKey(edge.target())) {
                        hasRealEdge = true;
                        break;
                    }
                }
                if (!hasRealEdge) {
                    node.addEdge(new GraphEdge(embedDownstream, "", 0L, false));
                }
            }
        }

        // Post-processing: when multiple Module nodes connect directly to a
        // single output, insert a combining operation (cat/add) node. This
        // happens when the tracing engine wires multiple return-tensor sources
        // to the same output, bypassing the sink-fan-out logic above.
        for (String outName : new ArrayList<>(outputNodeSet)) {
            List<String> allIncoming = new ArrayList<>();
            for (Map.Entry<String, GraphNode> e : graph.adjList().entrySet()) {
                String name = e.getKey();
                if (name.equals(outName)) continue;
                GraphNode node = e.getValue();
                for (GraphEdge edge : node.edges()) {
                    if (outName.equals(edge.target())) {
                        allIncoming.add(name);
                        break;
                    }
                }
            }
            if (allIncoming.size() <= 1) continue;
            // Multiple incoming nodes — filter out intermediate layers that
            // were spuriously wired to output. A node is "intermediate" only
            // if it has other downstream edges (to non-output nodes). A node
            // that connects ONLY to output is a true sink, regardless of its
            // name (e.g. MetaHeac criticGate produces weights that can't be
            // traced further, so it IS the final sink for that branch).
            List<String> incoming = new ArrayList<>();
            List<String> toRemove = new ArrayList<>();
            for (String name : allIncoming) {
                GraphNode node = graph.adjList().get(name);
                boolean hasOtherEdges = false;
                for (GraphEdge edge : node.edges()) {
                    if (!outName.equals(edge.target())
                            && graph.adjList().containsKey(edge.target())) {
                        hasOtherEdges = true;
                        break;
                    }
                }
                if (hasOtherEdges) {
                    toRemove.add(name);
                } else {
                    incoming.add(name);
                }
            }
            for (String src : toRemove) {
                GraphNode sn = graph.adjList().get(src);
                if (sn != null) {
                    sn.edges().removeIf(ge -> outName.equals(ge.target()));
                }
            }
            if (incoming.size() <= 1) continue;
            boolean hasOp = false;
            for (String src : incoming) {
                GraphNode n = graph.adjList().get(src);
                if (n != null && n.nodeType() == NodeType.OPERATION) {
                    hasOp = true;
                    break;
                }
            }
            if (hasOp) continue;
            boolean useCat = false;
            for (String src : incoming) {
                String attrName = graph.nodeToAttrName().get(src);
                String attrL = attrName == null ? "" : attrName.toLowerCase();
                if (attrL.contains("predict") || attrL.startsWith("pred")
                        || attrL.contains("tower") || attrL.contains("head")
                        || attrL.contains("output_layer")) {
                    useCat = true;
                    break;
                }
            }
            globalNodeCounter++;
            String combineName = (useCat ? "cat_" : "add_") + globalNodeCounter;
            String opLabel = useCat ? "cat" : "add";
            graph.graphNodeNameToWithoutSuffix().put(combineName, opLabel);
            graph.graphNodeDisplayNames().put(combineName, useCat ? "cat(tasks)" : "add");
            graph.nodeToModulePath().put(combineName, "torch");
            GraphNode combineNode = GraphNode.of(NodeType.OPERATION);
            graph.adjList().put(combineName, combineNode);
            for (String src : incoming) {
                GraphNode sn = graph.adjList().get(src);
                if (sn == null) continue;
                sn.edges().removeIf(ge -> outName.equals(ge.target()));
                sn.addEdge(new GraphEdge(combineName, "(out)"));
            }
            combineNode.addEdge(new GraphEdge(outName, "(out)"));
        }
    }

    private String metaShape(String outName, String fallback) {
        Map<String, Object> meta = graph.nodeMeta().get(outName);
        if (meta != null && meta.get("shape") != null) return String.valueOf(meta.get("shape"));
        return fallback;
    }

    // =========================================================================
    // Feature / label catalog (sparse · dense · sequence · label)
    // =========================================================================

    /**
     * Walk the model tree + reflective fields to harvest {@code Feature} lists
     * from EmbeddingLayer / multi-task constructors and task/label names.
     * Non-invasive — never modifies the model.
     */
    private void collectModelFeatureCatalog(Module root) {
        featureCatalog.clear();
        taskLabelNames.clear();
        if (root == null) return;
        Set<Long> seen = new HashSet<>();
        ArrayDeque<Module> q = new ArrayDeque<>();
        q.add(ModuleDiscovery.concrete(root));
        while (!q.isEmpty()) {
            Module m = q.poll();
            if (m == null || m.isNull()) continue;
            long id = moduleId(m);
            if (!seen.add(id)) continue;
            // EmbeddingLayer.features()
            try {
                java.lang.reflect.Method fm = m.getClass().getMethod("features");
                Object res = fm.invoke(m);
                if (res instanceof List) {
                    for (Object o : (List<?>) res) ingestFeatureObject(o);
                }
            } catch (Throwable ignored) {}
            // Common field names holding Feature lists
            harvestFeatureFields(m);
            harvestTaskNameFields(m);
            for (ModuleChildren.NamedChild c : ModuleChildren.list(m)) {
                q.add(ModuleDiscovery.concrete(c.module));
            }
        }
    }

    private void harvestFeatureFields(Module m) {
        Class<?> c = m.getClass();
        while (c != null && c != Object.class && c != Module.class) {
            for (java.lang.reflect.Field f : c.getDeclaredFields()) {
                try {
                    f.setAccessible(true);
                    Object v = f.get(m);
                    if (v instanceof List) {
                        for (Object o : (List<?>) v) ingestFeatureObject(o);
                    } else {
                        ingestFeatureObject(v);
                    }
                } catch (Throwable ignored) {}
            }
            c = c.getSuperclass();
        }
    }

    private void harvestTaskNameFields(Module m) {
        String[] names = {"taskNames", "taskTypes", "tasks", "labels", "labelNames", "task_names"};
        Class<?> c = m.getClass();
        while (c != null && c != Object.class && c != Module.class) {
            for (String n : names) {
                try {
                    java.lang.reflect.Field f = c.getDeclaredField(n);
                    f.setAccessible(true);
                    Object v = f.get(m);
                    if (v instanceof List) {
                        for (Object o : (List<?>) v) {
                            if (o != null) {
                                String s = String.valueOf(o);
                                if (!s.isEmpty() && !taskLabelNames.contains(s)) taskLabelNames.add(s);
                            }
                        }
                    } else if (v instanceof String[]) {
                        for (String s : (String[]) v) {
                            if (s != null && !s.isEmpty() && !taskLabelNames.contains(s)) taskLabelNames.add(s);
                        }
                    }
                } catch (Throwable ignored) {}
            }
            // Also try getter
            for (String n : names) {
                try {
                    String getter = "get" + Character.toUpperCase(n.charAt(0)) + n.substring(1);
                    java.lang.reflect.Method gm = c.getMethod(getter);
                    Object v = gm.invoke(m);
                    if (v instanceof List) {
                        for (Object o : (List<?>) v) {
                            if (o != null) {
                                String s = String.valueOf(o);
                                if (!s.isEmpty() && !taskLabelNames.contains(s)) taskLabelNames.add(s);
                            }
                        }
                    }
                } catch (Throwable ignored) {}
            }
            c = c.getSuperclass();
        }
    }

    private void ingestFeatureObject(Object o) {
        if (o == null) return;
        // Duck-typed: anything with name() that looks like a Feature
        String simple = o.getClass().getSimpleName();
        boolean looksFeature = simple.contains("Feature")
                || simple.contains("Sparse") || simple.contains("Dense")
                || simple.contains("Sequence") || simple.contains("Label");
        String fname = null;
        try {
            java.lang.reflect.Method nm = o.getClass().getMethod("name");
            Object nv = nm.invoke(o);
            if (nv != null) fname = String.valueOf(nv);
        } catch (Throwable ignored) {}
        if (fname == null || fname.isEmpty()) {
            if (!looksFeature) return;
            return;
        }
        Map<String, Object> meta = new LinkedHashMap<>();
        meta.put("name", fname);
        String ftype = "feature";
        String sl = simple.toLowerCase();
        if (sl.contains("sparse")) ftype = "sparse";
        else if (sl.contains("dense")) ftype = "dense";
        else if (sl.contains("sequence") || sl.contains("seq")) ftype = "sequence";
        else if (sl.contains("label")) ftype = "label";
        else {
            // Interface default: try isSequence / vocabSize heuristics later
            try {
                java.lang.reflect.Method isSeq = o.getClass().getMethod("isSequence");
                Object r = isSeq.invoke(o);
                if (Boolean.TRUE.equals(r)) ftype = "sequence";
            } catch (Throwable ignored) {}
        }
        meta.put("feature_type", ftype);
        putNum(meta, o, "vocabSize", "vocab_size");
        putNum(meta, o, "embedDim", "embed_dim");
        putStr(meta, o, "pooling", "pooling");
        putNum(meta, o, "maxLen", "max_len");
        putNum(meta, o, "paddingIdx", "padding_idx");
        putStr(meta, o, "sharedWith", "shared_with");
        featureCatalog.put(fname, meta);
        // aliases
        featureCatalog.putIfAbsent("input_" + fname, meta);
        featureCatalog.putIfAbsent(fname.toLowerCase(), meta);
    }

    private static void putNum(Map<String, Object> meta, Object o, String method, String key) {
        try {
            java.lang.reflect.Method m = o.getClass().getMethod(method);
            Object v = m.invoke(o);
            if (v instanceof Number) meta.put(key, v);
        } catch (Throwable ignored) {}
    }

    private static void putStr(Map<String, Object> meta, Object o, String method, String key) {
        try {
            java.lang.reflect.Method m = o.getClass().getMethod(method);
            Object v = m.invoke(o);
            if (v != null) meta.put(key, String.valueOf(v));
        } catch (Throwable ignored) {}
    }

    private Map<String, Object> lookupFeatureCatalog(String display) {
        if (display == null) return null;
        Map<String, Object> hit = featureCatalog.get(display);
        if (hit != null) return new LinkedHashMap<>(hit);
        hit = featureCatalog.get("input_" + display);
        if (hit != null) return new LinkedHashMap<>(hit);
        // strip common prefixes
        String bare = display;
        if (bare.startsWith("input_")) bare = bare.substring("input_".length());
        if (bare.startsWith("embed_")) bare = bare.substring("embed_".length());
        hit = featureCatalog.get(bare);
        if (hit != null) return new LinkedHashMap<>(hit);
        hit = featureCatalog.get(bare.toLowerCase());
        if (hit != null) return new LinkedHashMap<>(hit);
        return null;
    }

    private static String guessFeatureType(String display, Tensor t) {
        String d = display == null ? "" : display.toLowerCase();
        if (d.contains("label") || d.contains("target") || d.equals("y") || d.startsWith("y_")) {
            return "label";
        }
        if (d.contains("seq") || d.contains("hist") || d.contains("click") || d.contains("behavior")) {
            return "sequence";
        }
        try {
            if (t != null && !t.isNull()) {
                long dim = t.dim();
                String dt = TensorUtils.safeDtype(t);
                if (dim >= 2 && ("Long".equals(dt) || "Int".equals(dt) || "Byte".equals(dt))) {
                    return "sequence";
                }
                if ("Float".equals(dt) || "Double".equals(dt) || "Half".equals(dt)) {
                    return "dense";
                }
                if ("Long".equals(dt) || "Int".equals(dt)) {
                    return "sparse";
                }
            }
        } catch (Throwable ignored) {}
        if (d.contains("dense") || d.contains("num") || d.contains("age") || d.contains("price")) {
            return "dense";
        }
        return "sparse";
    }

    /**
     * After forward, refine Output node meta with task names / Map keys and
     * ensure every output has shape filled.
     */
    private void enrichOutputMeta(Module root, Object output) {
        // If Map output, re-key display names
        if (output instanceof Map) {
            int i = 0;
            for (Map.Entry<?, ?> e : ((Map<?, ?>) output).entrySet()) {
                String label = String.valueOf(e.getKey());
                List<Tensor> ts = TensorUtils.extractTensors(e.getValue());
                if (ts.isEmpty()) continue;
                // Find matching output node by tensor key
                for (Tensor t : ts) {
                    long tid = TensorUtils.tensorKey(t);
                    // Search edges pointing to outputs or output nodes whose source matches
                    for (String outName : outputNodeSet) {
                        Map<String, Object> meta = graph.nodeMeta().get(outName);
                        if (meta == null) continue;
                        // Prefer exact name match or index order
                        boolean match = label.equals(meta.get("name"))
                                || label.equals(meta.get("label"))
                                || label.equals(meta.get("task"))
                                || (outputNodeSet.size() > 1 && outName.endsWith("_" + label));
                        if (!match && outputNodeSet.size() == 1 && i == 0) match = true;
                        if (!match) continue;
                        meta.put("label", label);
                        meta.put("task", label);
                        meta.put("name", label);
                        meta.put("shape", TensorUtils.formatDims(t));
                        meta.put("dtype", TensorUtils.safeDtype(t));
                        graph.graphNodeDisplayNames().put(outName, label);
                        graph.graphNodeNameToWithoutSuffix().put(outName, "LABEL " + label);
                    }
                }
                i++;
            }
        }
        // Attach taskLabelNames in order when outputs are anonymous
        if (!taskLabelNames.isEmpty()) {
            List<String> outs = new ArrayList<>(outputNodeSet);
            for (int i = 0; i < outs.size() && i < taskLabelNames.size(); i++) {
                String outName = outs.get(i);
                Map<String, Object> meta = graph.nodeMeta().computeIfAbsent(outName, k -> new LinkedHashMap<>());
                if (meta.get("label") == null || "output".equals(meta.get("label"))
                        || String.valueOf(meta.get("label")).startsWith("output")) {
                    String label = taskLabelNames.get(i);
                    meta.put("kind", "output");
                    meta.put("feature_type", "label");
                    meta.put("label", label);
                    meta.put("task", label);
                    meta.put("name", label);
                    graph.graphNodeDisplayNames().put(outName, label);
                    graph.graphNodeNameToWithoutSuffix().put(outName, "LABEL " + label);
                }
            }
        }
    }

    // =========================================================================
    // Forward dispatch
    // =========================================================================

    /**
     * Call {@code module.forward} with the best matching arity we can derive
     * from {@code inputs}. Supports Tensor / Tensor[] / List&lt;Tensor&gt; /
     * {@code Map&lt;String,Tensor&gt;} (recommend multi_task models).
     */
    private Tensor callForward(Module module, Object inputs) {
        // named_children() yields bare Module; re-type to *Impl so forward()
        // hits the real C++ implementation (see ModuleDiscovery.concrete).
        Module m = ModuleDiscovery.concrete(module);

        // Prefer typed SequentialImpl when applicable
        if (m instanceof SequentialImpl) {
            return callSequential((SequentialImpl) m, inputs);
        }
        try {
            SequentialImpl seq = m.asSequential();
            if (seq != null && !seq.isNull()) {
                return callSequential(seq, inputs);
            }
        } catch (Throwable ignored) {}

        // Map / List custom forwards (SharedBottom, ESMM, MMOE, …).
        // CRITICAL: never flatten a Map into multi-Tensor Module.forward_tensorN —
        // that hits unimplemented C++ shims (e.g. ESMM.forward_tensor4) and produces
        // the scary stack the user sees in the HTML error banner.
        if (inputs instanceof Map || inputs instanceof List) {
            Tensor mapped = invokeReflectiveForward(m, inputs);
            if (mapped != null) return mapped;
            throw new IllegalArgumentException(
                    "No matching forward(Map/List) on " + ModuleDiscovery.typeName(m)
                            + " (refusing to explode feature map into forward_tensorN)");
        }

        // Multi-arg Object[] payloads (MetaHeac Map+Tensor, HLLM Tensor+Tensor, …).
        // Prefer reflective match before exploding into C++ forward_tensorN.
        if (inputs instanceof Object[] && !(inputs instanceof Tensor[])) {
            Tensor mapped = invokeReflectiveForward(m, inputs);
            if (mapped != null) return mapped;
        }

        // Prefer Java-declared forward(Tensor…) on custom modules before C++ shims
        if (ModuleDiscovery.isCustomForwardModule(m)
                || ModuleDiscovery.findForwardMethod(m, inputs) != null) {
            Tensor mapped = invokeReflectiveForward(m, inputs);
            if (mapped != null) return mapped;
        }

        Tensor[] args = toTensorArgs(inputs);
        if (args.length == 0) {
            Tensor mapped = invokeReflectiveForward(m, inputs);
            if (mapped != null) return mapped;
            throw new IllegalArgumentException(
                    "No Tensor inputs to forward on " + ModuleDiscovery.typeName(m));
        }
        // Built-in *Impl: only use 1-arg (or exact) typed forward. Avoid 3/4-arg
        // shims unless the concrete class actually overrides them.
        try {
            if (args.length == 1) {
                return m.forward(args[0]);
            }
            if (args.length == 2
                    && ModuleAsHelper.hasForwardOverride(m, Tensor.class, Tensor.class)) {
                return m.forward(args[0], args[1]);
            }
            if (args.length >= 3) {
                // Multi-tensor without an override → reflective or first tensor only
                Tensor mapped = invokeReflectiveForward(m, inputs);
                if (mapped != null) return mapped;
                return m.forward(args[0]);
            }
            return m.forward(args[0]);
        } catch (RuntimeException e) {
            Tensor mapped = invokeReflectiveForward(m, inputs);
            if (mapped != null) return mapped;
            throw e;
        } catch (Exception e) {
            Tensor mapped = invokeReflectiveForward(m, inputs);
            if (mapped != null) return mapped;
            throw new RuntimeException(
                    "forward failed on " + ModuleDiscovery.typeName(m) + ": " + e.getMessage(), e);
        }
    }

    /**
     * Invoke a Java-declared {@code forward(...)} via reflection (Map/List/custom).
     * Returns null if no suitable method or invocation failed without a Tensor result.
     *
     * <p>Handles common recommend signatures:
     * <ul>
     *   <li>{@code forward(Map)}</li>
     *   <li>{@code forward(Map, Map)} — empty second map</li>
     *   <li>{@code forward(Map, Map, boolean)} — EmbeddingLayer squeeze=true</li>
     *   <li>{@code forward(List)}</li>
     *   <li>{@code forward(Tensor…)}</li>
     * </ul>
     */
    private Tensor invokeReflectiveForward(Module m, Object inputs) {
        java.lang.reflect.Method method = ModuleDiscovery.findForwardMethod(m, inputs);
        if (method == null) return null;
        try {
            Class<?>[] pts = method.getParameterTypes();
            Object[] callArgs = new Object[pts.length];

            // Unpack multi-arg payloads: Tensor[] / Object[] / List → positional slots
            Object[] positional = null;
            if (inputs instanceof Tensor[]) {
                positional = (Object[]) inputs;
            } else if (inputs instanceof Object[] && !(inputs instanceof Tensor)) {
                // plain Object[] (e.g. {Map, Tensor} for MetaHeac) — not a single Map/List/Tensor
                positional = (Object[]) inputs;
            } else if (inputs instanceof List) {
                List<?> list = (List<?>) inputs;
                // Multi-tensor list used as multi-arg only when method arity > 1
                if (pts.length > 1 && !list.isEmpty() && list.get(0) instanceof Tensor
                        && !List.class.isAssignableFrom(pts[0])) {
                    positional = list.toArray();
                }
            }

            for (int i = 0; i < pts.length; i++) {
                Class<?> p = pts[i];
                if (positional != null && i < positional.length && positional[i] != null) {
                    callArgs[i] = coerceArg(positional[i], p);
                    continue;
                }
                if (i == 0 && positional == null) {
                    callArgs[0] = coerceArg(inputs, p);
                    continue;
                }
                // Defaults for trailing optional-ish params (EmbeddingLayer squeeze, etc.)
                if (java.util.Map.class.isAssignableFrom(p)) {
                    callArgs[i] = java.util.Collections.emptyMap();
                } else if (p == boolean.class || p == Boolean.class) {
                    callArgs[i] = Boolean.TRUE;
                } else if (p == int.class || p == Integer.class) {
                    callArgs[i] = 0;
                } else if (p == long.class || p == Long.class) {
                    callArgs[i] = 0L;
                } else {
                    callArgs[i] = null;
                }
            }
            Object result = method.invoke(m, callArgs);
            if (result != null) lastForwardResult = result;
            if (result instanceof Tensor) return (Tensor) result;
            List<Tensor> extracted = TensorUtils.extractTensors(result);
            return extracted.isEmpty() ? null : extracted.get(0);
        } catch (Throwable t) {
            System.err.println("[DEBUG invokeReflectiveForward] FAILED on " + m.getClass().getName()
                    + " method=" + (method != null ? method.getName() : "null")
                    + " params=" + (method != null ? java.util.Arrays.toString(method.getParameterTypes()) : "[]")
                    + " err=" + t);
            return null;
        }
    }

    /**
     * Run a one-shot reflective forward solely to capture a rich result object
     * (e.g. {@code Map<String,Tensor>} of multi-task labels). Does not record
     * graph nodes — the structural expand already did that.
     */
    private Object tryCaptureFullResult(Module root, Object inputs) {
        try {
            Module m = ModuleDiscovery.concrete(root);
            // If lastForwardResult already holds a Map/List from expand, reuse it
            if (lastForwardResult instanceof Map || lastForwardResult instanceof List) {
                return lastForwardResult;
            }
            java.lang.reflect.Method method = ModuleDiscovery.findForwardMethod(m, inputs);
            if (method == null) return lastForwardResult;
            Class<?>[] pts = method.getParameterTypes();
            Object[] callArgs = new Object[pts.length];
            Object[] positional = null;
            if (inputs instanceof Object[] && !(inputs instanceof Tensor[])) {
                positional = (Object[]) inputs;
            } else if (inputs instanceof Tensor[]) {
                positional = (Object[]) inputs;
            }
            for (int i = 0; i < pts.length; i++) {
                Class<?> p = pts[i];
                if (positional != null && i < positional.length && positional[i] != null) {
                    callArgs[i] = coerceArg(positional[i], p);
                } else if (i == 0) {
                    callArgs[0] = coerceArg(inputs, p);
                } else if (Map.class.isAssignableFrom(p)) {
                    callArgs[i] = Collections.emptyMap();
                } else if (p == boolean.class || p == Boolean.class) {
                    callArgs[i] = Boolean.TRUE;
                } else if (p == int.class || p == Integer.class) {
                    callArgs[i] = 0;
                } else if (p == long.class || p == Long.class) {
                    callArgs[i] = 0L;
                } else {
                    callArgs[i] = null;
                }
            }
            Object result = method.invoke(m, callArgs);
            if (result != null) lastForwardResult = result;
            return result;
        } catch (Throwable t) {
            return lastForwardResult;
        }
    }

    private static Object coerceArg(Object inputs, Class<?> target) {
        if (inputs == null) return null;
        if (target.isInstance(inputs)) return inputs;
        if (Map.class.isAssignableFrom(target) && inputs instanceof Map) return inputs;
        if (List.class.isAssignableFrom(target) && inputs instanceof List) return inputs;
        if (Tensor.class.isAssignableFrom(target)) {
            if (inputs instanceof Tensor) return inputs;
            List<Tensor> ts = TensorUtils.extractTensors(inputs);
            return ts.isEmpty() ? null : ts.get(0);
        }
        return inputs;
    }

    private Tensor callSequential(SequentialImpl seq, Object inputs) {
        Tensor[] args = toTensorArgs(inputs);
        if (args.length == 0) {
            throw new IllegalArgumentException("No Tensor inputs for Sequential.forward");
        }
        switch (args.length) {
            case 1:
                return seq.forward(args[0]);
            case 2:
                return seq.forward(args[0], args[1]);
            case 3:
                return seq.forward(args[0], args[1], args[2]);
            default:
                return seq.forward(args[0]);
        }
    }

    private static Tensor[] toTensorArgs(Object inputs) {
        if (inputs == null) return new Tensor[0];
        if (inputs instanceof Tensor) {
            Tensor t = (Tensor) inputs;
            return t.isNull() ? new Tensor[0] : new Tensor[]{t};
        }
        if (inputs instanceof Tensor[]) {
            return (Tensor[]) inputs;
        }
        List<Tensor> list = TensorUtils.extractTensors(inputs);
        return list.toArray(new Tensor[0]);
    }

    // =========================================================================
    // Graph cleanup (torchvista cleanup_graph)
    // =========================================================================

    private void cleanupGraph() {
        Map<String, GraphNode> adj = graph.adjList();

        // Remove nodes that produced no tensors
        for (String node : nodesToDelete) {
            adj.remove(node);
            for (GraphNode n : adj.values()) {
                n.edges().removeIf(e -> node.equals(e.target()));
            }
        }

        // Forward-reachable from inputs
        Set<String> forwardReachable = new HashSet<>();
        ArrayDeque<String> q = new ArrayDeque<>();
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            if (e.getValue().nodeType() == NodeType.INPUT) {
                q.add(e.getKey());
            }
        }
        while (!q.isEmpty()) {
            String n = q.poll();
            if (!forwardReachable.add(n)) continue;
            GraphNode node = adj.get(n);
            if (node == null) continue;
            for (GraphEdge edge : node.edges()) {
                if (adj.containsKey(edge.target())) q.add(edge.target());
            }
        }

        // Reverse adj
        Map<String, List<String>> reverse = new HashMap<>();
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            for (GraphEdge edge : e.getValue().edges()) {
                reverse.computeIfAbsent(edge.target(), k -> new ArrayList<>()).add(e.getKey());
            }
        }

        // Backward-reachable from outputs
        Set<String> backwardReachable = new HashSet<>();
        q.clear();
        for (String out : outputNodeSet) {
            if (adj.containsKey(out)) q.add(out);
        }
        // If forward failed before producing outputs, keep the last successful
        // op and anything that feeds the failed current op so the partial
        // graph remains useful for debugging (torchvista keeps failed nodes).
        if (q.isEmpty()) {
            if (currentOp != null && adj.containsKey(currentOp)) q.add(currentOp);
            if (lastSuccessfulOp != null && adj.containsKey(lastSuccessfulOp)) {
                q.add(lastSuccessfulOp);
            }
            // Also keep all failed nodes
            for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
                if (e.getValue().failed()) q.add(e.getKey());
            }
        }
        while (!q.isEmpty()) {
            String n = q.poll();
            if (!backwardReachable.add(n)) continue;
            List<String> preds = reverse.get(n);
            if (preds != null) q.addAll(preds);
        }

        Set<String> base = new HashSet<>(forwardReachable);
        base.addAll(backwardReachable);

        // Expand: all ancestors of base
        Set<String> expanded = new HashSet<>();
        q.clear();
        q.addAll(base);
        while (!q.isEmpty()) {
            String n = q.poll();
            if (!expanded.add(n)) continue;
            List<String> preds = reverse.get(n);
            if (preds != null) q.addAll(preds);
        }

        // Always keep failed nodes that were reached forward (debug partial graph)
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            if (e.getValue().failed() && forwardReachable.contains(e.getKey())) {
                expanded.add(e.getKey());
            }
        }

        // Prune
        List<String> toRemove = new ArrayList<>();
        for (String n : adj.keySet()) {
            if (!expanded.contains(n)) toRemove.add(n);
        }
        for (String n : toRemove) {
            adj.remove(n);
        }
        for (GraphNode node : adj.values()) {
            node.edges().removeIf(e -> !adj.containsKey(e.target()));
        }

        // Breakpoint repair: connect orphaned intermediate nodes (breakpoints)
        // to their correct downstream targets BEFORE filtering embedding edges.
        // This handles two scenarios:
        //  (A) embedding wrongly connected to combining op (add/cat) → mlp/linear
        //      left as breakpoint. Redirect: embedding → breakpoint → combining op.
        //  (B) mlp/etc. disconnected from output/combining op → wire to nearest
        //      terminal via BFS.
        repairBreakpoints(adj);

        // Filter: feature embeddings must NEVER connect directly to output nodes
        // or combining ops (add/cat) that feed output. They should flow through
        // downstream modules (linear/mlp/fm/...) first. Only strip the direct
        // edge when the embedding has other downstream consumers — codebook
        // embeddings (VQ-VAE) with no other consumers are left intact.
        Set<String> preOutputOps = new HashSet<>();
        for (String out : outputNodeSet) {
            for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
                for (GraphEdge edge : e.getValue().edges()) {
                    if (out.equals(edge.target())) {
                        String name = e.getKey();
                        String display = graph.graphNodeNameToWithoutSuffix()
                                .getOrDefault(name, name).toLowerCase();
                        if (display.contains("add") || display.contains("cat")
                                || name.startsWith("output_op_") || name.startsWith("cat_")) {
                            preOutputOps.add(name);
                        }
                    }
                }
            }
        }
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            String name = e.getKey();
            String attr = graph.nodeToAttrName().get(name);
            String attrL = attr == null ? "" : attr.toLowerCase();
            String display = graph.graphNodeNameToWithoutSuffix()
                    .getOrDefault(name, name).toLowerCase();
            boolean isEmbedding = name.toLowerCase().contains("embeddingimpl")
                    || name.toLowerCase().contains("embedding")
                    || attrL.contains("embed")
                    || display.contains("embedding") || display.contains("cat(embed)");
            if (!isEmbedding) continue;
            boolean hasOtherDownstream = false;
            for (GraphEdge edge : e.getValue().edges()) {
                if (!outputNodeSet.contains(edge.target()) && !preOutputOps.contains(edge.target())) {
                    hasOtherDownstream = true;
                    break;
                }
            }
            if (hasOtherDownstream) {
                final Set<String> block = preOutputOps;
                e.getValue().edges().removeIf(edge ->
                        outputNodeSet.contains(edge.target()) || block.contains(edge.target()));
            }
        }

        // Prune side tables
        pruneMapKeys(graph.moduleInfo(), adj.keySet());
        pruneMapKeys(graph.funcInfo(), adj.keySet());
        pruneMapKeys(graph.nodeToModulePath(), adj.keySet());
        pruneMapKeys(graph.graphNodeDisplayNames(), adj.keySet());
        pruneMapKeys(graph.graphNodeNameToWithoutSuffix(), adj.keySet());
        pruneMapKeys(graph.nodeToAttrName(), adj.keySet());
        pruneMapKeys(graph.nodeToAncestors(), adj.keySet());
        pruneMapKeys(graph.nodeMeta(), adj.keySet());
    }

    /**
     * Connect orphaned intermediate nodes (breakpoints) to their correct
     * downstream targets. Two scenarios:
     *  (A) embedding wrongly connected to combining op (add/cat) → mlp/linear
     *      left as breakpoint. Redirect: embedding → breakpoint → combining op.
     *  (B) mlp/etc. disconnected from output/combining op → wire to nearest
     *      terminal via BFS.
     */
    private void repairBreakpoints(Map<String, GraphNode> adj) {
        // Collect output nodes and combining ops (add_/cat_/moe_combine_/ait_out_)
        Set<String> terminalTargets = new HashSet<>(outputNodeSet);
        Set<String> combiningOps = new HashSet<>();
        for (String n : adj.keySet()) {
            String display = graph.graphNodeNameToWithoutSuffix().getOrDefault(n, n).toLowerCase();
            if (display.startsWith("add") || display.startsWith("cat")
                    || display.contains("moe_combine") || display.contains("ait_out")
                    || display.contains("output_op") || n.startsWith("output_op_")) {
                terminalTargets.add(n);
                combiningOps.add(n);
            }
        }

        // Find breakpoints: intermediate nodes with no real outgoing edge
        List<String> breakpoints = new ArrayList<>();
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            String name = e.getKey();
            GraphNode node = e.getValue();
            if (node.nodeType() == NodeType.INPUT
                    || node.nodeType() == NodeType.OUTPUT
                    || node.nodeType() == NodeType.CONSTANT
                    || node.nodeType() == NodeType.PARAMETER) {
                continue;
            }
            // Skip embedding nodes — they are sources, not breakpoints
            if (isEmbeddingNodeName(name)) continue;
            boolean hasRealEdge = false;
            for (GraphEdge edge : node.edges()) {
                if (adj.containsKey(edge.target())) {
                    hasRealEdge = true;
                    break;
                }
            }
            if (!hasRealEdge) {
                breakpoints.add(name);
            }
        }

        if (breakpoints.isEmpty()) return;

        // Build reverse adjacency for BFS (who points to whom)
        Map<String, List<String>> reverse = new HashMap<>();
        for (Map.Entry<String, GraphNode> e : adj.entrySet()) {
            for (GraphEdge edge : e.getValue().edges()) {
                reverse.computeIfAbsent(edge.target(), k -> new ArrayList<>()).add(e.getKey());
            }
        }

        // Scenario A: for each breakpoint, check if an embedding node is
        // wrongly connected to a combining op. If so, redirect:
        //   embedding → combining op  becomes  embedding → breakpoint → combining op
        for (String bp : breakpoints) {
            for (String embName : new ArrayList<>(adj.keySet())) {
                if (!isEmbeddingNodeName(embName)) continue;
                GraphNode embNode = adj.get(embName);
                if (embNode == null) continue;
                // Find edges from this embedding to any combining op
                List<String> embToCombining = new ArrayList<>();
                for (GraphEdge edge : embNode.edges()) {
                    if (combiningOps.contains(edge.target()) && adj.containsKey(edge.target())) {
                        embToCombining.add(edge.target());
                    }
                }
                if (embToCombining.isEmpty()) continue;
                // Redirect: remove embedding → combining, add embedding → bp, bp → combining
                for (String combOp : embToCombining) {
                    embNode.edges().removeIf(edge -> combOp.equals(edge.target()));
                    embNode.addEdge(new GraphEdge(bp, "", 0L, false));
                    GraphNode bpNode = adj.get(bp);
                    if (bpNode != null) {
                        boolean exists = false;
                        for (GraphEdge ge : bpNode.edges()) {
                            if (combOp.equals(ge.target())) { exists = true; break; }
                        }
                        if (!exists) {
                            bpNode.addEdge(new GraphEdge(combOp, "", 0L, false));
                        }
                    }
                }
            }
        }

        // Re-collect breakpoints after scenario A repair (some may be fixed)
        breakpoints.removeIf(bp -> {
            GraphNode n = adj.get(bp);
            if (n == null) return true;
            for (GraphEdge edge : n.edges()) {
                if (adj.containsKey(edge.target())) return true;
            }
            return false;
        });

        // Scenario B: for remaining breakpoints, BFS to find nearest terminal
        String primaryOut = outputNodeSet.isEmpty() ? null : outputNodeSet.iterator().next();
        for (String bp : breakpoints) {
            String target = findNearestTerminal(adj, bp, terminalTargets, reverse);
            if (target == null) target = primaryOut;
            if (target == null || target.equals(bp)) continue;
            GraphNode bpNode = adj.get(bp);
            if (bpNode == null) continue;
            boolean exists = false;
            for (GraphEdge ge : bpNode.edges()) {
                if (target.equals(ge.target())) { exists = true; break; }
            }
            if (!exists) {
                bpNode.addEdge(new GraphEdge(target, "", 0L, false));
            }
        }
    }

    /** True if a node name represents an embedding source. */
    private boolean isEmbeddingNodeName(String name) {
        if (name == null) return false;
        String lower = name.toLowerCase();
        String attr = graph.nodeToAttrName().get(name);
        String attrL = attr == null ? "" : attr.toLowerCase();
        String display = graph.graphNodeNameToWithoutSuffix().getOrDefault(name, name).toLowerCase();
        return lower.contains("embeddingimpl") || lower.contains("embedding")
                || attrL.contains("embed")
                || display.contains("embedding") || display.contains("cat(embed)");
    }

    /**
     * BFS to find the nearest terminal target reachable from a breakpoint.
     * Starts from the breakpoint's siblings (other successors of its
     * predecessors) and walks forward until a terminal is found.
     */
    private String findNearestTerminal(Map<String, GraphNode> adj, String start,
                                       Set<String> terminals,
                                       Map<String, List<String>> reverse) {
        // Collect predecessors of the breakpoint
        Set<String> preds = new HashSet<>();
        List<String> predList = reverse.get(start);
        if (predList != null) preds.addAll(predList);

        // BFS forward from predecessors' other successors
        ArrayDeque<String> queue = new ArrayDeque<>();
        Set<String> visited = new HashSet<>();
        // Seed with siblings: other targets of the same predecessors
        for (String pred : preds) {
            GraphNode pn = adj.get(pred);
            if (pn == null) continue;
            for (GraphEdge edge : pn.edges()) {
                String tgt = edge.target();
                if (tgt.equals(start) || visited.contains(tgt)) continue;
                if (adj.containsKey(tgt)) {
                    visited.add(tgt);
                    queue.add(tgt);
                }
            }
        }
        // Also seed from the breakpoint itself (in case it should connect to
        // a terminal directly reachable via structural position)
        int maxHops = 6;
        while (!queue.isEmpty() && maxHops > 0) {
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                String n = queue.poll();
                if (n == null) continue;
                if (terminals.contains(n)) return n;
                GraphNode node = adj.get(n);
                if (node == null) continue;
                for (GraphEdge edge : node.edges()) {
                    String tgt = edge.target();
                    if (visited.contains(tgt) || tgt.equals(start)) continue;
                    if (adj.containsKey(tgt)) {
                        visited.add(tgt);
                        queue.add(tgt);
                    }
                }
            }
            maxHops--;
        }
        // Fallback: any terminal in the graph
        for (String t : terminals) {
            if (adj.containsKey(t) && !t.equals(start)) return t;
        }
        return null;
    }

    private static void pruneMapKeys(Map<String, ?> map, Set<String> keep) {
        map.keySet().removeIf(k -> !keep.contains(k));
    }

    // =========================================================================
    // Naming helpers
    // =========================================================================

    private String nextModuleNodeName(String typeSimple, Module m) {
        globalNodeCounter++;
        String nodeName = typeSimple + "_" + globalNodeCounter;
        long id = moduleId(m);
        moduleToNodeNames.computeIfAbsent(id, k -> new ArrayList<>()).add(nodeName);
        return nodeName;
    }

    private String displayNameFor(Module m, String typeSimple) {
        if (options.showModuleAttrNames()) {
            String attr = moduleToAttrName.get(moduleId(m));
            if (attr != null && !attr.isEmpty()) return attr;
        }
        return typeSimple;
    }

    private static String packagePathFor(Module m) {
        try {
            Package p = m.getClass().getPackage();
            if (p != null && p.getName() != null) return p.getName();
        } catch (Throwable ignored) {}
        return "org.bytedeco.pytorch.nn";
    }

    private static long moduleId(Module m) {
        if (m == null || m.isNull()) return 0L;
        try {
            long id = m.moduleObjectId();
            if (id != 0L) return id;
        } catch (Throwable ignored) {}
        return m.address();
    }
}