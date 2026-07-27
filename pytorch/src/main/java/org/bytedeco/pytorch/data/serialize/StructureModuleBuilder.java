package org.bytedeco.pytorch.data.serialize;
import org.bytedeco.pytorch.nn.*;
import org.bytedeco.pytorch.nn.options.*;
import org.bytedeco.pytorch.nn.modules.*;
import org.bytedeco.pytorch.nn.modules.container.*;
import org.bytedeco.pytorch.jit.*;
import org.bytedeco.pytorch.autograd.*;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.pytorch.DoubleOptional;
import org.bytedeco.pytorch.LongVector;
import org.bytedeco.pytorch.NoGradGuard;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModuleAsHelper;
import org.bytedeco.pytorch.nn.modules.BatchNorm1dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm2dImpl;
import org.bytedeco.pytorch.nn.modules.BatchNorm3dImpl;
import org.bytedeco.pytorch.nn.modules.Conv1dImpl;
import org.bytedeco.pytorch.nn.modules.Conv2dImpl;
import org.bytedeco.pytorch.nn.modules.Conv3dImpl;
import org.bytedeco.pytorch.nn.modules.DropoutImpl;
import org.bytedeco.pytorch.nn.modules.EmbeddingImpl;
import org.bytedeco.pytorch.nn.modules.GELUImpl;
import org.bytedeco.pytorch.nn.modules.GroupNormImpl;
import org.bytedeco.pytorch.nn.modules.IdentityImpl;
import org.bytedeco.pytorch.nn.modules.LayerNormImpl;
import org.bytedeco.pytorch.nn.modules.LeakyReLUImpl;
import org.bytedeco.pytorch.nn.modules.LinearImpl;
import org.bytedeco.pytorch.nn.modules.ReLU6Impl;
import org.bytedeco.pytorch.nn.modules.ReLUImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;
import org.bytedeco.pytorch.nn.modules.SiLUImpl;
import org.bytedeco.pytorch.nn.modules.SigmoidImpl;
import org.bytedeco.pytorch.nn.modules.SoftmaxImpl;
import org.bytedeco.pytorch.nn.modules.TanhImpl;
import org.bytedeco.pytorch.nn.options.BatchNormOptions;
import org.bytedeco.pytorch.nn.options.EmbeddingOptions;
import org.bytedeco.pytorch.nn.options.GroupNormOptions;
import org.bytedeco.pytorch.nn.options.LayerNormOptions;
import org.bytedeco.pytorch.nn.options.LinearOptions;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Build a typed {@link WeightBagModule} from a precise {@link StructureSpec} +
 * state_dict — <b>no heuristics, no safetensors required</b>.
 *
 * <p>Algorithm:
 * <ol>
 *   <li>Walk structure tree in child-registration order</li>
 *   <li>Materialize containers (Sequential / ModuleList / ModuleDict / COMPOSITE)
 *       and leaves (Linear / Embedding / BN / Dropout(p) / Softmax / …)</li>
 *   <li>Bind tensors from {@code state_dict} by exact dotted key</li>
 * </ol>
 *
 * <p>Param-free nodes (ReLU, Dropout, Softmax, Sigmoid, Identity, …) come only
 * from the structure dump — they never appear in a pure state_dict.
 */
public final class StructureModuleBuilder {
    private StructureModuleBuilder() {}

    /**
     * Build a trainable WeightBagModule from precise structure + weights.
     *
     * @param spec          schema-v2 structure
     * @param stateDict     tensors from {@link TorchPthReader#loadStateDict}
     * @param requiresGrad  set requires_grad on parameters (not buffers)
     */
    public static WeightBagModule build(StructureSpec spec,
                                        Map<String, Tensor> stateDict,
                                        boolean requiresGrad) {
        return build(spec, stateDict, requiresGrad, /*strict=*/false);
    }

    /**
     * @param strict when true, fail if a structure-declared parameter key is
     *               missing from state_dict (or vice-versa for declared parameters list)
     */
    public static WeightBagModule build(StructureSpec spec,
                                        Map<String, Tensor> stateDict,
                                        boolean requiresGrad,
                                        boolean strict) {
        Objects.requireNonNull(spec, "spec");
        if (stateDict == null) stateDict = Map.of();

        WeightBagModule root = new WeightBagModule();
        Map<String, Module> nodes = new LinkedHashMap<>();
        Map<String, Tensor> owned = new LinkedHashMap<>();
        Map<String, Module> children = new LinkedHashMap<>();
        List<StateDictModuleBuilder.LayerInfo> layers = new ArrayList<>();

        nodes.put("", root);

        // 1) Create full module tree from structure (empty weights)
        createTree(spec, "", root, nodes, children, layers);

        // 2) Bind state_dict tensors onto leaves
        int bound = bindWeights(spec, stateDict, nodes, owned, requiresGrad, strict);

        // 3) Adopt into WeightBagModule bookkeeping
        root.adoptPreciseBuild(owned, children, layers, spec.toCompactMeta(), requiresGrad);

        if (strict && !spec.parameters.isEmpty()) {
            for (String k : spec.parameters) {
                if (!stateDict.containsKey(k)) {
                    throw new IllegalStateException("strict: structure parameter missing in state_dict: " + k);
                }
            }
        }
        System.out.println("[StructureModuleBuilder] built precise tree nodes="
                + nodes.size() + " bound_tensors=" + bound
                + " owned=" + owned.size() + " root=" + spec.root);
        return root;
    }

    /** Empty architecture only (for native .pt load into prebuilt tree). */
    public static WeightBagModule buildEmpty(StructureSpec spec) {
        return build(spec, Map.of(), /*requiresGrad=*/true, /*strict=*/false);
    }

    // ---- tree construction ----------------------------------------------------

    private static void createTree(StructureSpec spec, String path, Module parent,
                                   Map<String, Module> nodes,
                                   Map<String, Module> children,
                                   List<StateDictModuleBuilder.LayerInfo> layers) {
        StructureSpec.Node node = spec.node(path);
        if (node == null) return;

        List<String> kids = node.children;
        for (String childName : kids) {
            String childPath = path.isEmpty() ? childName : path + "." + childName;
            StructureSpec.Node childNode = spec.node(childPath);
            if (childNode == null) {
                // synthesize bare container so deeper paths still attach
                childNode = new StructureSpec.Node("COMPOSITE", childName,
                        List.of(), Map.of(), List.of(), List.of());
            }
            Module childMod = materialize(childNode, childPath, layers);
            attach(parent, childName, childMod, node);
            nodes.put(childPath, childMod);
            children.put(childPath, childMod);
            try { ModuleAsHelper.remember(childMod); } catch (Throwable ignored) {}

            // recurse into containers / composites
            if (childNode.isContainer() || !childNode.children.isEmpty()) {
                createTree(spec, childPath, childMod, nodes, children, layers);
            }
        }

        // Bare parameters on this module (e.g. interest_queries) — register empty
        // placeholders; bindWeights will copy_ real values.
        for (String pname : node.ownParameters) {
            String full = path.isEmpty() ? pname : path + "." + pname;
            // actual tensor arrives from state_dict; nothing to create if leaf is not a Module
            // WeightBagModule will register_parameter at bind time if missing.
        }
    }

    private static void attach(Module parent, String name, Module child, StructureSpec.Node parentNode) {
        if (parent == null || child == null || name == null) return;
        try { ModuleAsHelper.remember(child); } catch (Throwable ignored) {}

        boolean asSeq = parent instanceof SequentialImpl
                || (parentNode != null && parentNode.isSequential());
        if (asSeq && parent instanceof SequentialImpl) {
            try {
                ((SequentialImpl) parent).push_back(name, child);
                return;
            } catch (Throwable ignored) {}
        }
        try {
            parent.register_module(name, child);
        } catch (Throwable t) {
            // last resort
            try {
                if (parent instanceof SequentialImpl) {
                    ((SequentialImpl) parent).push_back(name, child);
                }
            } catch (Throwable ignored) {}
        }
    }

    private static Module materialize(StructureSpec.Node node, String path,
                                      List<StateDictModuleBuilder.LayerInfo> layers) {
        String kind = node.kind == null ? "COMPOSITE" : node.kind.toUpperCase(Locale.ROOT);
        if (kind.startsWith("COMPOSITE:")) kind = "COMPOSITE";
        if (kind.startsWith("DROPOUT:")) {
            // tolerate compact token form
            double p = 0.5;
            try { p = Double.parseDouble(node.kind.substring("DROPOUT:".length())); } catch (Exception ignored) {}
            Module m = new DropoutImpl(p);
            recordLayer(layers, path, "DROPOUT", Map.of("p", p));
            return m;
        }
        if (kind.startsWith("SOFTMAX:")) {
            long dim = -1;
            try { dim = Long.parseLong(node.kind.substring("SOFTMAX:".length())); } catch (Exception ignored) {}
            Module m = new SoftmaxImpl(dim);
            recordLayer(layers, path, "SOFTMAX", Map.of("dim", dim));
            return m;
        }

        switch (kind) {
            case "SEQUENTIAL":
                recordLayer(layers, path, "SEQUENTIAL", Map.of());
                return new SequentialImpl();
            case "MODULE_LIST":
            case "MODULE_DICT":
            case "CONTAINER":
            case "COMPOSITE":
                // Named container Module — preserves ModuleList/ModuleDict topology
                // without Sequential push_back semantics.
                recordLayer(layers, path, kind, Map.of("class_name",
                        node.className != null ? node.className : kind));
                return new Module(node.className != null ? node.className : kind.toLowerCase(Locale.ROOT));

            case "LINEAR": {
                long inF = node.hyperLong("in_features", 0);
                long outF = node.hyperLong("out_features", 0);
                boolean bias = node.hyperBool("bias", true);
                if (inF <= 0 || outF <= 0) {
                    throw new IllegalArgumentException("LINEAR " + path + " missing in/out features");
                }
                LinearOptions opt = new LinearOptions(inF, outF).bias(bias);
                LinearImpl lin = new LinearImpl(opt);
                recordLayer(layers, path, "LINEAR",
                        Map.of("in_features", inF, "out_features", outF, "bias", bias));
                return lin;
            }
            case "EMBEDDING": {
                long num = node.hyperLong("num_embeddings", 0);
                long dim = node.hyperLong("embedding_dim", 0);
                if (num <= 0 || dim <= 0) {
                    throw new IllegalArgumentException("EMBEDDING " + path + " missing num/dim");
                }
                EmbeddingImpl emb = new EmbeddingImpl(new EmbeddingOptions(num, dim));
                recordLayer(layers, path, "EMBEDDING",
                        Map.of("num_embeddings", num, "embedding_dim", dim));
                return emb;
            }
            case "LAYER_NORM": {
                long[] ns = node.hyperLongArray("normalized_shape");
                if (ns == null || ns.length == 0) ns = new long[]{node.hyperLong("normalized_shape", 1)};
                LongVector shape = new LongVector(ns);
                LayerNormOptions opt = new LayerNormOptions(shape);
                opt.elementwise_affine(node.hyperBool("elementwise_affine", true));
                LayerNormImpl ln = new LayerNormImpl(opt);
                recordLayer(layers, path, "LAYER_NORM", Map.of());
                return ln;
            }
            case "BATCH_NORM_1D":
            case "BATCH_NORM_2D":
            case "BATCH_NORM_3D": {
                long nf = node.hyperLong("num_features", 0);
                if (nf <= 0) throw new IllegalArgumentException("BN " + path + " missing num_features");
                BatchNormOptions opt = new BatchNormOptions(nf)
                        .affine(node.hyperBool("affine", true))
                        .track_running_stats(node.hyperBool("track_running_stats", true))
                        .eps(node.hyperDouble("eps", 1e-5));
                // momentum optional
                double mom = node.hyperDouble("momentum", 0.1);
                try {
                    opt.momentum(new DoubleOptional(mom));
                } catch (Throwable ignored) {}
                Module bn;
                if ("BATCH_NORM_2D".equals(kind)) bn = new BatchNorm2dImpl(opt);
                else if ("BATCH_NORM_3D".equals(kind)) bn = new BatchNorm3dImpl(opt);
                else bn = new BatchNorm1dImpl(opt);
                recordLayer(layers, path, kind, Map.of("num_features", nf,
                        "affine", node.hyperBool("affine", true)));
                return bn;
            }
            case "GROUP_NORM": {
                long groups = node.hyperLong("num_groups", 1);
                long channels = node.hyperLong("num_channels", 1);
                GroupNormImpl gn = new GroupNormImpl(new GroupNormOptions(groups, channels));
                recordLayer(layers, path, "GROUP_NORM", Map.of());
                return gn;
            }
            case "CONV_1D":
            case "CONV_2D":
            case "CONV_3D":
                // Minimal: fall back to COMPOSITE if full options too heavy; weights still bind via bag
                recordLayer(layers, path, kind, Map.of());
                return new Module(kind.toLowerCase(Locale.ROOT));

            case "RELU":
                recordLayer(layers, path, "RELU", Map.of());
                return new ReLUImpl();
            case "RELU6":
                recordLayer(layers, path, "RELU6", Map.of());
                return new ReLU6Impl();
            case "LEAKY_RELU":
                recordLayer(layers, path, "LEAKY_RELU", Map.of());
                return new LeakyReLUImpl();
            case "GELU":
                recordLayer(layers, path, "GELU", Map.of());
                return new GELUImpl();
            case "SILU":
                recordLayer(layers, path, "SILU", Map.of());
                return new SiLUImpl();
            case "TANH":
                recordLayer(layers, path, "TANH", Map.of());
                return new TanhImpl();
            case "SIGMOID":
                recordLayer(layers, path, "SIGMOID", Map.of());
                return new SigmoidImpl();
            case "SOFTMAX": {
                long dim = node.hyperLong("dim", -1);
                SoftmaxImpl sm = new SoftmaxImpl(dim);
                recordLayer(layers, path, "SOFTMAX", Map.of("dim", dim));
                return sm;
            }
            case "DROPOUT": {
                double p = node.hyperDouble("p", 0.5);
                DropoutImpl d = new DropoutImpl(p);
                recordLayer(layers, path, "DROPOUT", Map.of("p", p));
                return d;
            }
            case "IDENTITY":
                recordLayer(layers, path, "IDENTITY", Map.of());
                return new IdentityImpl();
            case "PARAMETER":
            case "PARAMETER_BAG":
                // bare parameter holder
                recordLayer(layers, path, "PARAMETER", Map.of());
                return new Module(path);
            default:
                // Unknown kind → named composite container (still preserves children)
                recordLayer(layers, path, "COMPOSITE", Map.of("class_name",
                        node.className != null ? node.className : kind));
                return new Module(node.className != null ? node.className : kind.toLowerCase(Locale.ROOT));
        }
    }

    private static void recordLayer(List<StateDictModuleBuilder.LayerInfo> layers,
                                    String path, String kind, Map<String, Object> hyper) {
        if (layers == null) return;
        try {
            StateDictModuleBuilder.LayerKind lk;
            try {
                lk = StateDictModuleBuilder.LayerKind.valueOf(kind);
            } catch (Exception e) {
                lk = StateDictModuleBuilder.LayerKind.CONTAINER;
            }
            layers.add(new StateDictModuleBuilder.LayerInfo(
                    path, lk, Map.of(), hyper, List.of()));
        } catch (Throwable ignored) {}
    }

    // ---- weight binding -------------------------------------------------------

    private static int bindWeights(StructureSpec spec,
                                   Map<String, Tensor> stateDict,
                                   Map<String, Module> nodes,
                                   Map<String, Tensor> owned,
                                   boolean requiresGrad,
                                   boolean strict) {
        int bound = 0;
        for (Map.Entry<String, Tensor> e : stateDict.entrySet()) {
            String key = e.getKey();
            Tensor src = e.getValue();
            if (src == null || !src.defined()) continue;

            // key = path.role  e.g. user_tower.mlp.0.weight
            int lastDot = key.lastIndexOf('.');
            if (lastDot < 0) {
                // top-level parameter on root
                registerRaw(nodes.get(""), key, src, requiresGrad, isBufferRole(key), owned);
                bound++;
                continue;
            }
            String path = key.substring(0, lastDot);
            String role = key.substring(lastDot + 1);
            Module leaf = nodes.get(path);
            if (leaf == null) {
                // try parent for nested role like mlp.1.bn.running_mean where bn is child
                // already handled by full path; if missing, raw-register on nearest parent
                Module nearest = findNearest(nodes, path);
                if (nearest != null) {
                    String rel = key.substring(nearestPath(nodes, path).length());
                    if (rel.startsWith(".")) rel = rel.substring(1);
                    registerRaw(nearest, rel.contains(".") ? role : rel, src,
                            requiresGrad && !isBufferRole(role), isBufferRole(role), owned);
                    // store under full key
                    owned.put(key, owned.getOrDefault(key, src));
                    bound++;
                } else if (strict) {
                    throw new IllegalStateException("no module for state_dict key: " + key);
                } else {
                    // register on root as bag parameter with full key segments
                    registerRawDotted(nodes.get(""), key, src, requiresGrad, isBufferRole(role), owned);
                    bound++;
                }
                continue;
            }

            boolean buffer = isBufferRole(role);
            if (copyRoleIntoLeaf(leaf, role, src, requiresGrad && !buffer)) {
                Tensor live = lookupRole(leaf, role);
                owned.put(key, live != null ? live : retain(src));
                bound++;
            } else {
                // fallback register_parameter/buffer on the leaf module
                registerRaw(leaf, role, src, requiresGrad && !buffer, buffer, owned);
                owned.put(key, owned.containsKey(key) ? owned.get(key) : retain(src));
                bound++;
            }
        }
        return bound;
    }

    private static boolean copyRoleIntoLeaf(Module leaf, String role, Tensor src, boolean requiresGrad) {
        if (leaf == null || role == null || src == null) return false;
        try {
            Tensor dest = lookupRole(leaf, role);
            if (dest != null && dest.defined()) {
                try (NoGradGuard g = new NoGradGuard()) {
                    dest.copy_(src.detach().contiguous());
                }
                if (!isBufferRole(role)) {
                    try { dest.requires_grad_(requiresGrad); } catch (Throwable ignored) {}
                }
                return true;
            }
        } catch (Throwable ignored) {}
        return false;
    }

    private static Tensor lookupRole(Module leaf, String role) {
        if (leaf == null || role == null) return null;
        try {
            Module typed = leaf;
            try {
                Module r = ModuleAsHelper.recover(leaf);
                if (r != null) typed = r;
            } catch (Throwable ignored) {}

            if (typed instanceof LinearImpl) {
                LinearImpl lin = (LinearImpl) typed;
                if ("weight".equals(role)) return retain(lin.weight());
                if ("bias".equals(role)) return retain(lin.bias());
            } else if (typed instanceof EmbeddingImpl) {
                if ("weight".equals(role)) return retain(((EmbeddingImpl) typed).weight());
            } else if (typed instanceof LayerNormImpl) {
                LayerNormImpl ln = (LayerNormImpl) typed;
                if ("weight".equals(role)) return retain(ln.weight());
                if ("bias".equals(role)) return retain(ln.bias());
            } else if (typed instanceof BatchNorm1dImpl) {
                BatchNorm1dImpl m = (BatchNorm1dImpl) typed;
                if ("weight".equals(role)) return retain(m.weight());
                if ("bias".equals(role)) return retain(m.bias());
                if ("running_mean".equals(role)) return retain(m.running_mean());
                if ("running_var".equals(role)) return retain(m.running_var());
                if ("num_batches_tracked".equals(role)) return retain(m.num_batches_tracked());
            } else if (typed instanceof BatchNorm2dImpl) {
                BatchNorm2dImpl m = (BatchNorm2dImpl) typed;
                if ("weight".equals(role)) return retain(m.weight());
                if ("bias".equals(role)) return retain(m.bias());
                if ("running_mean".equals(role)) return retain(m.running_mean());
                if ("running_var".equals(role)) return retain(m.running_var());
                if ("num_batches_tracked".equals(role)) return retain(m.num_batches_tracked());
            } else if (typed instanceof BatchNorm3dImpl) {
                BatchNorm3dImpl m = (BatchNorm3dImpl) typed;
                if ("weight".equals(role)) return retain(m.weight());
                if ("bias".equals(role)) return retain(m.bias());
                if ("running_mean".equals(role)) return retain(m.running_mean());
                if ("running_var".equals(role)) return retain(m.running_var());
                if ("num_batches_tracked".equals(role)) return retain(m.num_batches_tracked());
            } else if (typed instanceof GroupNormImpl) {
                GroupNormImpl m = (GroupNormImpl) typed;
                if ("weight".equals(role)) return retain(m.weight());
                if ("bias".equals(role)) return retain(m.bias());
            } else if (typed instanceof Conv1dImpl) {
                Conv1dImpl m = (Conv1dImpl) typed;
                if ("weight".equals(role)) return retain(m.weight());
                if ("bias".equals(role)) return retain(m.bias());
            } else if (typed instanceof Conv2dImpl) {
                Conv2dImpl m = (Conv2dImpl) typed;
                if ("weight".equals(role)) return retain(m.weight());
                if ("bias".equals(role)) return retain(m.bias());
            } else if (typed instanceof Conv3dImpl) {
                Conv3dImpl m = (Conv3dImpl) typed;
                if ("weight".equals(role)) return retain(m.weight());
                if ("bias".equals(role)) return retain(m.bias());
            }
        } catch (Throwable ignored) {}
        return null;
    }

    private static void registerRaw(Module target, String name, Tensor src,
                                    boolean requiresGrad, boolean buffer,
                                    Map<String, Tensor> owned) {
        if (target == null || name == null || src == null) return;
        // only leaf name segment
        String leaf = name;
        int d = name.lastIndexOf('.');
        if (d >= 0) leaf = name.substring(d + 1);
        Tensor ownedT = src.detach().clone().contiguous();
        try {
            if (buffer) {
                ownedT.requires_grad_(false);
                target.register_buffer(leaf, ownedT);
            } else {
                ownedT.requires_grad_(requiresGrad);
                target.register_parameter(leaf, ownedT, requiresGrad);
            }
        } catch (Throwable ignored) {}
        // caller stores under full key
        owned.put(name, ownedT);
    }

    private static void registerRawDotted(Module root, String fullKey, Tensor src,
                                          boolean requiresGrad, boolean buffer,
                                          Map<String, Tensor> owned) {
        if (root == null || fullKey == null) return;
        String[] parts = fullKey.split("\\.");
        Module cur = root;
        for (int i = 0; i < parts.length - 1; i++) {
            String seg = parts[i];
            Module next = null;
            try {
                // try named_children lookup is heavy; create on demand
                next = new Module(seg);
                try {
                    cur.register_module(seg, next);
                } catch (Throwable t) {
                    // already exists — try to recover from children later; use new as best effort
                }
            } catch (Throwable ignored) {}
            if (next != null) cur = next;
        }
        registerRaw(cur, parts[parts.length - 1], src, requiresGrad, buffer, owned);
        owned.put(fullKey, owned.getOrDefault(parts[parts.length - 1], retain(src)));
    }

    private static Module findNearest(Map<String, Module> nodes, String path) {
        String p = path;
        while (p != null) {
            Module m = nodes.get(p);
            if (m != null) return m;
            int d = p.lastIndexOf('.');
            if (d < 0) return nodes.get("");
            p = p.substring(0, d);
        }
        return nodes.get("");
    }

    private static String nearestPath(Map<String, Module> nodes, String path) {
        String p = path;
        while (p != null) {
            if (nodes.containsKey(p)) return p;
            int d = p.lastIndexOf('.');
            if (d < 0) return "";
            p = p.substring(0, d);
        }
        return "";
    }

    private static boolean isBufferRole(String role) {
        if (role == null) return false;
        String r = role.toLowerCase(Locale.ROOT);
        return r.equals("running_mean") || r.equals("running_var")
                || r.equals("num_batches_tracked")
                || r.endsWith("_mean") || r.endsWith("_var");
    }

    private static Tensor retain(Tensor byRef) {
        if (byRef == null) return null;
        try {
            if (byRef.isNull() || !byRef.defined()) return null;
            return new Tensor(byRef);
        } catch (Throwable t) {
            return byRef;
        }
    }
}
