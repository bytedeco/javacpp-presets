package org.bytedeco.pytorch.plot.vista;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.ModuleAsHelper;
import org.bytedeco.pytorch.nn.modules.container.ModuleDictImpl;
import org.bytedeco.pytorch.nn.modules.container.ModuleListImpl;
import org.bytedeco.pytorch.nn.modules.container.ParameterDictImpl;
import org.bytedeco.pytorch.nn.modules.container.ParameterListImpl;
import org.bytedeco.pytorch.nn.modules.container.SequentialImpl;

/**
 * Identifies which modules are treated as atomic leaves vs transparent
 * containers during tracing.
 *
 * <p>In torchvista, {@code MODULES} is the set of all discovered {@code nn.Module}
 * subclasses minus {@code CONTAINER_MODULES}. Leaves are traced as single nodes
 * (their internal ops are hidden unless {@code forced_module_tracing_depth}
 * forces expansion). Containers ({@code Sequential}, {@code ModuleList}, …)
 * never form a leaf node themselves — their children are walked instead.
 *
 * <p>JavaCPP cannot dynamically scan torchvision/torchaudio packages the way
 * Python does. Instead we treat:
 * <ul>
 *   <li>Known containers → always transparent (not a leaf).</li>
 *   <li>Any module whose {@code named_children()} is empty → leaf.</li>
 *   <li>Any module with children that is <em>not</em> a known container →
 *       treated as a user-defined composite; default behaviour matches
 *       torchvista with {@code forced_module_tracing_depth == null}: the
 *       composite itself is a leaf (black box) unless forced depth expands it.</li>
 * </ul>
 */
public final class ModuleDiscovery {
    private ModuleDiscovery() {}

    private static final Set<Class<?>> CONTAINER_CLASSES;
    static {
        Set<Class<?>> s = new HashSet<>();
        s.add(Module.class);
        s.add(SequentialImpl.class);
        s.add(ModuleListImpl.class);
        s.add(ModuleDictImpl.class);
        s.add(ParameterListImpl.class);
        s.add(ParameterDictImpl.class);
        CONTAINER_CLASSES = Collections.unmodifiableSet(s);
    }

    public static Set<Class<?>> containerClasses() {
        return CONTAINER_CLASSES;
    }

    public static Module recover(Module m) {
        if (m == null || m.isNull()) return m;
        try {
            Module r = ModuleAsHelper.recover(m);
            return r != null ? r : m;
        } catch (Throwable e) {
            return m;
        }
    }

    /**
     * Re-type a possibly bare {@link Module} (e.g. from {@code named_children()})
     * to its concrete {@code *Impl} Java peer so {@code forward()} hits the real
     * C++ implementation instead of {@code Module.forward_tensor} (which throws
     * for built-ins).
     *
     * <p>Order: registry {@link ModuleAsHelper#recover} → already-concrete class
     * → {@code asXxx()} from demangled C++ name → reflective as-walk → original.
     */
    public static Module concrete(Module m) {
        if (m == null || m.isNull()) return m;
        Module recovered = recover(m);
        if (recovered.getClass() != Module.class) {
            return recovered;
        }
        // Try asXxx from demangled / simple name: LinearImpl → asLinear()
        String simple = simpleTypeName(recovered);
        if (simple.endsWith("Impl")) {
            simple = simple.substring(0, simple.length() - "Impl".length());
        }
        if (!simple.isEmpty() && !simple.equals("Module")) {
            try {
                java.lang.reflect.Method asMethod = Module.class.getMethod("as" + simple);
                Object cast = asMethod.invoke(recovered);
                if (cast instanceof Module) {
                    Module typed = (Module) cast;
                    if (!typed.isNull()) {
                        ModuleAsHelper.remember(typed);
                        return typed;
                    }
                }
            } catch (Throwable ignored) {}
        }
        // Slow path: walk every asXxx() helper (same idea as ModuleAsHelper.findUnderlyingType)
        try {
            for (java.lang.reflect.Method method : Module.class.getMethods()) {
                String name = method.getName();
                if (!name.startsWith("as") || name.equals("as") || method.getParameterCount() != 0) {
                    continue;
                }
                Class<?> ret = method.getReturnType();
                if (!Module.class.isAssignableFrom(ret) || ret == Module.class) continue;
                try {
                    Object cast = method.invoke(recovered);
                    if (cast instanceof Module) {
                        Module typed = (Module) cast;
                        if (!typed.isNull()) {
                            ModuleAsHelper.remember(typed);
                            return typed;
                        }
                    }
                } catch (Throwable ignored) {}
            }
        } catch (Throwable ignored) {}
        return recovered;
    }

    public static boolean isContainer(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = recover(m);
        Class<?> c = typed.getClass();
        if (CONTAINER_CLASSES.contains(c)) return true;
        // asXxx helpers — pointer may be bare Module wrapping SequentialImpl etc.
        try {
            if (typed.asSequential() != null && !typed.asSequential().isNull()) return true;
        } catch (Throwable ignored) {}
        try {
            if (typed.asModuleList() != null && !typed.asModuleList().isNull()) return true;
        } catch (Throwable ignored) {}
        try {
            if (typed.asModuleDict() != null && !typed.asModuleDict().isNull()) return true;
        } catch (Throwable ignored) {}
        try {
            if (typed.asParameterList() != null && !typed.asParameterList().isNull()) return true;
        } catch (Throwable ignored) {}
        try {
            if (typed.asParameterDict() != null && !typed.asParameterDict().isNull()) return true;
        } catch (Throwable ignored) {}
        // Name-based fallback (demangled C++ name)
        String name = typeName(typed);
        return name.contains("Sequential")
                || name.contains("ModuleList")
                || name.contains("ModuleDict")
                || name.contains("ParameterList")
                || name.contains("ParameterDict");
    }

    /**
     * Modules without a real {@code forward} (ModuleList / ModuleDict /
     * ParameterList / ParameterDict) — torchvista still walks their children
     * but does not count them toward tracing depth.
     */
    public static boolean hasForwardMethod(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = recover(m);
        if (isModuleListLike(typed) || isModuleDictLike(typed)
                || isParameterListLike(typed) || isParameterDictLike(typed)) {
            return false;
        }
        // Sequential and every leaf *Impl have forward.
        return true;
    }

    public static boolean isSequential(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = recover(m);
        if (typed instanceof SequentialImpl) return true;
        try {
            SequentialImpl s = typed.asSequential();
            return s != null && !s.isNull();
        } catch (Throwable e) {
            return typeName(typed).contains("Sequential");
        }
    }

    public static boolean isModuleListLike(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = recover(m);
        if (typed instanceof ModuleListImpl) return true;
        try {
            ModuleListImpl s = typed.asModuleList();
            return s != null && !s.isNull();
        } catch (Throwable e) {
            return typeName(typed).contains("ModuleList");
        }
    }

    public static boolean isModuleDictLike(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = recover(m);
        if (typed instanceof ModuleDictImpl) return true;
        try {
            ModuleDictImpl s = typed.asModuleDict();
            return s != null && !s.isNull();
        } catch (Throwable e) {
            return typeName(typed).contains("ModuleDict");
        }
    }

    public static boolean isParameterListLike(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = recover(m);
        if (typed instanceof ParameterListImpl) return true;
        try {
            ParameterListImpl s = typed.asParameterList();
            return s != null && !s.isNull();
        } catch (Throwable e) {
            return typeName(typed).contains("ParameterList");
        }
    }

    public static boolean isParameterDictLike(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = recover(m);
        if (typed instanceof ParameterDictImpl) return true;
        try {
            ParameterDictImpl s = typed.asParameterDict();
            return s != null && !s.isNull();
        } catch (Throwable e) {
            return typeName(typed).contains("ParameterDict");
        }
    }

    /**
     * Recommend basic layers ({@code EmbeddingLayer}, {@code PredictionLayer}, …)
     * that should render as atomic leaves — except single-Sequential wrappers
     * like {@code MLP} which {@link #canChainChildrenAsSequential} expands.
     */
    public static boolean isLibraryLayerLeaf(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = concrete(m);
        if (canChainChildrenAsSequential(typed)) return false;
        try {
            Package p = typed.getClass().getPackage();
            if (p == null || p.getName() == null) return false;
            String name = p.getName();
            return name.startsWith("org.bytedeco.pytorch.utils.recommend.basic.layers")
                    || name.startsWith("org.bytedeco.pytorch.utils.recommend.basic.features");
        } catch (Throwable e) {
            return false;
        }
    }

    /**
     * Built-in libtorch {@code *Impl} layer (LinearImpl, ReLUImpl, …) — treated
     * as an atomic leaf. Detected by concrete class name ending in {@code Impl}
     * under {@code org.bytedeco.pytorch.nn.modules} (not a user subclass).
     */
    public static boolean isBuiltinLeaf(Module m) {
        if (m == null || m.isNull()) return false;
        if (isLibraryLayerLeaf(m)) return true;
        Module typed = concrete(m);
        if (isContainer(typed)) return false;
        Class<?> c = typed.getClass();
        // User-defined Java Module subclass with its own forward → not builtin
        if (ModuleAsHelper.hasForwardOverride(typed, org.bytedeco.pytorch.Tensor.class)
                || ModuleAsHelper.hasForwardOverride(typed,
                        org.bytedeco.pytorch.Tensor.class, org.bytedeco.pytorch.Tensor.class)) {
            // Could still be a thin Java wrapper; prefer package check
            Package p = c.getPackage();
            String pkg = p == null ? "" : p.getName();
            if (!pkg.startsWith("org.bytedeco.pytorch.nn.modules")
                    && !pkg.equals("org.bytedeco.pytorch.nn")) {
                return false;
            }
        }
        String simple = c.getSimpleName();
        if (simple.endsWith("Impl") && (
                pkgStartsWith(c, "org.bytedeco.pytorch.nn.modules")
                        || simple.equals("LinearImpl") || simple.equals("ReLUImpl")
                        || simple.equals("Conv2dImpl") || simple.equals("DropoutImpl")
                        || simple.equals("LayerNormImpl") || simple.equals("EmbeddingImpl")
                        || simple.equals("BatchNorm2dImpl") || simple.equals("GELUImpl")
                        || simple.equals("SiLUImpl") || simple.equals("SoftmaxImpl"))) {
            return true;
        }
        // asXxx recovered types live in nn.modules
        return pkgStartsWith(c, "org.bytedeco.pytorch.nn.modules")
                && simple.endsWith("Impl")
                && !simple.contains("Sequential")
                && !simple.contains("ModuleList")
                && !simple.contains("ModuleDict");
    }

    private static boolean pkgStartsWith(Class<?> c, String prefix) {
        try {
            Package p = c.getPackage();
            return p != null && p.getName() != null && p.getName().startsWith(prefix);
        } catch (Throwable e) {
            return false;
        }
    }

    /**
     * True when a custom module has a single chainable child (typically
     * {@code Sequential}) — e.g. recommend {@code MLP} which only wraps
     * {@code sequential}. Expanding children then exposes Linear/ReLU nodes
     * with real tensor shapes instead of one opaque custom leaf.
     */
    public static boolean canChainChildrenAsSequential(Module m) {
        if (m == null || m.isNull()) return false;
        if (isSequential(m)) return true;
        java.util.List<ModuleChildren.NamedChild> kids = ModuleChildren.list(m);
        if (kids.size() != 1) return false;
        Module only = kids.get(0).module;
        return isSequential(only)
                || (ModuleChildren.hasChildren(only) && !isBuiltinLeaf(only));
    }

    /**
     * User-defined Java {@code Module} with an overridden {@code forward}
     * (Tensor, multi-Tensor, {@code Map}, or {@code List}) — torchvista keeps
     * these open so free ops / child modules inside become graph nodes.
     */
    public static boolean isCustomForwardModule(Module m) {
        if (m == null || m.isNull()) return false;
        Module typed = concrete(m);
        if (isBuiltinLeaf(typed) || isSequential(typed)) return false;
        if (isContainer(typed) && !isSequential(typed)) return false;
        if (ModuleAsHelper.hasForwardOverride(typed, org.bytedeco.pytorch.Tensor.class)
                || ModuleAsHelper.hasForwardOverride(typed,
                        org.bytedeco.pytorch.Tensor.class, org.bytedeco.pytorch.Tensor.class)
                || ModuleAsHelper.hasForwardOverride(typed,
                        org.bytedeco.pytorch.Tensor.class, org.bytedeco.pytorch.Tensor.class,
                        org.bytedeco.pytorch.Tensor.class)) {
            return true;
        }
        // recommend multi_task models: forward(Map<String,Tensor>) / forward(List)
        return findForwardMethod(typed) != null;
    }

    /**
     * Find a public instance {@code forward(...)} declared on the concrete Java
     * class (not {@link Module} itself). Used for Map/List multi-task forwards.
     * Prefers a method whose first parameter is compatible with {@code hint}
     * when non-null.
     */
    public static java.lang.reflect.Method findForwardMethod(Module m) {
        return findForwardMethod(m, null);
    }

    public static java.lang.reflect.Method findForwardMethod(Module m, Object hintArg) {
        if (m == null) return null;
        Class<?> c = m.getClass();
        if (c == Module.class) return null;
        java.lang.reflect.Method best = null;
        int bestScore = -1;

        // Normalize multi-arg hints: Tensor[] / List / Object[] → arity + first type
        int hintArity = -1;
        Class<?> hint0 = null;
        Object[] hintArr = null;
        if (hintArg instanceof Tensor[]) {
            hintArr = (Object[]) hintArg;
            hintArity = hintArr.length;
            if (hintArity > 0) hint0 = Tensor.class;
        } else if (hintArg instanceof Object[]) {
            hintArr = (Object[]) hintArg;
            hintArity = hintArr.length;
            if (hintArity > 0 && hintArr[0] != null) hint0 = hintArr[0].getClass();
        } else if (hintArg instanceof java.util.List) {
            java.util.List<?> list = (java.util.List<?>) hintArg;
            hintArity = list.size();
            if (hintArity > 0 && list.get(0) != null) hint0 = list.get(0).getClass();
            // List of tensors used as multi-arg: treat like Tensor[]
            if (hintArity > 0 && list.get(0) instanceof Tensor) hint0 = Tensor.class;
        } else if (hintArg != null) {
            hintArity = 1;
            hint0 = hintArg.getClass();
        }

        for (java.lang.reflect.Method method : c.getMethods()) {
            if (!"forward".equals(method.getName())) continue;
            if (method.getDeclaringClass() == Module.class) continue;
            if (method.getParameterCount() < 1) continue;
            Class<?>[] pts = method.getParameterTypes();
            int score = 10 - pts.length; // prefer fewer args as baseline

            if (hintArg != null) {
                Class<?> p0 = pts[0];
                if (p0.isInstance(hintArg)) {
                    score += 100; // exact single-arg match (Map, List, Tensor)
                } else if (hintArg instanceof java.util.Map && java.util.Map.class.isAssignableFrom(p0)) {
                    score += 90;
                } else if (hintArg instanceof java.util.List && java.util.List.class.isAssignableFrom(p0)
                        && !(hint0 == Tensor.class && pts.length > 1)) {
                    // Prefer List-as-single-arg only when method is not multi-Tensor
                    score += 80;
                } else if (hintArg instanceof Tensor && Tensor.class.isAssignableFrom(p0)) {
                    score += 80;
                } else if (hintArr != null || (hintArg instanceof java.util.List && hint0 == Tensor.class)) {
                    // Multi-arg Tensor[] / List<Tensor> / Object[]
                    int match = 0;
                    int n = Math.min(pts.length, hintArity);
                    for (int i = 0; i < n; i++) {
                        Object hi;
                        if (hintArr != null) hi = i < hintArr.length ? hintArr[i] : null;
                        else hi = ((java.util.List<?>) hintArg).get(i);
                        if (hi != null && pts[i].isInstance(hi)) match++;
                        else if (hi instanceof Tensor && Tensor.class.isAssignableFrom(pts[i])) match++;
                        else if (hi instanceof java.util.Map && java.util.Map.class.isAssignableFrom(pts[i])) match++;
                    }
                    // Prefer exact arity match heavily (MetaHeac Map+Tensor, HLLM Tensor+Tensor)
                    if (match == pts.length && hintArity >= pts.length) score += 120 + match * 10;
                    else if (match > 0 && pts.length == hintArity) score += 60 + match * 10;
                    else if (match > 0) score += match * 15;
                    else score -= 40;
                } else {
                    score -= 50;
                }
            }
            if (score > bestScore) {
                bestScore = score;
                best = method;
            }
        }
        return best;
    }

    /**
     * Should this module be recorded as a single leaf node (not expanded)?
     *
     * <p>Matches torchvista:
     * <ul>
     *   <li>Built-in {@code *Impl} → leaf (internals hidden).</li>
     *   <li>Custom Java {@code Module.forward} → <em>not</em> leaf (open frame
     *       so {@link VistaOps} free ops become visible).</li>
     *   <li>Sequential → not leaf (children chained).</li>
     *   <li>{@code forcedDepth != null}: leaf when stack depth ≥ forcedDepth
     *       (except Sequential which still expands when depth allows).</li>
     * </ul>
     */
    public static boolean isTracedLeaf(Module m, int stackDepth, Integer forcedDepth) {
        if (m == null || m.isNull()) return false;
        if (isContainer(m) && !isSequential(m)) {
            // ModuleList/Dict never form a leaf node themselves
            return false;
        }
        if (isSequential(m)) return false;
        if (forcedDepth != null) {
            // At/above forced depth everything becomes a leaf, including custom
            return stackDepth >= forcedDepth;
        }
        if (isCustomForwardModule(m)) return false;
        // Built-in or unknown bare Module → leaf
        return true;
    }

    public static String typeName(Module m) {
        if (m == null || m.isNull()) return "null";
        Module typed = recover(m);
        try {
            org.bytedeco.javacpp.BytePointer bp = typed.name();
            if (bp != null && !bp.isNull()) {
                String raw = bp.getString();
                if (raw != null && !raw.isEmpty()) {
                    return demangle(raw);
                }
            }
        } catch (Throwable ignored) {}
        try {
            return typed.getClass().getSimpleName();
        } catch (Throwable e) {
            return "Module";
        }
    }

    public static String simpleTypeName(Module m) {
        String full = typeName(m);
        int cc = full.lastIndexOf("::");
        if (cc >= 0) return full.substring(cc + 2);
        int dot = full.lastIndexOf('.');
        if (dot >= 0) return full.substring(dot + 1);
        return full;
    }

    /** Demangle JavaCPP-encoded names like {@code JavaCPP_torch_0003a_...}. */
    public static String demangle(String raw) {
        if (raw == null) return "";
        if (!raw.startsWith("JavaCPP_")) return raw;
        String body = raw.substring("JavaCPP_".length());
        StringBuilder sb = new StringBuilder(body.length());
        for (int i = 0; i < body.length(); ) {
            if (i + 5 <= body.length() && body.charAt(i) == '_'
                    && Character.isDigit(body.charAt(i + 1))) {
                // _0003a → decode 4 hex digits
                try {
                    int code = Integer.parseInt(body.substring(i + 1, i + 5), 16);
                    sb.append((char) code);
                    i += 5;
                    continue;
                } catch (NumberFormatException ignored) {}
            }
            sb.append(body.charAt(i));
            i++;
        }
        return sb.toString();
    }
}
