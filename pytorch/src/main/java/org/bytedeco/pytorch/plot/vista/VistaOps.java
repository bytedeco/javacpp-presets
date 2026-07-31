package org.bytedeco.pytorch.plot.vista;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.nn.Module;

/**
 * Optional free-function / nested-module wrappers for vista.
 *
 * <p><b>Not required for model visualization.</b> {@link VistaEngine} expands
 * {@code named_children} and runs child modules non-invasively — library models
 * (recommend multi_task, etc.) must <em>not</em> be edited to call these APIs.
 *
 * <p>Use only in <em>your own</em> application code when you want free ops
 * ({@code relu}/{@code add}/…) or nested {@code module(child, x)} calls to
 * appear as graph nodes. When no engine is bound, every method falls through
 * to the raw {@link torch} / {@link Module#forward} path (one ThreadLocal read).
 *
 * <pre>
 *   // Optional — only in user-owned Module.forward, never in library models:
 *   h = VistaOps.relu(h);
 *   h = VistaOps.add(h, x);
 *   h = VistaOps.module(child, h);
 * </pre>
 *
 * <p>{@link VistaEngine} binds/unbinds itself around a trace via
 * {@link #bind(VistaEngine)} / {@link #unbind(VistaEngine)}.
 */
public final class VistaOps {
    private VistaOps() {}

    private static final ThreadLocal<VistaEngine> BOUND = new ThreadLocal<>();

    /** Bind engine for this thread (called by {@link VistaEngine}). */
    public static void bind(VistaEngine engine) {
        BOUND.set(engine);
    }

    /** Clear binding if it still points at {@code engine}. */
    public static void unbind(VistaEngine engine) {
        if (BOUND.get() == engine) {
            BOUND.remove();
        }
    }

    public static VistaEngine current() {
        return BOUND.get();
    }

    public static boolean isTracing() {
        return BOUND.get() != null;
    }

    // =========================================================================
    // Unary activations
    // =========================================================================

    public static Tensor relu(Tensor self) {
        return traceUnary("relu", "torch", self, () -> torch.relu(self));
    }

    public static Tensor gelu(Tensor self) {
        return traceUnary("gelu", "torch", self, () -> torch.gelu(self));
    }

    public static Tensor silu(Tensor self) {
        return traceUnary("silu", "torch", self, () -> torch.silu(self));
    }

    public static Tensor sigmoid(Tensor self) {
        return traceUnary("sigmoid", "torch", self, () -> torch.sigmoid(self));
    }

    public static Tensor tanh(Tensor self) {
        return traceUnary("tanh", "torch", self, () -> torch.tanh(self));
    }

    public static Tensor softmax(Tensor self, long dim) {
        return trace1("softmax", "torch", self, () -> torch.softmax(self, dim), dim);
    }

    public static Tensor log_softmax(Tensor self, long dim) {
        return trace1("log_softmax", "torch", self, () -> torch.log_softmax(self, dim), dim);
    }

    public static Tensor dropout(Tensor input, double p, boolean train) {
        return trace1("dropout", "torch", input, () -> torch.dropout(input, p, train), p, train);
    }

    // =========================================================================
    // Binary arithmetic
    // =========================================================================

    public static Tensor add(Tensor self, Tensor other) {
        return traceBinary("add", "torch", self, other, () -> torch.add(self, other));
    }

    public static Tensor add(Tensor self, Tensor other, Scalar alpha) {
        return traceBinary("add", "torch", self, other, () -> torch.add(self, other, alpha));
    }

    public static Tensor sub(Tensor self, Tensor other) {
        return traceBinary("sub", "torch", self, other, () -> torch.sub(self, other));
    }

    public static Tensor mul(Tensor self, Tensor other) {
        return traceBinary("mul", "torch", self, other, () -> torch.mul(self, other));
    }

    public static Tensor div(Tensor self, Tensor other) {
        return traceBinary("div", "torch", self, other, () -> torch.div(self, other));
    }

    public static Tensor matmul(Tensor self, Tensor other) {
        return traceBinary("matmul", "torch", self, other, () -> torch.matmul(self, other));
    }

    public static Tensor bmm(Tensor self, Tensor other) {
        return traceBinary("bmm", "torch", self, other, () -> torch.bmm(self, other));
    }

    // =========================================================================
    // Shape / indexing helpers
    // =========================================================================

    public static Tensor cat(TensorVector tensors, long dim) {
        VistaEngine eng = BOUND.get();
        if (eng == null) return torch.cat(tensors, dim);
        Object[] args = new Object[]{tensors, dim};
        return eng.traceFreeOp("cat", "torch", args, () -> torch.cat(tensors, dim));
    }

    public static Tensor cat(TensorVector tensors) {
        return cat(tensors, 0L);
    }

    public static Tensor flatten(Tensor self, long startDim, long endDim) {
        return trace1("flatten", "torch", self, () -> torch.flatten(self, startDim, endDim),
                startDim, endDim);
    }

    public static Tensor reshape(Tensor self, long... shape) {
        return trace1("reshape", "torch", self, () -> self.reshape(shape), (Object) shape);
    }

    public static Tensor view(Tensor self, long... shape) {
        // view is a Tensor method — keep as free-op label
        return trace1("view", "torch.Tensor", self, () -> self.view(shape), (Object) shape);
    }

    public static Tensor transpose(Tensor self, long dim0, long dim1) {
        return trace1("transpose", "torch", self, () -> torch.transpose(self, dim0, dim1), dim0, dim1);
    }

    public static Tensor permute(Tensor self, long... dims) {
        return trace1("permute", "torch", self, () -> self.permute(dims), (Object) dims);
    }

    public static Tensor layer_norm(Tensor input, long[] normalizedShape) {
        return trace1("layer_norm", "torch", input,
                () -> torch.layer_norm(input, normalizedShape), (Object) normalizedShape);
    }

    // =========================================================================
    // Nested module calls (from inside a custom Module.forward)
    // =========================================================================

    /**
     * Run a child module through the active {@link VistaEngine} so it appears
     * as a Module / Sequential node in the graph.
     *
     * <p>Inside a custom {@code Module.forward}, prefer:
     * <pre>
     *   Tensor h = VistaOps.module(this.fc, x);   // traced
     *   // not: this.fc.forward(x);               // invisible to vista
     * </pre>
     * When no engine is bound, falls through to {@code child.forward(input)}.
     */
    public static Tensor module(Module child, Tensor input) {
        VistaEngine eng = BOUND.get();
        if (eng == null) {
            Module m = ModuleDiscovery.concrete(child);
            return m.forward(input);
        }
        return eng.traceNestedModule(child, input);
    }

    public static Tensor module(Module child, Tensor input1, Tensor input2) {
        VistaEngine eng = BOUND.get();
        if (eng == null) {
            Module m = ModuleDiscovery.concrete(child);
            return m.forward(input1, input2);
        }
        return eng.traceNestedModule(child, new Tensor[]{input1, input2});
    }

    public static Tensor module(Module child, Tensor[] inputs) {
        VistaEngine eng = BOUND.get();
        if (eng == null) {
            Module m = ModuleDiscovery.concrete(child);
            if (inputs == null || inputs.length == 0) {
                throw new IllegalArgumentException("no inputs");
            }
            if (inputs.length == 1) return m.forward(inputs[0]);
            if (inputs.length == 2) return m.forward(inputs[0], inputs[1]);
            if (inputs.length == 3) return m.forward(inputs[0], inputs[1], inputs[2]);
            return m.forward(inputs[0]);
        }
        return eng.traceNestedModule(child, inputs);
    }

    /**
     * Nested module call with a feature map (recommend multi_task models:
     * {@code EmbeddingLayer}, {@code SharedBottom}, …).
     */
    @SuppressWarnings("rawtypes")
    public static Tensor module(Module child, java.util.Map inputMap) {
        VistaEngine eng = BOUND.get();
        if (eng == null) {
            return invokeMapForward(child, inputMap);
        }
        return eng.traceNestedModule(child, inputMap);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static Tensor invokeMapForward(Module child, java.util.Map inputMap) {
        Module m = ModuleDiscovery.concrete(child);
        java.lang.reflect.Method method = ModuleDiscovery.findForwardMethod(m);
        if (method != null) {
            try {
                Class<?>[] pts = method.getParameterTypes();
                if (pts.length >= 1 && java.util.Map.class.isAssignableFrom(pts[0])) {
                    Object result;
                    if (pts.length == 1) {
                        result = method.invoke(m, inputMap);
                    } else if (pts.length == 2 && java.util.Map.class.isAssignableFrom(pts[1])) {
                        result = method.invoke(m, inputMap, java.util.Collections.emptyMap());
                    } else if (pts.length == 3
                            && java.util.Map.class.isAssignableFrom(pts[1])
                            && (pts[2] == boolean.class || pts[2] == Boolean.class)) {
                        result = method.invoke(m, inputMap, java.util.Collections.emptyMap(), true);
                    } else {
                        result = method.invoke(m, inputMap);
                    }
                    if (result instanceof Tensor) return (Tensor) result;
                    java.util.List<Tensor> ts = TensorUtils.extractTensors(result);
                    if (!ts.isEmpty()) return ts.get(0);
                }
            } catch (Throwable t) {
                throw new RuntimeException("Map forward failed on "
                        + ModuleDiscovery.typeName(m) + ": " + t.getMessage(), t);
            }
        }
        throw new IllegalArgumentException("No Map forward on " + ModuleDiscovery.typeName(m));
    }

    // =========================================================================
    // Generic entry (advanced / custom ops)
    // =========================================================================

    /**
     * Trace an arbitrary free op. Prefer the typed helpers above; use this for
     * ops not yet wrapped.
     */
    public static Tensor trace(String opName, String namespace, Object[] inputs, TensorSupplier body) {
        VistaEngine eng = BOUND.get();
        if (eng == null) return body.get();
        return eng.traceFreeOp(opName, namespace == null ? "torch" : namespace, inputs, body);
    }

    @FunctionalInterface
    public interface TensorSupplier {
        Tensor get();
    }

    // ---- internals ----------------------------------------------------------

    private static Tensor traceUnary(String name, String ns, Tensor self, TensorSupplier body) {
        VistaEngine eng = BOUND.get();
        if (eng == null) return body.get();
        return eng.traceFreeOp(name, ns, new Object[]{self}, body);
    }

    private static Tensor trace1(String name, String ns, Tensor self, TensorSupplier body, Object... extra) {
        VistaEngine eng = BOUND.get();
        if (eng == null) return body.get();
        Object[] args = new Object[1 + (extra == null ? 0 : extra.length)];
        args[0] = self;
        if (extra != null) System.arraycopy(extra, 0, args, 1, extra.length);
        return eng.traceFreeOp(name, ns, args, body);
    }

    private static Tensor traceBinary(String name, String ns, Tensor a, Tensor b, TensorSupplier body) {
        VistaEngine eng = BOUND.get();
        if (eng == null) return body.get();
        return eng.traceFreeOp(name, ns, new Object[]{a, b}, body);
    }
}
