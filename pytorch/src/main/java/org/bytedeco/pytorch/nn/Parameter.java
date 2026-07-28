/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 *
 * Hand-written peer — not generated from C++. libtorch has no torch::nn::Parameter;
 * parameters are plain Tensor registered via Module::register_parameter. This class
 * mirrors Python torch.nn.parameter.Parameter for type tagging + ergonomics.
 */
package org.bytedeco.pytorch.nn;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

/**
 * A {@link Tensor} tagged as a module parameter (Python {@code torch.nn.Parameter}).
 *
 * <h2>Why this exists</h2>
 * In C++/libtorch, parameters are ordinary {@code Tensor}s discovered via
 * {@link Module#register_parameter}. Python adds a thin subclass for:
 * <ul>
 *   <li>type identity ({@code isinstance(t, nn.Parameter)})</li>
 *   <li>default {@code requires_grad=True}</li>
 *   <li>preserving parameter-ness across some ops</li>
 * </ul>
 * This class provides the same ergonomics for Java without inventing a second
 * autograd system.
 *
 * <h2>Design rules (enterprise / JavaCPP)</h2>
 * <ol>
 *   <li><b>IS-A Tensor</b> — usable anywhere a {@link Tensor} is expected,
 *       including {@code module.register_parameter(name, param)}.</li>
 *   <li><b>Share storage</b> — the Tensor copy constructor bumps the refcount;
 *       we do <em>not</em> clone by default. Cloning would break leaf identity
 *       and desync the handle you keep in a Java field from the one Module owns.</li>
 *   <li><b>Never shadow {@code grad} / {@code requires_grad}</b> — both live in
 *       native {@code AutogradMeta}. A Java field copy goes stale after
 *       {@code backward()} and is the root cause of the old geometric Parameter bugs.</li>
 *   <li><b>Ownership</b> — keep the original Java handle. {@code register_parameter}
 *       returns a {@code @ByRef} view that <em>must not</em> be stored in a field
 *       (dangling → SIGSEGV on {@code numel()}/{@code t()}). Prefer
 *       {@link #register(Module, String)} which encodes this rule.</li>
 *   <li><b>One type for all modules</b> — PyG / LLM / RL parameters are the same
 *       concept. Canonical (and only) type is {@code org.bytedeco.pytorch.nn.Parameter}.</li>
 * </ol>
 *
 * <h2>Correct usage</h2>
 * <pre>{@code
 * // 1) own a leaf tensor
 * Tensor wInit = torch.randn(out, in).contiguous().clone();
 * Parameter weight = Parameter.of(wInit);          // requires_grad=true
 *
 * // 2) register for Module.parameters() / optim discovery — IGNORE ByRef return
 * register_parameter("weight", weight);            // or: weight.register(this, "weight");
 * this.weight = weight;                            // keep original handle
 *
 * // 3) after backward
 * optimizer.step();
 * weight.zero_grad(true);  // setToNone=true; or module.zero_grad()
 * }</pre>
 *
 * <h2>What this deliberately does NOT do</h2>
 * <ul>
 *   <li>Override {@link #grad()} with a cached Java field (always wrong after backward).</li>
 *   <li>Call {@code retain_grad()} on construction (leaf parameters already keep grad).</li>
 *   <li>Install a permanent {@code register_hook} inside {@code set_grad}.</li>
 *   <li>Return {@code this} from {@link #data()} (Python {@code .data} is a plain Tensor view).</li>
 * </ul>
 *
 * @see Module#register_parameter(String, Tensor, boolean)
 * @see Module#zero_grad(boolean)
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Parameter extends Tensor {

    static {
        Loader.load();
    }

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    /** JavaCPP / pointer rewrap — does not change requires_grad. */
    public Parameter(Pointer p) {
        super(p);
    }

    /**
     * Like Python {@code Parameter(data, requires_grad=True)}.
     * Shares storage with {@code data} (Tensor copy-ctor bumps refcount).
     */
    public Parameter(Tensor data) {
        this(data, true);
    }

    /**
     * Like Python {@code Parameter(data, requires_grad)}.
     *
     * <p>Storage is shared, not cloned. Pass a leaf tensor (fresh {@code randn}/
     * {@code empty}, or {@code t.detach()}) so the Parameter remains a leaf and
     * {@link #grad()} is populated by {@code backward()}.
     *
     * @param data           source tensor (must be non-null and {@link #defined()})
     * @param requires_grad  whether autograd should record ops on this parameter
     * @throws NullPointerException     if {@code data} is null
     * @throws IllegalArgumentException if {@code data} is undefined
     */
    public Parameter(Tensor data, boolean requires_grad) {
        super(requireDefined(data));
        // Native AutogradMeta is the single source of truth — no Java boolean mirror.
        super.requires_grad_(requires_grad);
    }

    // -------------------------------------------------------------------------
    // Factories
    // -------------------------------------------------------------------------

    /** {@code Parameter.of(t)} ≡ {@code new Parameter(t, true)}. */
    public static Parameter of(Tensor data) {
        return of(data, true);
    }

    /**
     * Wrap {@code data} as a Parameter. If {@code data} is already a
     * {@code Parameter}, returns it as-is (idempotent) after applying
     * {@code requires_grad} only when it differs.
     */
    public static Parameter of(Tensor data, boolean requires_grad) {
        if (data instanceof Parameter) {
            Parameter p = (Parameter) data;
            if (p.requires_grad() != requires_grad) {
                p.requires_grad_(requires_grad);
            }
            return p;
        }
        return new Parameter(data, requires_grad);
    }

    /**
     * Guarantee a leaf Parameter: {@code detach()} first, then wrap.
     * Use when re-wrapping a tensor that may already sit in a graph
     * ({@code grad_fn != null}). Prefer plain {@link #of(Tensor)} for fresh
     * init tensors.
     */
    public static Parameter leaf(Tensor data) {
        return leaf(data, true);
    }

    /** See {@link #leaf(Tensor)}. */
    public static Parameter leaf(Tensor data, boolean requires_grad) {
        requireDefined(data);
        // detach() breaks grad_fn so the result is a leaf; share that storage.
        return new Parameter(data.detach(), requires_grad);
    }

    /** True iff {@code t} is a non-null Parameter instance. */
    public static boolean isParameter(Tensor t) {
        return t instanceof Parameter;
    }

    // -------------------------------------------------------------------------
    // Python-aligned surface
    // -------------------------------------------------------------------------

    /**
     * Covariant {@code requires_grad_} so fluent chains stay typed as Parameter.
     * Delegates entirely to native AutogradMeta — no Java field cache.
     *
     * <p>Note: {@link #data()} is intentionally <em>not</em> overridden. Native
     * {@code Tensor.data()} already matches Python's legacy {@code .data} view.
     * The old geometric Parameter returned {@code this}, which broke call sites
     * expecting a plain Tensor buffer.
     */
    @Override
    public Parameter requires_grad_(boolean requires_grad) {
        super.requires_grad_(requires_grad);
        return this;
    }

    /**
     * Device / dtype move that preserves Parameter identity.
     * Equivalent to Python {@code Parameter.to(...)} (subclass preserved).
     *
     * <p>Note: the underlying {@code Tensor.to} allocates a new storage; the
     * returned Parameter is a <em>new</em> leaf. Callers that keep a field
     * must reassign: {@code this.weight = this.weight.to(device, dtype)}.
     */
    public Parameter to(Device device, torch.ScalarType dtype) {
        boolean rg = requires_grad();
        Tensor moved = super.to(device, dtype);
        return new Parameter(moved, rg);
    }

    /**
     * Device-only move, preserves Parameter + requires_grad + dtype.
     * Implemented as {@code to(device, scalar_type())} — libtorch has no
     * single-arg {@code Tensor.to(Device)} overload in these bindings.
     */
    public Parameter to(Device device) {
        return to(device, scalar_type());
    }

    // -------------------------------------------------------------------------
    // Grad helpers (thin, correct — no shadowing)
    // -------------------------------------------------------------------------

    /**
     * Assign into native {@code mutable_grad()} (C++ {@code param.mutable_grad() = g}).
     *
     * <p>Prefer letting {@code backward()} populate grad. Use this only for
     * tests, gradient surgery, or manual optim steps. Pass {@code null} or an
     * undefined tensor to clear (Python {@code param.grad = None}).
     *
     * <p><b>Does not</b> install a permanent autograd hook — that was a bug in
     * the old geometric implementation and silently replaced every future grad.
     */
    public void set_grad(Tensor newGrad) {
        if (newGrad == null || !newGrad.defined()) {
            // set_to_none semantics
            mutable_grad().put(new Tensor());
            return;
        }
        if (!sameShape(newGrad, this)) {
            throw new IllegalArgumentException(
                    "set_grad: grad shape " + shapeOf(newGrad)
                            + " incompatible with parameter shape " + shapeOf(this));
        }
        // Match device/dtype of this parameter (optimizer / backward contract).
        Tensor aligned = newGrad;
        if (!newGrad.device().equals(device())
                || newGrad.scalar_type() != scalar_type()) {
            aligned = newGrad.to(device(), scalar_type());
        }
        mutable_grad().put(aligned);
    }

    /**
     * Clear this parameter's grad. Matches {@link Module#zero_grad(boolean)}
     * semantics for a single tensor.
     *
     * @param setToNone {@code true} → grad becomes undefined (Python default
     *                  since 1.7); {@code false} → in-place zero_ on existing grad
     */
    public void zero_grad(boolean setToNone) {
        Tensor g = grad();
        if (g == null || !g.defined()) {
            return;
        }
        if (setToNone) {
            mutable_grad().put(new Tensor());
        } else {
            g.detach_();
            g.zero_();
        }
    }

    /** {@code zero_grad(true)} — preferred; matches modern Module.zero_grad. */
    public void zero_grad() {
        zero_grad(true);
    }

    // -------------------------------------------------------------------------
    // Module registration helper (encodes ByRef ownership rule)
    // -------------------------------------------------------------------------

    /**
     * Register this parameter on {@code module} under {@code name} and return
     * <em>this</em> (the owned Java handle).
     *
     * <p>Encodes the JavaCPP ownership rule:
     * <pre>{@code
     * // BAD — stores dangling @ByRef
     * this.weight = register_parameter("weight", init, true);
     *
     * // GOOD
     * this.weight = Parameter.of(init).register(this, "weight");
     * }</pre>
     *
     * @return {@code this} for fluent field assignment
     * @see Module#register_parameter(String, Tensor, boolean)
     */
    public Parameter register(Module module, String name) {
        if (module == null) {
            throw new NullPointerException("module");
        }
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("parameter name must be non-empty");
        }
        // Ignore @ByRef return — keep the pre-register handle (this).
        module.register_parameter(name, this, requires_grad());
        return this;
    }

    /**
     * Same as {@link #register(Module, String)} but forces the Module-side
     * {@code requires_grad} flag (passed through to
     * {@code register_parameter(name, tensor, requires_grad)}).
     */
    public Parameter register(Module module, String name, boolean requires_grad) {
        if (module == null) {
            throw new NullPointerException("module");
        }
        if (name == null || name.isEmpty()) {
            throw new IllegalArgumentException("parameter name must be non-empty");
        }
        if (requires_grad() != requires_grad) {
            requires_grad_(requires_grad);
        }
        module.register_parameter(name, this, requires_grad);
        return this;
    }

    // -------------------------------------------------------------------------
    // Diagnostics
    // -------------------------------------------------------------------------

    @Override
    public String toString() {
        // Mirror Python: "Parameter containing:\n<tensor repr>"
        String body;
        try {
            body = super.toString();
        } catch (Exception e) {
            body = "Tensor(defined=" + defined() + ")";
        }
        return "Parameter containing:\n" + body
                + ", requires_grad=" + requires_grad();
    }

    // -------------------------------------------------------------------------
    // Internals
    // -------------------------------------------------------------------------

    private static Tensor requireDefined(Tensor data) {
        if (data == null) {
            throw new NullPointerException("Parameter data must not be null");
        }
        if (!data.defined()) {
            throw new IllegalArgumentException(
                    "Parameter data is an undefined Tensor; pass a defined leaf "
                            + "(e.g. torch.randn(...).clone()) or use Module.register_parameter "
                            + "with an empty Tensor to declare an optional slot");
        }
        return data;
    }

    private static boolean sameShape(Tensor a, Tensor b) {
        if (a.dim() != b.dim()) {
            return false;
        }
        long d = a.dim();
        for (long i = 0; i < d; i++) {
            if (a.size(i) != b.size(i)) {
                return false;
            }
        }
        return true;
    }

    private static String shapeOf(Tensor t) {
        if (t == null || !t.defined()) {
            return "<undefined>";
        }
        StringBuilder sb = new StringBuilder("[");
        long d = t.dim();
        for (long i = 0; i < d; i++) {
            if (i > 0) sb.append(", ");
            sb.append(t.size(i));
        }
        return sb.append(']').toString();
    }
}
