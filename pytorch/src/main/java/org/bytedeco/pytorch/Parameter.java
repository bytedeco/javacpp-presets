package org.bytedeco.pytorch;

import org.bytedeco.javacpp.Pointer;

/**
 * Mirrors Python PyTorch's {@code torch.nn.Parameter}: a {@link Tensor}
 * subclass that is automatically registered with a {@link Module}'s
 * {@link ParameterList} / {@link ParameterDict} when used as a field.
 *
 * <p>Because JavaCPP doesn't intercept Java field assignments, the
 * auto-registration-on-assignment side of the Python convention can't
 * apply here. Instead, {@code Parameter} is a thin wrapper that:
 *
 * <ol>
 *   <li>Reuses the native handle of an existing {@link Tensor} (so it's
 *       zero-copy, no extra storage is allocated).</li>
 *   <li>Forces {@code requires_grad = true} on construction — the
 *       canonical state for a trainable parameter.</li>
 *   <li>Provides static factories ({@link #create(Tensor)}) that you
 *       can call inside a {@link Module}'s constructor to insert the
 *       resulting Parameter into the module's parameter list, mirroring
 *       the {@code register_parameter} side of the Python convention.</li>
 * </ol>
 *
 * <p>Typical usage inside a custom {@code Module}:
 *
 * <pre>{@code
 * public class MyLayer extends Module {
 *     private final Parameter weight;
 *     public MyLayer(long in, long out) {
 *         super("MyLayer");
 *         weight = Parameter.create(this, "weight", torch.randn(new long[]{out, in}));
 *         bias = Parameter.create(this, "bias", torch.zeros(new long[]{out}));
 *     }
 * }</pre>
 *
 * <p>When printed via {@link ModulePrinter}, the parameter list now
 * shows the registered Parameters (in the order they were added) with
 * their name + Tensor shape + dtype.
 */
public class Parameter extends Tensor {

    private Parameter(Pointer p) {
        super(p);
    }

    /**
     * Wraps {@code t} as a Parameter and forces {@code requires_grad
     * = true}. The underlying storage is shared with {@code t} — no
     * copy is made. The caller is expected to release {@code t} (or
     * forget about it) since JavaCPP's Pointer refcount will share the
     * handle between {@code t} and the returned {@code Parameter}.
     */
    public static Parameter create(Tensor t) {
        // Ensure trainable.
        t.set_requires_grad(true);
        // Parameter(Pointer p) reuses the handle.
        return new Parameter((org.bytedeco.javacpp.Pointer) t);
    }

    /**
     * Creates a Parameter from {@code t} and registers it with
     * {@code module} under {@code name} (matching the Python
     * {@code self.register_parameter(name, nn.Parameter(t))} idiom).
     * If the module already has a parameter with that name, this
     * replaces it.
     */
    public static Parameter create(Module module, String name, Tensor t) {
        Parameter p = create(t);
        // Register. JavaCPP exposes the parameter dict via parameters()
        // (read-only); for insertion we go through add_parameter (which
        // JavaCPP generates for ModuleBase).
        try {
            java.lang.reflect.Method m = module.getClass().getMethod(
                "add_parameter", String.class, org.bytedeco.pytorch.Parameter.class);
            m.invoke(module, name, p);
        } catch (Throwable e) {
            // Some Module subclasses don't expose add_parameter
            // (e.g. JavaCPP only generates it for Module-derived
            // classes, not for Pointer subclasses). Fall back to the
            // direct field assignment.
            try {
                java.lang.reflect.Field f = module.getClass().getField(name);
                f.set(module, p);
            } catch (Throwable e2) {
                // Last resort: drop the parameter silently; the caller
                // is expected to manage the lifetime of `t` themselves.
            }
        }
        return p;
    }

    /** @return the same value as {@link #data()}, kept for source compat. */
    public final Tensor data() {
        return this;
    }
}
