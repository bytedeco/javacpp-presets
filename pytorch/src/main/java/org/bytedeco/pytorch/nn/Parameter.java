/*
 * Copyright (C) 2026 bytedeco.org and pytorch JavaCPP presets contributors
 * Hand-written peer — not generated from C++ (libtorch has no torch::nn::Parameter;
 * parameters are plain Tensor via Module::register_parameter). Mirrors Python
 * torch.nn.parameter.Parameter for API ergonomics.
 */
package org.bytedeco.pytorch.nn;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;

/**
 * A Tensor that is to be considered a module parameter (Python-style).
 *
 * <p>In libtorch, parameters are ordinary {@link Tensor}s registered with
 * {@link Module#register_parameter}. This class is a pure-Java subclass that
 * makes the intent explicit and defaults {@code requires_grad} to {@code true},
 * matching {@code torch.nn.Parameter} in Python.
 *
 * <p>Usable anywhere a {@link Tensor} is expected, including
 * {@code module.register_parameter(name, param)}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Parameter extends Tensor {
    static {
        Loader.load();
    }

    public Parameter(Pointer p) {
        super(p);
    }

    /**
     * Like Python {@code Parameter(data, requires_grad=True)}.
     * Shares storage with {@code data} (Tensor copy constructor bumps refcount).
     */
    public Parameter(Tensor data) {
        this(data, true);
    }

    /**
     * Like Python {@code Parameter(data, requires_grad)}.
     */
    public Parameter(Tensor data, boolean requires_grad) {
        super(data);
        this.requires_grad_(requires_grad);
    }

    /** Factory: {@code Parameter.of(tensor)} with {@code requires_grad=true}. */
    public static Parameter of(Tensor data) {
        return new Parameter(data, true);
    }

    /** Factory: {@code Parameter.of(tensor, requires_grad)}. */
    public static Parameter of(Tensor data, boolean requires_grad) {
        return new Parameter(data, requires_grad);
    }
}
