package org.bytedeco.pytorch;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/* This is an instantiation of a specialized template defined in torch/data/transforms/stack.h.
 * Template specializations are ignored since 1.5.10.
 * The primary template being a simple declaration without body, Parser would create
 * an @Opaque class without mapping any native constructor.
 * So we give this explicit definition with a native constructor and ignore stack.h during parsing. */

/** A {@code Collation} for {@code Example<Tensor, Tensor>} types that stacks all data
 *  tensors into one tensor, and all target (label) tensors into one tensor. */
@Name("torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::Tensor> >")  @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ExampleStack extends ExampleCollation {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public ExampleStack() { super(null); allocate(); }
    private native void allocate();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ExampleStack(Pointer p) { super(p); }
}
