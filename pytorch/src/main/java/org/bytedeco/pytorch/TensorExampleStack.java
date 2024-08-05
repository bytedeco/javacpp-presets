package org.bytedeco.pytorch;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/* See ExampleStack */

/** A {@code Collation} for {@code Example<Tensor, NoTarget>} types that stacks all data
 *  tensors into one tensor. */
@Name("torch::data::transforms::Stack<torch::data::Example<torch::Tensor,torch::data::example::NoTarget> >")  @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorExampleStack extends TensorExampleCollation {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public TensorExampleStack() { super(null); allocate(); }
    private native void allocate();
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TensorExampleStack(Pointer p) { super(p); }
}
