package org.bytedeco.pytorch;

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/* This is a modified version of the variant container without the get2 method, that would
 * return a std::function and not a function pointer. */
@NoOffset @Name("std::variant<torch::enumtype::kReLU,torch::enumtype::kGELU,std::function<torch::Tensor(const torch::Tensor&)> >") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TransformerActivation extends Pointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TransformerActivation(Pointer p) {
        super(p);
    }

    public TransformerActivation(kReLU value) {
        this();
        put(value);
    }

    public TransformerActivation(kGELU value) {
        this();
        put(value);
    }

    public TransformerActivation(TensorMapper value) {
        this();
        put(value);
    }

    public TransformerActivation() {
        allocate();
    }

    private native void allocate();

    public native @Name("operator =") @ByRef TransformerActivation put(@ByRef TransformerActivation x);

    public kReLU get0() {
        return get0(this);
    }

    @Namespace @Name("std::get<0>") static native @ByRef kReLU get0(@ByRef TransformerActivation container);

    @ValueSetter public native TransformerActivation put(@ByRef kReLU value);

    public kGELU get1() {
        return get1(this);
    }

    @Namespace @Name("std::get<1>") static native @ByRef kGELU get1(@ByRef TransformerActivation container);

    @ValueSetter public native TransformerActivation put(@ByRef kGELU value);

    @ValueSetter public native TransformerActivation put(@ByRef TensorMapper value);
}

