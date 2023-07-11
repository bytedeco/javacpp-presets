package org.bytedeco.pytorch;

import org.bytedeco.pytorch.functions.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

/* This is a modified version of the variant container whose get2 method
 * returns a function pointer instead of a std::function */
@NoOffset @Name("c10::variant<torch::enumtype::kReLU,torch::enumtype::kGELU,std::function<torch::Tensor(const torch::Tensor&)> >") @Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
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

    @Namespace @Name("c10::get<0>") static native @ByRef kReLU get0(@ByRef TransformerActivation container);

    @ValueSetter public native TransformerActivation put(@ByRef kReLU value);

    public kGELU get1() {
        return get1(this);
    }

    @Namespace @Name("c10::get<1>") static native @ByRef kGELU get1(@ByRef TransformerActivation container);

    @ValueSetter public native TransformerActivation put(@ByRef kGELU value);

    public TensorMapper get2() {
        return get2(this).target();
    }

    @Namespace @Name("c10::get<2>") static native @ByRef TensorMapperFunction get2(@ByRef TransformerActivation container);

    @ValueSetter public native TransformerActivation put(@ByRef TensorMapper value);

    @Name("std::function<torch::Tensor(const torch::Tensor&)>")
    @Namespace
    static class TensorMapperFunction extends Pointer {
        @Name("target<torch::Tensor(const torch::Tensor&)>") native TensorMapper target();
    }
}

