package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NamedSharedModuleApplyFunction extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public NamedSharedModuleApplyFunction(Pointer p) {
        super(p);
    }

    protected NamedSharedModuleApplyFunction() {
        allocate();
    }

    private native void allocate();

    public native void call(@Const @StdString BytePointer name, @ByRef @SharedPtr @Cast({"", "std::shared_ptr<torch::nn::Module>"}) Module m);
}
