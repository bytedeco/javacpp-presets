package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.SharedPtr;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class SharedModuleApplyFunction extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public SharedModuleApplyFunction(Pointer p) {
        super(p);
    }

    protected SharedModuleApplyFunction() {
        allocate();
    }

    private native void allocate();

    public native void call(@SharedPtr @ByRef @Cast({"", "std::shared_ptr<torch::nn::Module>"}) Module m);
}
