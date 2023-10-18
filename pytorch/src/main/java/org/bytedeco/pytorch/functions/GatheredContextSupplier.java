package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.SharedPtr;
import org.bytedeco.pytorch.GatheredContext;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GatheredContextSupplier extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public GatheredContextSupplier(Pointer p) {
        super(p);
    }

    protected GatheredContextSupplier() {
        allocate();
    }

    private native void allocate();

    // See issue JavaCPP #720
    public native @Cast({"", "std::shared_ptr<c10::GatheredContext>"}) @SharedPtr GatheredContext call();
}
