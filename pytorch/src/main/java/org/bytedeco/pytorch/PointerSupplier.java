package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PointerSupplier extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public PointerSupplier(Pointer p) {
        super(p);
    }

    protected PointerSupplier() {
        allocate();
    }

    private native void allocate();

    public native Pointer call();
}
