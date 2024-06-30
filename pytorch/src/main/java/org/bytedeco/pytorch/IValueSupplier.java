package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByPtr;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.IValue;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class IValueSupplier extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public IValueSupplier(Pointer p) {
        super(p);
    }

    protected IValueSupplier() {
        allocate();
    }

    private native void allocate();

    public native @ByPtr IValue call();
}
