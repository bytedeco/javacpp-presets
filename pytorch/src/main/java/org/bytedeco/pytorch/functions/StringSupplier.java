package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.StdString;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StringSupplier extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public StringSupplier(Pointer p) {
        super(p);
    }

    protected StringSupplier() {
        allocate();
    }

    private native void allocate();

    public native @StdString String call();
}
