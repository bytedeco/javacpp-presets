package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Type.TypePtr;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TypeSupplier extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TypeSupplier(Pointer p) {
        super(p);
    }

    protected TypeSupplier() {
        allocate();
    }

    private native void allocate();

    public native @ByVal TypePtr call();
}
