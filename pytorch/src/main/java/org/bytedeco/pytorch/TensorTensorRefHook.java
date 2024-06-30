package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.TensorBase;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorTensorRefHook extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TensorTensorRefHook(Pointer p) {
        super(p);
    }

    protected TensorTensorRefHook() {
        allocate();
    }

    private native void allocate();

    public native @ByRef TensorBase call(@Const @ByRef TensorBase a);
}
