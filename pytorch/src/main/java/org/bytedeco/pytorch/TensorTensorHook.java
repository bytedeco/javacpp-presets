package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorTensorHook extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TensorTensorHook(Pointer p) {
        super(p);
    }

    protected TensorTensorHook() {
        allocate();
    }

    private native void allocate();

    public native @ByRef TensorBase call(@ByVal TensorBase a);
}
