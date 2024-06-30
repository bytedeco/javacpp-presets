package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.pytorch.TensorBase;


@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class VoidTensorHook extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public VoidTensorHook(Pointer p) {
        super(p);
    }

    protected VoidTensorHook() {
        allocate();
    }

    private native void allocate();

    public native void call(@ByVal TensorBase a);
}
