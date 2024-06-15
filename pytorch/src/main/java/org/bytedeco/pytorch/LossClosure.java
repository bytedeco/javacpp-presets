package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.Tensor;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class LossClosure extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public LossClosure(Pointer p) {
        super(p);
    }

    protected LossClosure() {
        allocate();
    }

    private native void allocate();

    public native @ByRef Tensor call();
}
