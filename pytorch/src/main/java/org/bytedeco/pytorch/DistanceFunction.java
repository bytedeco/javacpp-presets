package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class DistanceFunction extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public DistanceFunction(Pointer p) {
        super(p);
    }

    protected DistanceFunction() {
        allocate();
    }

    private native void allocate();

    public native @ByVal Tensor call(@ByRef Tensor t1, @ByRef Tensor t2);
}
