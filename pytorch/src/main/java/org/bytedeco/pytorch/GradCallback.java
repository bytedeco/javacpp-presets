package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.ByVal;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Tensor;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class GradCallback extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public GradCallback(Pointer p) {
        super(p);
    }

    protected GradCallback() {
        allocate();
    }

    private native void allocate();

    //  std::function<bool(torch::Tensor&)>
    public native @ByVal boolean call(@ByRef Tensor a);
}
