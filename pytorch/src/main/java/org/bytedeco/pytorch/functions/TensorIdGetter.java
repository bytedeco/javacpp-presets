package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.Tensor;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TensorIdGetter extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TensorIdGetter(Pointer p) {
        super(p);
    }

    protected TensorIdGetter() {
        allocate();
    }

    private native void allocate();

    // std::function<std::string(const at::Tensor&)>
    public native @StdString String call(@Const @ByRef Tensor tensor);
}
