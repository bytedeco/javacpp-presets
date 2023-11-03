package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.StringBoolMap;
import org.bytedeco.pytorch.Tensor;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class BackendMetaPtr extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public BackendMetaPtr(Pointer p) {
        super(p);
    }

    protected BackendMetaPtr() {
        allocate();
    }

    private native void allocate();

    // std::function<void(const at::Tensor&, std::unordered_map<std::string, bool>&)>
    public native void call(@Const @ByRef Tensor tensor, @ByRef StringBoolMap map);
}
