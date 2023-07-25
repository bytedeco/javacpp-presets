package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.Type;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class TypeMapper extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public TypeMapper(Pointer p) {
        super(p);
    }

    protected TypeMapper() {
        allocate();
    }

    private native void allocate();

    public native @ByRef Type.TypePtr call(@ByVal Type.TypePtr t);
}
