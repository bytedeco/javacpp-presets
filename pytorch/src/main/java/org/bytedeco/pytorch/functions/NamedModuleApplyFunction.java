package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.Module;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class NamedModuleApplyFunction extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public NamedModuleApplyFunction(Pointer p) {
        super(p);
    }

    protected NamedModuleApplyFunction() {
        allocate();
    }

    private native void allocate();

    public native void call(@Const @StdString @ByRef String name, @ByRef Module m);
}
