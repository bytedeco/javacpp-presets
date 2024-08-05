package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;

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

    public native void call(@Const @StdString BytePointer name, @ByRef Module m);
}
