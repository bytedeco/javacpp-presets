package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.ByRef;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.JitModule;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class JitModuleApplyFunction extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public JitModuleApplyFunction(Pointer p) {
        super(p);
    }

    protected JitModuleApplyFunction() {
        allocate();
    }

    private native void allocate();

    public native void call(@ByRef JitModule m);
}
