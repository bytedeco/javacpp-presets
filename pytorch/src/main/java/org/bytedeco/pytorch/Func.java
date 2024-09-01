package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Func extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public Func(Pointer p) {
        super(p);
    }

    protected Func() {
        allocate();
    }

    private native void allocate();

    public native void call();
}
