package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.Value;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ValueMapper extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public ValueMapper(Pointer p) {
        super(p);
    }

    protected ValueMapper() {
        allocate();
    }

    private native void allocate();

    public native @ByPtr Value call(@ByPtr Value v);
}
