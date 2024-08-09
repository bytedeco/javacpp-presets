package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class MemCopyFunction extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public MemCopyFunction(Pointer p) {
        super(p);
    }

    protected MemCopyFunction() {
        allocate();
    }

    private native void allocate();

    // std::function<void(void*,const void*,size_t)>
    public native void call(Pointer dest, @Const Pointer src, @Cast("size_t") long n);
}
