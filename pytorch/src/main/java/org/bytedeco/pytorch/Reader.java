package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class Reader extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public Reader(Pointer p) {
        super(p);
    }

    protected Reader() {
        allocate();
    }

    private native void allocate();

    public native @Cast("size_t") long call(@Cast("uint64_t") long pos, Pointer buf, @Cast("size_t") long nbytes);
}
