package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PickleWriter extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public PickleWriter(Pointer p) {
        super(p);
    }

    protected PickleWriter() {
        allocate();
    }

    private native void allocate();

    public native void call(@Cast("const char*") BytePointer buf, @Cast("size_t") long nbytes);
}
