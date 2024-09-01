package org.bytedeco.pytorch;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class PickleReader extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public PickleReader(Pointer p) {
        super(p);
    }

    protected PickleReader() {
        allocate();
    }

    private native void allocate();

    // std::function<long unsigned int(char*, long unsigned int)>
    public native @Cast("size_t") long call(@Cast("char*") BytePointer buf, @Cast("size_t") long nbytes);
}
