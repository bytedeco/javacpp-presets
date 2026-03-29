package org.bytedeco.pytorch;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Const;
import org.bytedeco.javacpp.annotation.Properties;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class ArchiveWriter extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public ArchiveWriter(Pointer p) {
        super(p);
    }

    protected ArchiveWriter() {
        allocate();
    }

    private native void allocate();

    public native @Cast("size_t") long call(@Const Pointer buf, @Cast("size_t") long nbytes);
}
