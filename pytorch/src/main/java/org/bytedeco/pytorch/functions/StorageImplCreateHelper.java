package org.bytedeco.pytorch.functions;

import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.annotation.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Allocator;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public class StorageImplCreateHelper extends FunctionPointer {
    static {
        Loader.load();
    }

    /**
     * Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}.
     */
    public StorageImplCreateHelper(Pointer p) {
        super(p);
    }

    protected StorageImplCreateHelper() {
        allocate();
    }

    private native void allocate();

    public native @ByVal StorageImplPtr call(
        @ByVal StorageImpl.use_byte_size_t arg0,
        @ByVal SymInt size_bytes,
        Allocator allocator,
        @Cast("bool") boolean resizable);
}
