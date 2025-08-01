// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;

@Convention("CUDA_CB") @Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class CUlogsCallback extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    CUlogsCallback(Pointer p) { super(p); }
    protected CUlogsCallback() { allocate(); }
    private native void allocate();
    public native void call(Pointer data, @Cast("CUlogLevel") int logLevel, @Cast("char*") BytePointer message, @Cast("size_t") long length);
}
