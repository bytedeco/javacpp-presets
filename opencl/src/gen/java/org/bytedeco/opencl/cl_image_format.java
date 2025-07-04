// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.opencl;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.opencl.global.OpenCL.*;

// #endif

@Properties(inherit = org.bytedeco.opencl.presets.OpenCL.class)
public class cl_image_format extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cl_image_format() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public cl_image_format(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cl_image_format(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public cl_image_format position(long position) {
        return (cl_image_format)super.position(position);
    }
    @Override public cl_image_format getPointer(long i) {
        return new cl_image_format((Pointer)this).offsetAddress(i);
    }

    public native @Cast("cl_channel_order") int image_channel_order(); public native cl_image_format image_channel_order(int setter);
    public native @Cast("cl_channel_type") int image_channel_data_type(); public native cl_image_format image_channel_data_type(int setter);
}
