// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.nvml;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.nvml.*;


/**
 * Utilization information for a device.
 * Each sample period may be between 1 second and 1/6 second, depending on the product being queried.
 */
@Properties(inherit = org.bytedeco.cuda.presets.nvml.class)
public class nvmlUtilization_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public nvmlUtilization_t() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public nvmlUtilization_t(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public nvmlUtilization_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public nvmlUtilization_t position(long position) {
        return (nvmlUtilization_t)super.position(position);
    }
    @Override public nvmlUtilization_t getPointer(long i) {
        return new nvmlUtilization_t((Pointer)this).offsetAddress(i);
    }

    /** Percent of time over the past sample period during which one or more kernels was executing on the GPU */
    public native @Cast("unsigned int") int gpu(); public native nvmlUtilization_t gpu(int setter);
    /** Percent of time over the past sample period during which global (device) memory was being read or written */
    public native @Cast("unsigned int") int memory(); public native nvmlUtilization_t memory(int setter);
}
