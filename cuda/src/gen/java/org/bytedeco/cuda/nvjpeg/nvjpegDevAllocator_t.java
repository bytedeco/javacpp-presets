// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.nvjpeg;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.nvjpeg.*;


// Memory allocator using mentioned prototypes, provided to nvjpegCreateEx
// This allocator will be used for all device memory allocations inside library
// In any way library is doing smart allocations (reallocates memory only if needed)
@Properties(inherit = org.bytedeco.cuda.presets.nvjpeg.class)
public class nvjpegDevAllocator_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public nvjpegDevAllocator_t() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public nvjpegDevAllocator_t(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public nvjpegDevAllocator_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public nvjpegDevAllocator_t position(long position) {
        return (nvjpegDevAllocator_t)super.position(position);
    }
    @Override public nvjpegDevAllocator_t getPointer(long i) {
        return new nvjpegDevAllocator_t((Pointer)this).offsetAddress(i);
    }

    public native tDevMalloc dev_malloc(); public native nvjpegDevAllocator_t dev_malloc(tDevMalloc setter);
    public native tDevFree dev_free(); public native nvjpegDevAllocator_t dev_free(tDevFree setter);
}
