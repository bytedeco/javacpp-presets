// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.onnxruntime;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.opencl.*;
import static org.bytedeco.opencl.global.OpenCL.*;
import org.bytedeco.dnnl.*;
import static org.bytedeco.dnnl.global.dnnl.*;

import static org.bytedeco.onnxruntime.global.onnxruntime.*;

@Name("Ort::detail::AllocatorImpl<OrtAllocator>") @Properties(inherit = org.bytedeco.onnxruntime.presets.onnxruntime.class)
public class AllocatorImpl extends BaseAllocator {
    static { Loader.load(); }
    /** Default native constructor. */
    public AllocatorImpl() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public AllocatorImpl(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AllocatorImpl(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public AllocatorImpl position(long position) {
        return (AllocatorImpl)super.position(position);
    }
    @Override public AllocatorImpl getPointer(long i) {
        return new AllocatorImpl((Pointer)this).offsetAddress(i);
    }


  public native Pointer Alloc(@Cast("size_t") long size);
  public native @ByVal MemoryAllocation GetAllocation(@Cast("size_t") long size);
  public native void Free(Pointer p);
  public native @ByVal @Cast("Ort::ConstMemoryInfo*") MemoryInfoImpl GetInfo();
}
