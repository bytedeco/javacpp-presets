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

@Name("Ort::detail::Base<Ort::detail::Unowned<const OrtSessionOptions> >") @NoOffset @Properties(inherit = org.bytedeco.onnxruntime.presets.onnxruntime.class)
public class BaseConstSessionOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BaseConstSessionOptions(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BaseConstSessionOptions(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public BaseConstSessionOptions position(long position) {
        return (BaseConstSessionOptions)super.position(position);
    }
    @Override public BaseConstSessionOptions getPointer(long i) {
        return new BaseConstSessionOptions((Pointer)this).offsetAddress(i);
    }


  public BaseConstSessionOptions() { super((Pointer)null); allocate(); }
  private native void allocate();
  public BaseConstSessionOptions(@Cast("Ort::detail::Base<Ort::detail::Unowned<const OrtSessionOptions> >::contained_type*") UnownedAllocator p) { super((Pointer)null); allocate(p); }
  @NoException(true) private native void allocate(@Cast("Ort::detail::Base<Ort::detail::Unowned<const OrtSessionOptions> >::contained_type*") UnownedAllocator p);

  
  

  public BaseConstSessionOptions(@ByRef(true) BaseConstSessionOptions v) { super((Pointer)null); allocate(v); }
  @NoException(true) private native void allocate(@ByRef(true) BaseConstSessionOptions v);
  public native @ByRef @Name("operator =") @NoException(true) BaseConstSessionOptions put(@ByRef(true) BaseConstSessionOptions v);

  public native @Cast("Ort::detail::Base<Ort::detail::Unowned<const OrtSessionOptions> >::contained_type*") @Name("operator Ort::detail::Base<Ort::detail::Unowned<const OrtSessionOptions> >::contained_type*") @NoException(true) UnownedAllocator asUnownedAllocator();

  /** \brief Relinquishes ownership of the contained C object pointer
   *  The underlying object is not destroyed */
  
}
