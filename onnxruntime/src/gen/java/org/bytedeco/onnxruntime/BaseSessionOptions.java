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

@Name("Ort::detail::Base<OrtSessionOptions>") @NoOffset @Properties(inherit = org.bytedeco.onnxruntime.presets.onnxruntime.class)
public class BaseSessionOptions extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public BaseSessionOptions(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public BaseSessionOptions(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public BaseSessionOptions position(long position) {
        return (BaseSessionOptions)super.position(position);
    }
    @Override public BaseSessionOptions getPointer(long i) {
        return new BaseSessionOptions((Pointer)this).offsetAddress(i);
    }


  public BaseSessionOptions() { super((Pointer)null); allocate(); }
  private native void allocate();
  public BaseSessionOptions(@Cast("Ort::detail::Base<OrtSessionOptions>::contained_type*") OrtSessionOptions p) { super((Pointer)null); allocate(p); }
  @NoException(true) private native void allocate(@Cast("Ort::detail::Base<OrtSessionOptions>::contained_type*") OrtSessionOptions p);

  
  

  public BaseSessionOptions(@ByRef(true) BaseSessionOptions v) { super((Pointer)null); allocate(v); }
  @NoException(true) private native void allocate(@ByRef(true) BaseSessionOptions v);
  public native @ByRef @Name("operator =") @NoException(true) BaseSessionOptions put(@ByRef(true) BaseSessionOptions v);

  public native @Cast("Ort::detail::Base<OrtSessionOptions>::contained_type*") @Name("operator Ort::detail::Base<OrtSessionOptions>::contained_type*") @NoException(true) OrtSessionOptions asOrtSessionOptions();

  /** \brief Relinquishes ownership of the contained C object pointer
   *  The underlying object is not destroyed */
  public native @Cast("Ort::detail::Base<OrtSessionOptions>::contained_type*") OrtSessionOptions release();
}
