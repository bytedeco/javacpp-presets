// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cupti;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.cupti.*;


/**
 * \brief Stream attribute data passed into a resource callback function
 * for CUPTI_CBID_RESOURCE_STREAM_ATTRIBUTE_CHANGED callback
 <p>
 * Data passed into a resource callback function as the \p cbdata
 * argument to \ref CUpti_CallbackFunc. The \p cbdata will be this
 * type for \p domain equal to CUPTI_CB_DOMAIN_RESOURCE. The
 * stream attribute data is valid only within the invocation of the callback
 * function that is passed the data. If you need to retain some data
 * for use outside of the callback, you must make a copy of that data.
 */
@Properties(inherit = org.bytedeco.cuda.presets.cupti.class)
public class CUpti_StreamAttrData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUpti_StreamAttrData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUpti_StreamAttrData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUpti_StreamAttrData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUpti_StreamAttrData position(long position) {
        return (CUpti_StreamAttrData)super.position(position);
    }
    @Override public CUpti_StreamAttrData getPointer(long i) {
        return new CUpti_StreamAttrData((Pointer)this).offsetAddress(i);
    }

  /**
   * The CUDA stream handle for the attribute
   */
  public native CUstream_st stream(); public native CUpti_StreamAttrData stream(CUstream_st setter);

  /**
   * The type of the CUDA stream attribute
   */
  public native @Cast("CUstreamAttrID") int attr(); public native CUpti_StreamAttrData attr(int setter);

  /**
   * The value of the CUDA stream attribute
   */
  public native @Cast("const CUstreamAttrValue*") CUlaunchAttributeValue value(); public native CUpti_StreamAttrData value(CUlaunchAttributeValue setter);
}
