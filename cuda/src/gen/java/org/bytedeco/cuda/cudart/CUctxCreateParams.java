// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;


/**
* Params for creating CUDA context
* Exactly one of execAffinityParams and cigParams 
* must be non-NULL.
*/
@Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class CUctxCreateParams extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CUctxCreateParams() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CUctxCreateParams(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUctxCreateParams(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CUctxCreateParams position(long position) {
        return (CUctxCreateParams)super.position(position);
    }
    @Override public CUctxCreateParams getPointer(long i) {
        return new CUctxCreateParams((Pointer)this).offsetAddress(i);
    }

    public native @Cast("CUexecAffinityParam*") CUexecAffinityParam_v1 execAffinityParams(); public native CUctxCreateParams execAffinityParams(CUexecAffinityParam_v1 setter);
    public native int numExecAffinityParams(); public native CUctxCreateParams numExecAffinityParams(int setter);
    public native CUctxCigParam cigParams(); public native CUctxCreateParams cigParams(CUctxCigParam setter);
}
