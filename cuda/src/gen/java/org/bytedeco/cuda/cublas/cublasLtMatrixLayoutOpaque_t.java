// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cublas;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.cublas.*;


/** Semi-opaque descriptor for matrix memory layout
 */
@Properties(inherit = org.bytedeco.cuda.presets.cublas.class)
public class cublasLtMatrixLayoutOpaque_t extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public cublasLtMatrixLayoutOpaque_t() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public cublasLtMatrixLayoutOpaque_t(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public cublasLtMatrixLayoutOpaque_t(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public cublasLtMatrixLayoutOpaque_t position(long position) {
        return (cublasLtMatrixLayoutOpaque_t)super.position(position);
    }
    @Override public cublasLtMatrixLayoutOpaque_t getPointer(long i) {
        return new cublasLtMatrixLayoutOpaque_t((Pointer)this).offsetAddress(i);
    }

  public native @Cast("uint64_t") long data(int i); public native cublasLtMatrixLayoutOpaque_t data(int i, long setter);
  @MemberGetter public native @Cast("uint64_t*") LongPointer data();
}
