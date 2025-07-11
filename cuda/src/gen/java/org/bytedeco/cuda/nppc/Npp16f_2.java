// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.nppc;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.nppc.*;


/** 
  * Npp16f_2 
  */

@Properties(inherit = org.bytedeco.cuda.presets.nppc.class)
public class Npp16f_2 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Npp16f_2() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Npp16f_2(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Npp16f_2(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public Npp16f_2 position(long position) {
        return (Npp16f_2)super.position(position);
    }
    @Override public Npp16f_2 getPointer(long i) {
        return new Npp16f_2((Pointer)this).offsetAddress(i);
    }

   /** Original Cuda fp16 data size and format. */
   public native short fp16_0(); public native Npp16f_2 fp16_0(short setter);
   /** Original Cuda fp16 data size and format. */
   public native short fp16_1(); public native Npp16f_2 fp16_1(short setter);
}
