// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;


@Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class longlong3 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public longlong3() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public longlong3(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public longlong3(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public longlong3 position(long position) {
        return (longlong3)super.position(position);
    }
    @Override public longlong3 getPointer(long i) {
        return new longlong3((Pointer)this).offsetAddress(i);
    }

    public native long x(); public native longlong3 x(long setter);
    public native long y(); public native longlong3 y(long setter);
    public native long z(); public native longlong3 z(long setter);
}
