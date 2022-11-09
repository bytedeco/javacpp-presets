// Targeted by JavaCPP version 1.5.8: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.nppc;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import static org.bytedeco.cuda.global.cudart.*;

import static org.bytedeco.cuda.global.nppc.*;


/** 
 * 2D Npp32f Point
 */
@Properties(inherit = org.bytedeco.cuda.presets.nppc.class)
public class NppiPoint32f extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NppiPoint32f() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NppiPoint32f(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NppiPoint32f(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NppiPoint32f position(long position) {
        return (NppiPoint32f)super.position(position);
    }
    @Override public NppiPoint32f getPointer(long i) {
        return new NppiPoint32f((Pointer)this).offsetAddress(i);
    }

    /**  x-coordinate. */
    public native @Cast("Npp32f") float x(); public native NppiPoint32f x(float setter);
    /**  y-coordinate. */
    public native @Cast("Npp32f") float y(); public native NppiPoint32f y(float setter);
}