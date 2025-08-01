// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_core;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.opencv.global.opencv_core.*;

// #endif

@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public class Cv16suf extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public Cv16suf() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Cv16suf(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Cv16suf(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public Cv16suf position(long position) {
        return (Cv16suf)super.position(position);
    }
    @Override public Cv16suf getPointer(long i) {
        return new Cv16suf((Pointer)this).offsetAddress(i);
    }

    public native short i(); public native Cv16suf i(short setter);
    public native @Cast("ushort") short u(); public native Cv16suf u(short setter);
// #if CV_FP16_TYPE
// #endif
}
