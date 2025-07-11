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
 * Data structure for HaarClassifier_32f.
 */

@Properties(inherit = org.bytedeco.cuda.presets.nppc.class)
public class NppiHaarClassifier_32f extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NppiHaarClassifier_32f() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NppiHaarClassifier_32f(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NppiHaarClassifier_32f(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NppiHaarClassifier_32f position(long position) {
        return (NppiHaarClassifier_32f)super.position(position);
    }
    @Override public NppiHaarClassifier_32f getPointer(long i) {
        return new NppiHaarClassifier_32f((Pointer)this).offsetAddress(i);
    }

    /**  number of classifiers. */
    public native int numClassifiers(); public native NppiHaarClassifier_32f numClassifiers(int setter);
    /**  packed classifier data 40 bytes each. */
    public native @Cast("Npp32s*") IntPointer classifiers(); public native NppiHaarClassifier_32f classifiers(IntPointer setter);
    /**  packed classifier byte step. */
    public native @Cast("size_t") long classifierStep(); public native NppiHaarClassifier_32f classifierStep(long setter);
    /**  packed classifier size. */
    public native @ByRef NppiSize classifierSize(); public native NppiHaarClassifier_32f classifierSize(NppiSize setter);
    /**  counter device. */
    public native @Cast("Npp32s*") IntPointer counterDevice(); public native NppiHaarClassifier_32f counterDevice(IntPointer setter);
}
