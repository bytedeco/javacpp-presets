// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_core;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.opencv.global.opencv_core.*;


/** @see TermCriteria
 */
@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public class CvTermCriteria extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvTermCriteria() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CvTermCriteria(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvTermCriteria(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CvTermCriteria position(long position) {
        return (CvTermCriteria)super.position(position);
    }
    @Override public CvTermCriteria getPointer(long i) {
        return new CvTermCriteria((Pointer)this).offsetAddress(i);
    }

    /** may be combination of
                         CV_TERMCRIT_ITER
                         CV_TERMCRIT_EPS */
    public native int type(); public native CvTermCriteria type(int setter);
    public native int max_iter(); public native CvTermCriteria max_iter(int setter);
    public native double epsilon(); public native CvTermCriteria epsilon(double setter);
// #if defined(CV__ENABLE_C_API_CTORS) && defined(__cplusplus)
// #endif
// #ifdef __cplusplus
    public native @ByVal @Name("operator cv::TermCriteria") TermCriteria asTermCriteria();
// #endif
}
