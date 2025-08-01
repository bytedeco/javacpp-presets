// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_core;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.opencv.global.opencv_core.*;

@Name("cv::Point_<double>") @NoOffset @Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public class Point2d extends DoublePointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public Point2d(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public Point2d(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public Point2d position(long position) {
        return (Point2d)super.position(position);
    }
    @Override public Point2d getPointer(long i) {
        return new Point2d((Pointer)this).offsetAddress(i);
    }


    /** default constructor */
    public Point2d() { super((Pointer)null); allocate(); }
    private native void allocate();
    public Point2d(double _x, double _y) { super((Pointer)null); allocate(_x, _y); }
    private native void allocate(double _x, double _y);
// #if (defined(__GNUC__) && __GNUC__ < 5) && !defined(__clang__)  // GCC 4.x bug. Details: https://github.com/opencv/opencv/pull/20837
    public Point2d(@Const @ByRef Point2d pt) { super((Pointer)null); allocate(pt); }
    private native void allocate(@Const @ByRef Point2d pt);
// #elif OPENCV_ABI_COMPATIBILITY < 500
// #endif
    public Point2d(@Const @ByRef Size2d sz) { super((Pointer)null); allocate(sz); }
    private native void allocate(@Const @ByRef Size2d sz);

// #if (defined(__GNUC__) && __GNUC__ < 5) && !defined(__clang__)  // GCC 4.x bug. Details: https://github.com/opencv/opencv/pull/20837
    public native @ByRef @Name("operator =") Point2d put(@Const @ByRef Point2d pt);
// #elif OPENCV_ABI_COMPATIBILITY < 500
// #endif
    /** conversion to another data type */

    /** conversion to the old-style C structures */

    /** dot product */
    public native double dot(@Const @ByRef Point2d pt);
    /** dot product computed in double-precision arithmetics */
    public native double ddot(@Const @ByRef Point2d pt);
    /** cross-product */
    public native double cross(@Const @ByRef Point2d pt);
    /** checks whether the point is inside the specified rectangle */
    public native @Cast("bool") boolean inside(@Const @ByRef Rect2d r);
    /** x coordinate of the point */
    public native double x(); public native Point2d x(double setter);
    /** y coordinate of the point */
    public native double y(); public native Point2d y(double setter);
}
