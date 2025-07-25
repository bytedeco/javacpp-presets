// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_core;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.opencv.global.opencv_core.*;


@Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public class CvGraphScanner extends AbstractCvGraphScanner {
    static { Loader.load(); }
    /** Default native constructor. */
    public CvGraphScanner() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CvGraphScanner(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CvGraphScanner(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CvGraphScanner position(long position) {
        return (CvGraphScanner)super.position(position);
    }
    @Override public CvGraphScanner getPointer(long i) {
        return new CvGraphScanner((Pointer)this).offsetAddress(i);
    }

    public native CvGraphVtx vtx(); public native CvGraphScanner vtx(CvGraphVtx setter);       /* current graph vertex (or current edge origin) */
    public native CvGraphVtx dst(); public native CvGraphScanner dst(CvGraphVtx setter);       /* current graph edge destination vertex */
    public native CvGraphEdge edge(); public native CvGraphScanner edge(CvGraphEdge setter);     /* current edge */

    public native CvGraph graph(); public native CvGraphScanner graph(CvGraph setter);        /* the graph */
    public native CvSeq stack(); public native CvGraphScanner stack(CvSeq setter);        /* the graph vertex stack */
    public native int index(); public native CvGraphScanner index(int setter);        /* the lower bound of certainly visited vertices */
    public native int mask(); public native CvGraphScanner mask(int setter);         /* event mask */
}
