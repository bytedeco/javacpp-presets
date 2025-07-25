// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.opencv.opencv_core;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import static org.bytedeco.openblas.global.openblas_nolapack.*;
import static org.bytedeco.openblas.global.openblas.*;

import static org.bytedeco.opencv.global.opencv_core.*;




////////////////////////////////// SparseMatIterator /////////////////////////////////

/** \brief  Read-write Sparse Matrix Iterator
 <p>
 The class is similar to cv::SparseMatConstIterator,
 but can be used for in-place modification of the matrix elements.
*/
@Namespace("cv") @Properties(inherit = org.bytedeco.opencv.presets.opencv_core.class)
public class SparseMatIterator extends SparseMatConstIterator {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SparseMatIterator(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public SparseMatIterator(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public SparseMatIterator position(long position) {
        return (SparseMatIterator)super.position(position);
    }
    @Override public SparseMatIterator getPointer(long i) {
        return new SparseMatIterator((Pointer)this).offsetAddress(i);
    }

    /** the default constructor */
    public SparseMatIterator() { super((Pointer)null); allocate(); }
    private native void allocate();
    /** the full constructor setting the iterator to the first sparse matrix element */
    public SparseMatIterator(SparseMat _m) { super((Pointer)null); allocate(_m); }
    private native void allocate(SparseMat _m);
    /** the full constructor setting the iterator to the specified sparse matrix element */
    
    /** the copy constructor */
    public SparseMatIterator(@Const @ByRef SparseMatIterator it) { super((Pointer)null); allocate(it); }
    private native void allocate(@Const @ByRef SparseMatIterator it);

    /** the assignment operator */
    public native @ByRef @Name("operator =") SparseMatIterator put(@Const @ByRef SparseMatIterator it);
    /** returns read-write reference to the current sparse matrix element */
    /** returns pointer to the current sparse matrix node. it.node->idx is the index of the current element (do not modify it!) */
    public native SparseMat.Node node();

    /** moves iterator to the next element */
    public native @ByRef @Name("operator ++") SparseMatIterator increment();
    /** moves iterator to the next element */
    public native @ByVal @Name("operator ++") SparseMatIterator increment(int arg0);
}
