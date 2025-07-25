// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cpython;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cpython.global.python.*;

// #endif

/* Slice object interface */

/*

A slice object containing start, stop, and step data members (the
names are from range).  After much talk with Guido, it was decided to
let these be any arbitrary python type.  Py_None stands for omitted values.
*/
// #ifndef Py_LIMITED_API
@Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class PySliceObject extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PySliceObject() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public PySliceObject(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PySliceObject(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public PySliceObject position(long position) {
        return (PySliceObject)super.position(position);
    }
    @Override public PySliceObject getPointer(long i) {
        return new PySliceObject((Pointer)this).offsetAddress(i);
    }

    public native @ByRef PyObject ob_base(); public native PySliceObject ob_base(PyObject setter);
    public native PyObject start(); public native PySliceObject start(PyObject setter);
    public native PyObject stop(); public native PySliceObject stop(PyObject setter);
    public native PyObject step(); public native PySliceObject step(PyObject setter);      /* not NULL */
}
