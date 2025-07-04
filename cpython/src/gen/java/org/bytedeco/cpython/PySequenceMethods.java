// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cpython;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cpython.global.python.*;


@Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class PySequenceMethods extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PySequenceMethods() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public PySequenceMethods(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PySequenceMethods(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public PySequenceMethods position(long position) {
        return (PySequenceMethods)super.position(position);
    }
    @Override public PySequenceMethods getPointer(long i) {
        return new PySequenceMethods((Pointer)this).offsetAddress(i);
    }

    public native lenfunc sq_length(); public native PySequenceMethods sq_length(lenfunc setter);
    public native binaryfunc sq_concat(); public native PySequenceMethods sq_concat(binaryfunc setter);
    public native ssizeargfunc sq_repeat(); public native PySequenceMethods sq_repeat(ssizeargfunc setter);
    public native ssizeargfunc sq_item(); public native PySequenceMethods sq_item(ssizeargfunc setter);
    public native Pointer was_sq_slice(); public native PySequenceMethods was_sq_slice(Pointer setter);
    public native ssizeobjargproc sq_ass_item(); public native PySequenceMethods sq_ass_item(ssizeobjargproc setter);
    public native Pointer was_sq_ass_slice(); public native PySequenceMethods was_sq_ass_slice(Pointer setter);
    public native objobjproc sq_contains(); public native PySequenceMethods sq_contains(objobjproc setter);

    public native binaryfunc sq_inplace_concat(); public native PySequenceMethods sq_inplace_concat(binaryfunc setter);
    public native ssizeargfunc sq_inplace_repeat(); public native PySequenceMethods sq_inplace_repeat(ssizeargfunc setter);
}
