// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cpython;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cpython.global.python.*;


@Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class PyAttributeErrorObject extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PyAttributeErrorObject() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public PyAttributeErrorObject(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PyAttributeErrorObject(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public PyAttributeErrorObject position(long position) {
        return (PyAttributeErrorObject)super.position(position);
    }
    @Override public PyAttributeErrorObject getPointer(long i) {
        return new PyAttributeErrorObject((Pointer)this).offsetAddress(i);
    }

    public native @ByRef PyObject ob_base(); public native PyAttributeErrorObject ob_base(PyObject setter); public native PyObject dict(); public native PyAttributeErrorObject dict(PyObject setter);
             public native PyObject args(); public native PyAttributeErrorObject args(PyObject setter); public native PyObject notes(); public native PyAttributeErrorObject notes(PyObject setter); public native PyObject traceback(); public native PyAttributeErrorObject traceback(PyObject setter);
             public native PyObject context(); public native PyAttributeErrorObject context(PyObject setter); public native PyObject cause(); public native PyAttributeErrorObject cause(PyObject setter);
             public native @Cast("char") byte suppress_context(); public native PyAttributeErrorObject suppress_context(byte setter);
    public native PyObject obj(); public native PyAttributeErrorObject obj(PyObject setter);
    public native PyObject name(); public native PyAttributeErrorObject name(PyObject setter);
}
