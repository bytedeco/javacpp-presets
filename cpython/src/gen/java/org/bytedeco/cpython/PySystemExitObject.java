// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cpython;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cpython.global.python.*;


@Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class PySystemExitObject extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PySystemExitObject() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public PySystemExitObject(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PySystemExitObject(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public PySystemExitObject position(long position) {
        return (PySystemExitObject)super.position(position);
    }
    @Override public PySystemExitObject getPointer(long i) {
        return new PySystemExitObject((Pointer)this).offsetAddress(i);
    }

    public native @ByRef PyObject ob_base(); public native PySystemExitObject ob_base(PyObject setter); public native PyObject dict(); public native PySystemExitObject dict(PyObject setter);
             public native PyObject args(); public native PySystemExitObject args(PyObject setter); public native PyObject notes(); public native PySystemExitObject notes(PyObject setter); public native PyObject traceback(); public native PySystemExitObject traceback(PyObject setter);
             public native PyObject context(); public native PySystemExitObject context(PyObject setter); public native PyObject cause(); public native PySystemExitObject cause(PyObject setter);
             public native @Cast("char") byte suppress_context(); public native PySystemExitObject suppress_context(byte setter);
    public native PyObject code(); public native PySystemExitObject code(PyObject setter);
}
