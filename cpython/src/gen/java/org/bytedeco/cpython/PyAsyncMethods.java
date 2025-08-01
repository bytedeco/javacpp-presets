// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cpython;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cpython.global.python.*;


@Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class PyAsyncMethods extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PyAsyncMethods() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public PyAsyncMethods(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PyAsyncMethods(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public PyAsyncMethods position(long position) {
        return (PyAsyncMethods)super.position(position);
    }
    @Override public PyAsyncMethods getPointer(long i) {
        return new PyAsyncMethods((Pointer)this).offsetAddress(i);
    }

    public native unaryfunc am_await(); public native PyAsyncMethods am_await(unaryfunc setter);
    public native unaryfunc am_aiter(); public native PyAsyncMethods am_aiter(unaryfunc setter);
    public native unaryfunc am_anext(); public native PyAsyncMethods am_anext(unaryfunc setter);
    public native sendfunc am_send(); public native PyAsyncMethods am_send(sendfunc setter);
}
