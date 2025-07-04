// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.cpython;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cpython.global.python.*;

@Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class PyOS_ReadlineFunctionPointer_Pointer_Pointer_BytePointer extends FunctionPointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public    PyOS_ReadlineFunctionPointer_Pointer_Pointer_BytePointer(Pointer p) { super(p); }
    protected PyOS_ReadlineFunctionPointer_Pointer_Pointer_BytePointer() { allocate(); }
    private native void allocate();
    public native @Cast("char*") BytePointer call(@Cast("FILE*") Pointer arg0, @Cast("FILE*") Pointer arg1, @Cast("const char*") BytePointer arg2);
}
