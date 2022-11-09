// Targeted by JavaCPP version 1.5.8: DO NOT EDIT THIS FILE

package org.bytedeco.cuda.cudart;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.cuda.global.cudart.*;

/** CUDA user object for graphs */
@Opaque @Properties(inherit = org.bytedeco.cuda.presets.cudart.class)
public class CUuserObject_st extends Pointer {
    /** Empty constructor. Calls {@code super((Pointer)null)}. */
    public CUuserObject_st() { super((Pointer)null); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CUuserObject_st(Pointer p) { super(p); }
}