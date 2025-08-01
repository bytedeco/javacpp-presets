// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class XSTATE_CONTEXT extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public XSTATE_CONTEXT() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public XSTATE_CONTEXT(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public XSTATE_CONTEXT(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public XSTATE_CONTEXT position(long position) {
        return (XSTATE_CONTEXT)super.position(position);
    }
    @Override public XSTATE_CONTEXT getPointer(long i) {
        return new XSTATE_CONTEXT((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD64") long Mask(); public native XSTATE_CONTEXT Mask(long setter);
    public native @Cast("DWORD") int Length(); public native XSTATE_CONTEXT Length(int setter);
    public native @Cast("BYTE") byte Flags(); public native XSTATE_CONTEXT Flags(byte setter);
    public native @Cast("BYTE") byte Reserved0(int i); public native XSTATE_CONTEXT Reserved0(int i, byte setter);
    @MemberGetter public native @Cast("BYTE*") BytePointer Reserved0();
    public native @Cast("PXSAVE_AREA") XSAVE_AREA Area(); public native XSTATE_CONTEXT Area(XSAVE_AREA setter);

// #if defined(_X86_)
// #endif

    public native @Cast("PVOID") Pointer Buffer(); public native XSTATE_CONTEXT Buffer(Pointer setter);

// #if defined(_X86_)
// #endif

}
