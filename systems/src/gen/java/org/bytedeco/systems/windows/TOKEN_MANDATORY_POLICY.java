// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;

// #endif

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class TOKEN_MANDATORY_POLICY extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public TOKEN_MANDATORY_POLICY() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public TOKEN_MANDATORY_POLICY(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public TOKEN_MANDATORY_POLICY(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public TOKEN_MANDATORY_POLICY position(long position) {
        return (TOKEN_MANDATORY_POLICY)super.position(position);
    }
    @Override public TOKEN_MANDATORY_POLICY getPointer(long i) {
        return new TOKEN_MANDATORY_POLICY((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int Policy(); public native TOKEN_MANDATORY_POLICY Policy(int setter);
}
