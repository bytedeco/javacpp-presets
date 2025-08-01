// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


// #if defined(MIDL_PASS)
// #else // MIDL_PASS
@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class ULARGE_INTEGER extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public ULARGE_INTEGER() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public ULARGE_INTEGER(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public ULARGE_INTEGER(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public ULARGE_INTEGER position(long position) {
        return (ULARGE_INTEGER)super.position(position);
    }
    @Override public ULARGE_INTEGER getPointer(long i) {
        return new ULARGE_INTEGER((Pointer)this).offsetAddress(i);
    }

        public native @Cast("DWORD") int LowPart(); public native ULARGE_INTEGER LowPart(int setter);
        public native @Cast("DWORD") int HighPart(); public native ULARGE_INTEGER HighPart(int setter); 
        @Name("u.LowPart") public native @Cast("DWORD") int u_LowPart(); public native ULARGE_INTEGER u_LowPart(int setter);
        @Name("u.HighPart") public native @Cast("DWORD") int u_HighPart(); public native ULARGE_INTEGER u_HighPart(int setter);
    public native @Cast("ULONGLONG") long QuadPart(); public native ULARGE_INTEGER QuadPart(long setter);
}
