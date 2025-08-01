// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


// #ifdef UNICODE
// #else
// #endif // !UNICODE

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class MODULEINFO extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public MODULEINFO() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public MODULEINFO(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MODULEINFO(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public MODULEINFO position(long position) {
        return (MODULEINFO)super.position(position);
    }
    @Override public MODULEINFO getPointer(long i) {
        return new MODULEINFO((Pointer)this).offsetAddress(i);
    }

    public native @Cast("LPVOID") Pointer lpBaseOfDll(); public native MODULEINFO lpBaseOfDll(Pointer setter);
    public native @Cast("DWORD") int SizeOfImage(); public native MODULEINFO SizeOfImage(int setter);
    public native @Cast("LPVOID") Pointer EntryPoint(); public native MODULEINFO EntryPoint(Pointer setter);
}
