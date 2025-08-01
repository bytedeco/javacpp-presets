// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class IMAGE_DYNAMIC_RELOCATION64 extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public IMAGE_DYNAMIC_RELOCATION64() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public IMAGE_DYNAMIC_RELOCATION64(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public IMAGE_DYNAMIC_RELOCATION64(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public IMAGE_DYNAMIC_RELOCATION64 position(long position) {
        return (IMAGE_DYNAMIC_RELOCATION64)super.position(position);
    }
    @Override public IMAGE_DYNAMIC_RELOCATION64 getPointer(long i) {
        return new IMAGE_DYNAMIC_RELOCATION64((Pointer)this).offsetAddress(i);
    }

    public native @Cast("ULONGLONG") long Symbol(); public native IMAGE_DYNAMIC_RELOCATION64 Symbol(long setter);
    public native @Cast("DWORD") int BaseRelocSize(); public native IMAGE_DYNAMIC_RELOCATION64 BaseRelocSize(int setter);
//  IMAGE_BASE_RELOCATION BaseRelocations[0];
}
