// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class MESSAGE_RESOURCE_BLOCK extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public MESSAGE_RESOURCE_BLOCK() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public MESSAGE_RESOURCE_BLOCK(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public MESSAGE_RESOURCE_BLOCK(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public MESSAGE_RESOURCE_BLOCK position(long position) {
        return (MESSAGE_RESOURCE_BLOCK)super.position(position);
    }
    @Override public MESSAGE_RESOURCE_BLOCK getPointer(long i) {
        return new MESSAGE_RESOURCE_BLOCK((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int LowId(); public native MESSAGE_RESOURCE_BLOCK LowId(int setter);
    public native @Cast("DWORD") int HighId(); public native MESSAGE_RESOURCE_BLOCK HighId(int setter);
    public native @Cast("DWORD") int OffsetToEntries(); public native MESSAGE_RESOURCE_BLOCK OffsetToEntries(int setter);
}
