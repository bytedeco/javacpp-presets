// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


// begin_ntoshvp

//
// Structure to represent a group-specific affinity, such as that of a
// thread.  Specifies the group number and the affinity within that group.
//

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class GROUP_AFFINITY extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public GROUP_AFFINITY() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public GROUP_AFFINITY(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GROUP_AFFINITY(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public GROUP_AFFINITY position(long position) {
        return (GROUP_AFFINITY)super.position(position);
    }
    @Override public GROUP_AFFINITY getPointer(long i) {
        return new GROUP_AFFINITY((Pointer)this).offsetAddress(i);
    }

    public native @Cast("KAFFINITY") long Mask(); public native GROUP_AFFINITY Mask(long setter);
    public native @Cast("WORD") short Group(); public native GROUP_AFFINITY Group(short setter);
    public native @Cast("WORD") short Reserved(int i); public native GROUP_AFFINITY Reserved(int i, short setter);
    @MemberGetter public native @Cast("WORD*") ShortPointer Reserved();
}
