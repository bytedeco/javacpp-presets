// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class NUMA_NODE_RELATIONSHIP extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public NUMA_NODE_RELATIONSHIP() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public NUMA_NODE_RELATIONSHIP(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public NUMA_NODE_RELATIONSHIP(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public NUMA_NODE_RELATIONSHIP position(long position) {
        return (NUMA_NODE_RELATIONSHIP)super.position(position);
    }
    @Override public NUMA_NODE_RELATIONSHIP getPointer(long i) {
        return new NUMA_NODE_RELATIONSHIP((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int NodeNumber(); public native NUMA_NODE_RELATIONSHIP NodeNumber(int setter);
    public native @Cast("BYTE") byte Reserved(int i); public native NUMA_NODE_RELATIONSHIP Reserved(int i, byte setter);
    @MemberGetter public native @Cast("BYTE*") BytePointer Reserved();
    public native @Cast("WORD") short GroupCount(); public native NUMA_NODE_RELATIONSHIP GroupCount(short setter);
        public native @ByRef GROUP_AFFINITY GroupMask(); public native NUMA_NODE_RELATIONSHIP GroupMask(GROUP_AFFINITY setter);
        public native @ByRef GROUP_AFFINITY GroupMasks(int i); public native NUMA_NODE_RELATIONSHIP GroupMasks(int i, GROUP_AFFINITY setter);
        @MemberGetter public native GROUP_AFFINITY GroupMasks(); 
}
