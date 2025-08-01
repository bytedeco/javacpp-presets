// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX position(long position) {
        return (_SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)super.position(position);
    }
    @Override public _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX getPointer(long i) {
        return new _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX((Pointer)this).offsetAddress(i);
    }

    public native @Cast("LOGICAL_PROCESSOR_RELATIONSHIP") int Relationship(); public native _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX Relationship(int setter);
    public native @Cast("DWORD") int Size(); public native _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX Size(int setter);
        public native @ByRef PROCESSOR_RELATIONSHIP Processor(); public native _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX Processor(PROCESSOR_RELATIONSHIP setter);
        public native @ByRef NUMA_NODE_RELATIONSHIP NumaNode(); public native _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX NumaNode(NUMA_NODE_RELATIONSHIP setter);
        public native @ByRef CACHE_RELATIONSHIP Cache(); public native _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX Cache(CACHE_RELATIONSHIP setter);
        public native @ByRef GROUP_RELATIONSHIP Group(); public native _SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX Group(GROUP_RELATIONSHIP setter); 
}
