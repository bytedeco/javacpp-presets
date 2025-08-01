// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class CACHE_RELATIONSHIP extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public CACHE_RELATIONSHIP() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public CACHE_RELATIONSHIP(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public CACHE_RELATIONSHIP(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public CACHE_RELATIONSHIP position(long position) {
        return (CACHE_RELATIONSHIP)super.position(position);
    }
    @Override public CACHE_RELATIONSHIP getPointer(long i) {
        return new CACHE_RELATIONSHIP((Pointer)this).offsetAddress(i);
    }

    public native @Cast("BYTE") byte Level(); public native CACHE_RELATIONSHIP Level(byte setter);
    public native @Cast("BYTE") byte Associativity(); public native CACHE_RELATIONSHIP Associativity(byte setter);
    public native @Cast("WORD") short LineSize(); public native CACHE_RELATIONSHIP LineSize(short setter);
    public native @Cast("DWORD") int CacheSize(); public native CACHE_RELATIONSHIP CacheSize(int setter);
    public native @Cast("PROCESSOR_CACHE_TYPE") int Type(); public native CACHE_RELATIONSHIP Type(int setter);
    public native @Cast("BYTE") byte Reserved(int i); public native CACHE_RELATIONSHIP Reserved(int i, byte setter);
    @MemberGetter public native @Cast("BYTE*") BytePointer Reserved();
    public native @Cast("WORD") short GroupCount(); public native CACHE_RELATIONSHIP GroupCount(short setter);
        public native @ByRef GROUP_AFFINITY GroupMask(); public native CACHE_RELATIONSHIP GroupMask(GROUP_AFFINITY setter);
        public native @ByRef GROUP_AFFINITY GroupMasks(int i); public native CACHE_RELATIONSHIP GroupMasks(int i, GROUP_AFFINITY setter);
        @MemberGetter public native GROUP_AFFINITY GroupMasks(); 
}
