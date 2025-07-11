// Targeted by JavaCPP version 1.5.12: DO NOT EDIT THIS FILE

package org.bytedeco.ffmpeg.avutil;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.ffmpeg.global.avutil.*;


@Properties(inherit = org.bytedeco.ffmpeg.presets.avutil.class)
public class AVBlowfish extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public AVBlowfish() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public AVBlowfish(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public AVBlowfish(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public AVBlowfish position(long position) {
        return (AVBlowfish)super.position(position);
    }
    @Override public AVBlowfish getPointer(long i) {
        return new AVBlowfish((Pointer)this).offsetAddress(i);
    }

    public native @Cast("uint32_t") int p(int i); public native AVBlowfish p(int i, int setter);
    @MemberGetter public native @Cast("uint32_t*") IntPointer p();
    public native @Cast("uint32_t") int s(int i, int j); public native AVBlowfish s(int i, int j, int setter);
    @MemberGetter public native @Cast("uint32_t(* /*[4]*/ )[256]") IntPointer s();
}
