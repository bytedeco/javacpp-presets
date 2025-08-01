// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class HW_PROFILE_INFOW extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public HW_PROFILE_INFOW() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public HW_PROFILE_INFOW(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public HW_PROFILE_INFOW(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public HW_PROFILE_INFOW position(long position) {
        return (HW_PROFILE_INFOW)super.position(position);
    }
    @Override public HW_PROFILE_INFOW getPointer(long i) {
        return new HW_PROFILE_INFOW((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int dwDockInfo(); public native HW_PROFILE_INFOW dwDockInfo(int setter);
    public native @Cast("WCHAR") char szHwProfileGuid(int i); public native HW_PROFILE_INFOW szHwProfileGuid(int i, char setter);
    @MemberGetter public native @Cast("WCHAR*") CharPointer szHwProfileGuid();
    public native @Cast("WCHAR") char szHwProfileName(int i); public native HW_PROFILE_INFOW szHwProfileName(int i, char setter);
    @MemberGetter public native @Cast("WCHAR*") CharPointer szHwProfileName();
}
