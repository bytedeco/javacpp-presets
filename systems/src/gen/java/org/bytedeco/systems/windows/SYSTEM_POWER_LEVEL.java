// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


// system battery drain policies
@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class SYSTEM_POWER_LEVEL extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SYSTEM_POWER_LEVEL() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public SYSTEM_POWER_LEVEL(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SYSTEM_POWER_LEVEL(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public SYSTEM_POWER_LEVEL position(long position) {
        return (SYSTEM_POWER_LEVEL)super.position(position);
    }
    @Override public SYSTEM_POWER_LEVEL getPointer(long i) {
        return new SYSTEM_POWER_LEVEL((Pointer)this).offsetAddress(i);
    }

    public native @Cast("BOOLEAN") boolean Enable(); public native SYSTEM_POWER_LEVEL Enable(boolean setter);
    public native @Cast("BYTE") byte Spare(int i); public native SYSTEM_POWER_LEVEL Spare(int i, byte setter);
    @MemberGetter public native @Cast("BYTE*") BytePointer Spare();
    public native @Cast("DWORD") int BatteryLevel(); public native SYSTEM_POWER_LEVEL BatteryLevel(int setter);
    public native @ByRef POWER_ACTION_POLICY PowerPolicy(); public native SYSTEM_POWER_LEVEL PowerPolicy(POWER_ACTION_POLICY setter);
    public native @Cast("SYSTEM_POWER_STATE") int MinSystemState(); public native SYSTEM_POWER_LEVEL MinSystemState(int setter);
}
