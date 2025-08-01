// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class SYSTEM_POWER_SOURCE_STATE extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SYSTEM_POWER_SOURCE_STATE() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public SYSTEM_POWER_SOURCE_STATE(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SYSTEM_POWER_SOURCE_STATE(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public SYSTEM_POWER_SOURCE_STATE position(long position) {
        return (SYSTEM_POWER_SOURCE_STATE)super.position(position);
    }
    @Override public SYSTEM_POWER_SOURCE_STATE getPointer(long i) {
        return new SYSTEM_POWER_SOURCE_STATE((Pointer)this).offsetAddress(i);
    }

    public native @ByRef SYSTEM_BATTERY_STATE BatteryState(); public native SYSTEM_POWER_SOURCE_STATE BatteryState(SYSTEM_BATTERY_STATE setter);
    public native @Cast("DWORD") int InstantaneousPeakPower(); public native SYSTEM_POWER_SOURCE_STATE InstantaneousPeakPower(int setter);
    public native @Cast("DWORD") int InstantaneousPeakPeriod(); public native SYSTEM_POWER_SOURCE_STATE InstantaneousPeakPeriod(int setter);
    public native @Cast("DWORD") int SustainablePeakPower(); public native SYSTEM_POWER_SOURCE_STATE SustainablePeakPower(int setter);
    public native @Cast("DWORD") int SustainablePeakPeriod(); public native SYSTEM_POWER_SOURCE_STATE SustainablePeakPeriod(int setter);
    public native @Cast("DWORD") int PeakPower(); public native SYSTEM_POWER_SOURCE_STATE PeakPower(int setter);
    public native @Cast("DWORD") int MaxOutputPower(); public native SYSTEM_POWER_SOURCE_STATE MaxOutputPower(int setter);
    public native @Cast("DWORD") int MaxInputPower(); public native SYSTEM_POWER_SOURCE_STATE MaxInputPower(int setter);
    public native @Cast("LONG") int BatteryRateInCurrent(); public native SYSTEM_POWER_SOURCE_STATE BatteryRateInCurrent(int setter);
    public native @Cast("DWORD") int BatteryVoltage(); public native SYSTEM_POWER_SOURCE_STATE BatteryVoltage(int setter);
}
