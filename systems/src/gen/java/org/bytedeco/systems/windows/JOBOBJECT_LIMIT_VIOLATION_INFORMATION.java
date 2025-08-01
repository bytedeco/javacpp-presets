// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


//
//

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class JOBOBJECT_LIMIT_VIOLATION_INFORMATION extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public JOBOBJECT_LIMIT_VIOLATION_INFORMATION() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public JOBOBJECT_LIMIT_VIOLATION_INFORMATION(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public JOBOBJECT_LIMIT_VIOLATION_INFORMATION(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public JOBOBJECT_LIMIT_VIOLATION_INFORMATION position(long position) {
        return (JOBOBJECT_LIMIT_VIOLATION_INFORMATION)super.position(position);
    }
    @Override public JOBOBJECT_LIMIT_VIOLATION_INFORMATION getPointer(long i) {
        return new JOBOBJECT_LIMIT_VIOLATION_INFORMATION((Pointer)this).offsetAddress(i);
    }

    public native @Cast("DWORD") int LimitFlags(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION LimitFlags(int setter);
    public native @Cast("DWORD") int ViolationLimitFlags(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION ViolationLimitFlags(int setter);
    public native @Cast("DWORD64") long IoReadBytes(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION IoReadBytes(long setter);
    public native @Cast("DWORD64") long IoReadBytesLimit(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION IoReadBytesLimit(long setter);
    public native @Cast("DWORD64") long IoWriteBytes(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION IoWriteBytes(long setter);
    public native @Cast("DWORD64") long IoWriteBytesLimit(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION IoWriteBytesLimit(long setter);
    public native @ByRef LARGE_INTEGER PerJobUserTime(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION PerJobUserTime(LARGE_INTEGER setter);
    public native @ByRef LARGE_INTEGER PerJobUserTimeLimit(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION PerJobUserTimeLimit(LARGE_INTEGER setter);
    public native @Cast("DWORD64") long JobMemory(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION JobMemory(long setter);
    public native @Cast("DWORD64") long JobMemoryLimit(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION JobMemoryLimit(long setter);
    public native @Cast("JOBOBJECT_RATE_CONTROL_TOLERANCE") int RateControlTolerance(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION RateControlTolerance(int setter);
    public native @Cast("JOBOBJECT_RATE_CONTROL_TOLERANCE") int RateControlToleranceLimit(); public native JOBOBJECT_LIMIT_VIOLATION_INFORMATION RateControlToleranceLimit(int setter);
}
