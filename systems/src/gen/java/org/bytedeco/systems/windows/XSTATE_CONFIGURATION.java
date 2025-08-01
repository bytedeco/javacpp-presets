// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class XSTATE_CONFIGURATION extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public XSTATE_CONFIGURATION() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public XSTATE_CONFIGURATION(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public XSTATE_CONFIGURATION(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public XSTATE_CONFIGURATION position(long position) {
        return (XSTATE_CONFIGURATION)super.position(position);
    }
    @Override public XSTATE_CONFIGURATION getPointer(long i) {
        return new XSTATE_CONFIGURATION((Pointer)this).offsetAddress(i);
    }

    // Mask of all enabled features
    public native @Cast("DWORD64") long EnabledFeatures(); public native XSTATE_CONFIGURATION EnabledFeatures(long setter);

    // Mask of volatile enabled features
    public native @Cast("DWORD64") long EnabledVolatileFeatures(); public native XSTATE_CONFIGURATION EnabledVolatileFeatures(long setter);

    // Total size of the save area for user states
    public native @Cast("DWORD") int Size(); public native XSTATE_CONFIGURATION Size(int setter);

    // Control Flags
        public native @Cast("DWORD") int ControlFlags(); public native XSTATE_CONFIGURATION ControlFlags(int setter);
            public native @Cast("DWORD") @NoOffset int OptimizedSave(); public native XSTATE_CONFIGURATION OptimizedSave(int setter);
            public native @Cast("DWORD") @NoOffset int CompactionEnabled(); public native XSTATE_CONFIGURATION CompactionEnabled(int setter);
            public native @Cast("DWORD") @NoOffset int ExtendedFeatureDisable(); public native XSTATE_CONFIGURATION ExtendedFeatureDisable(int setter);  

    // List of features
    public native @ByRef XSTATE_FEATURE Features(int i); public native XSTATE_CONFIGURATION Features(int i, XSTATE_FEATURE setter);
    @MemberGetter public native XSTATE_FEATURE Features();

    // Mask of all supervisor features
    public native @Cast("DWORD64") long EnabledSupervisorFeatures(); public native XSTATE_CONFIGURATION EnabledSupervisorFeatures(long setter);

    // Mask of features that require start address to be 64 byte aligned
    public native @Cast("DWORD64") long AlignedFeatures(); public native XSTATE_CONFIGURATION AlignedFeatures(long setter);

    // Total size of the save area for user and supervisor states
    public native @Cast("DWORD") int AllFeatureSize(); public native XSTATE_CONFIGURATION AllFeatureSize(int setter);

    // List which holds size of each user and supervisor state supported by CPU
    public native @Cast("DWORD") int AllFeatures(int i); public native XSTATE_CONFIGURATION AllFeatures(int i, int setter);
    @MemberGetter public native @Cast("DWORD*") IntPointer AllFeatures();

    // Mask of all supervisor features that are exposed to user-mode
    public native @Cast("DWORD64") long EnabledUserVisibleSupervisorFeatures(); public native XSTATE_CONFIGURATION EnabledUserVisibleSupervisorFeatures(long setter);

    // Mask of features that can be disabled via XFD
    public native @Cast("DWORD64") long ExtendedFeatureDisableFeatures(); public native XSTATE_CONFIGURATION ExtendedFeatureDisableFeatures(long setter);

    // Total size of the save area for non-large user and supervisor states
    public native @Cast("DWORD") int AllNonLargeFeatureSize(); public native XSTATE_CONFIGURATION AllNonLargeFeatureSize(int setter);

    // The maximum supported ARM64 SVE vector length that can be used in the
    // current environment, in bytes.
    public native @Cast("WORD") short MaxSveVectorLength(); public native XSTATE_CONFIGURATION MaxSveVectorLength(short setter);

    public native @Cast("WORD") short Spare1(); public native XSTATE_CONFIGURATION Spare1(short setter);

}
