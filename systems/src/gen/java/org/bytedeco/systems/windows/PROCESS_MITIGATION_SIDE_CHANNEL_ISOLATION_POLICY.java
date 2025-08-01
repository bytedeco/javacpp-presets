// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;


@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY position(long position) {
        return (PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY)super.position(position);
    }
    @Override public PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY getPointer(long i) {
        return new PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY((Pointer)this).offsetAddress(i);
    }

        public native @Cast("DWORD") int Flags(); public native PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY Flags(int setter);

            //
            // Prevent branch target pollution cross-SMT-thread in user mode.
            //

            public native @Cast("DWORD") @NoOffset int SmtBranchTargetIsolation(); public native PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY SmtBranchTargetIsolation(int setter);

            //
            // Isolate this process into a distinct security domain, even from
            // other processes running as the same security context.  This
            // prevents branch target injection cross-process (normally such
            // branch target injection is only inhibited across different
            // security contexts).
            //
            // Page combining is limited to processes within the same security
            // domain.  This flag thus also effectively limits the process to
            // only being able to combine internally to the process itself,
            // except for common pages (unless further restricted by the
            // DisablePageCombine policy).
            //

            public native @Cast("DWORD") @NoOffset int IsolateSecurityDomain(); public native PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY IsolateSecurityDomain(int setter);

            //
            // Disable all page combining for this process, even internally to
            // the process itself, except for common pages (zeroes or ones).
            //

            public native @Cast("DWORD") @NoOffset int DisablePageCombine(); public native PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY DisablePageCombine(int setter);

            //
            // Memory Disambiguation Disable.
            //

            public native @Cast("DWORD") @NoOffset int SpeculativeStoreBypassDisable(); public native PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY SpeculativeStoreBypassDisable(int setter);

            //
            // Prevent this process' threads from being scheduled on the same
            // core as threads outside its security domain.
            //

            public native @Cast("DWORD") @NoOffset int RestrictCoreSharing(); public native PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY RestrictCoreSharing(int setter);

            public native @Cast("DWORD") @NoOffset int ReservedFlags(); public native PROCESS_MITIGATION_SIDE_CHANNEL_ISOLATION_POLICY ReservedFlags(int setter);  
}
