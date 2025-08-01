// Targeted by JavaCPP version 1.5.13-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.systems.windows;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.systems.global.windows.*;

@Properties(inherit = org.bytedeco.systems.presets.windows.class)
public class SID extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public SID() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public SID(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public SID(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public SID position(long position) {
        return (SID)super.position(position);
    }
    @Override public SID getPointer(long i) {
        return new SID((Pointer)this).offsetAddress(i);
    }

   public native @Cast("BYTE") byte Revision(); public native SID Revision(byte setter);
   public native @Cast("BYTE") byte SubAuthorityCount(); public native SID SubAuthorityCount(byte setter);
   public native @ByRef SID_IDENTIFIER_AUTHORITY IdentifierAuthority(); public native SID IdentifierAuthority(SID_IDENTIFIER_AUTHORITY setter);
// #ifdef MIDL_PASS
// #else // MIDL_PASS
   public native @Cast("DWORD") int SubAuthority(int i); public native SID SubAuthority(int i, int setter);
   @MemberGetter public native @Cast("DWORD*") IntPointer SubAuthority();
// #endif // MIDL_PASS
}
