// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.LinearMath;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.bullet.global.LinearMath.*;


@Properties(inherit = org.bytedeco.bullet.presets.LinearMath.class)
public class btPointerUid extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public btPointerUid() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public btPointerUid(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btPointerUid(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public btPointerUid position(long position) {
        return (btPointerUid)super.position(position);
    }
    @Override public btPointerUid getPointer(long i) {
        return new btPointerUid((Pointer)this).offsetAddress(i);
    }

		public native Pointer m_ptr(); public native btPointerUid m_ptr(Pointer setter);
		public native int m_uniqueIds(int i); public native btPointerUid m_uniqueIds(int i, int setter);
		@MemberGetter public native IntPointer m_uniqueIds();
}
