// Targeted by JavaCPP version 1.5.7: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


@Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btShortIntIndexTripletData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public btShortIntIndexTripletData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public btShortIntIndexTripletData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btShortIntIndexTripletData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public btShortIntIndexTripletData position(long position) {
        return (btShortIntIndexTripletData)super.position(position);
    }
    @Override public btShortIntIndexTripletData getPointer(long i) {
        return new btShortIntIndexTripletData((Pointer)this).offsetAddress(i);
    }

	public native short m_values(int i); public native btShortIntIndexTripletData m_values(int i, short setter);
	@MemberGetter public native ShortPointer m_values();
	public native @Cast("char") byte m_pad(int i); public native btShortIntIndexTripletData m_pad(int i, byte setter);
	@MemberGetter public native @Cast("char*") BytePointer m_pad();
}