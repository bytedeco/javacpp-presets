// Targeted by JavaCPP version 1.5.8-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.Bullet3OpenCL;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.Bullet3Common.*;
import static org.bytedeco.bullet.global.Bullet3Common.*;
import org.bytedeco.bullet.Bullet3Collision.*;
import static org.bytedeco.bullet.global.Bullet3Collision.*;
import org.bytedeco.bullet.Bullet3Dynamics.*;
import static org.bytedeco.bullet.global.Bullet3Dynamics.*;

import static org.bytedeco.bullet.global.Bullet3OpenCL.*;


@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.Bullet3OpenCL.class)
public class b3OpenCLPlatformInfo extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public b3OpenCLPlatformInfo(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public b3OpenCLPlatformInfo(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public b3OpenCLPlatformInfo position(long position) {
        return (b3OpenCLPlatformInfo)super.position(position);
    }
    @Override public b3OpenCLPlatformInfo getPointer(long i) {
        return new b3OpenCLPlatformInfo((Pointer)this).offsetAddress(i);
    }

	public native @Cast("char") byte m_platformVendor(int i); public native b3OpenCLPlatformInfo m_platformVendor(int i, byte setter);
	@MemberGetter public native @Cast("char*") BytePointer m_platformVendor();
	public native @Cast("char") byte m_platformName(int i); public native b3OpenCLPlatformInfo m_platformName(int i, byte setter);
	@MemberGetter public native @Cast("char*") BytePointer m_platformName();
	public native @Cast("char") byte m_platformVersion(int i); public native b3OpenCLPlatformInfo m_platformVersion(int i, byte setter);
	@MemberGetter public native @Cast("char*") BytePointer m_platformVersion();

	public b3OpenCLPlatformInfo() { super((Pointer)null); allocate(); }
	private native void allocate();
}