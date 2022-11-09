// Targeted by JavaCPP version 1.5.8: DO NOT EDIT THIS FILE

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
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.Bullet3OpenCL.*;


@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.Bullet3OpenCL.class)
public class b3BufferInfoCL extends Pointer {
    static { Loader.load(); }

	//b3BufferInfoCL(){}

	//	template<typename T>
	public b3BufferInfoCL(@Cast("cl_mem") Pointer buff, @Cast("bool") boolean isReadOnly/*=false*/) { super((Pointer)null); allocate(buff, isReadOnly); }
	private native void allocate(@Cast("cl_mem") Pointer buff, @Cast("bool") boolean isReadOnly/*=false*/);
	public b3BufferInfoCL(@Cast("cl_mem") Pointer buff) { super((Pointer)null); allocate(buff); }
	private native void allocate(@Cast("cl_mem") Pointer buff);

	public native @Cast("cl_mem") Pointer m_clBuffer(); public native b3BufferInfoCL m_clBuffer(Pointer setter);
	public native @Cast("bool") boolean m_isReadOnly(); public native b3BufferInfoCL m_isReadOnly(boolean setter);
}