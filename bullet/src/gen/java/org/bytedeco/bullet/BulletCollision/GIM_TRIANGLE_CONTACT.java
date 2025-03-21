// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


/** Structure for collision */
@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class GIM_TRIANGLE_CONTACT extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public GIM_TRIANGLE_CONTACT(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public GIM_TRIANGLE_CONTACT(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public GIM_TRIANGLE_CONTACT position(long position) {
        return (GIM_TRIANGLE_CONTACT)super.position(position);
    }
    @Override public GIM_TRIANGLE_CONTACT getPointer(long i) {
        return new GIM_TRIANGLE_CONTACT((Pointer)this).offsetAddress(i);
    }

	public native @Cast("btScalar") double m_penetration_depth(); public native GIM_TRIANGLE_CONTACT m_penetration_depth(double setter);
	public native int m_point_count(); public native GIM_TRIANGLE_CONTACT m_point_count(int setter);
	public native @ByRef btVector4 m_separating_normal(); public native GIM_TRIANGLE_CONTACT m_separating_normal(btVector4 setter);
	public native @ByRef btVector3 m_points(int i); public native GIM_TRIANGLE_CONTACT m_points(int i, btVector3 setter);
	@MemberGetter public native btVector3 m_points();

	public native void copy_from(@Const @ByRef GIM_TRIANGLE_CONTACT other);

	public GIM_TRIANGLE_CONTACT() { super((Pointer)null); allocate(); }
	private native void allocate();

	public GIM_TRIANGLE_CONTACT(@Const @ByRef GIM_TRIANGLE_CONTACT other) { super((Pointer)null); allocate(other); }
	private native void allocate(@Const @ByRef GIM_TRIANGLE_CONTACT other);

	/** classify points that are closer */
	public native void merge_points(@Const @ByRef btVector4 plane,
						  @Cast("btScalar") double margin, @Const btVector3 points, int point_count);
}
