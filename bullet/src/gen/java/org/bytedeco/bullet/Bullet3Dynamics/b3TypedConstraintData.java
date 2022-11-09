// Targeted by JavaCPP version 1.5.8: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.Bullet3Dynamics;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.Bullet3Common.*;
import static org.bytedeco.bullet.global.Bullet3Common.*;
import org.bytedeco.bullet.Bullet3Collision.*;
import static org.bytedeco.bullet.global.Bullet3Collision.*;

import static org.bytedeco.bullet.global.Bullet3Dynamics.*;


// clang-format off
/**do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64 */
@Properties(inherit = org.bytedeco.bullet.presets.Bullet3Dynamics.class)
public class b3TypedConstraintData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public b3TypedConstraintData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public b3TypedConstraintData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public b3TypedConstraintData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public b3TypedConstraintData position(long position) {
        return (b3TypedConstraintData)super.position(position);
    }
    @Override public b3TypedConstraintData getPointer(long i) {
        return new b3TypedConstraintData((Pointer)this).offsetAddress(i);
    }

	public native int m_bodyA(); public native b3TypedConstraintData m_bodyA(int setter);
	public native int m_bodyB(); public native b3TypedConstraintData m_bodyB(int setter);
	public native @Cast("char*") BytePointer m_name(); public native b3TypedConstraintData m_name(BytePointer setter);

	public native int m_objectType(); public native b3TypedConstraintData m_objectType(int setter);
	public native int m_userConstraintType(); public native b3TypedConstraintData m_userConstraintType(int setter);
	public native int m_userConstraintId(); public native b3TypedConstraintData m_userConstraintId(int setter);
	public native int m_needsFeedback(); public native b3TypedConstraintData m_needsFeedback(int setter);

	public native float m_appliedImpulse(); public native b3TypedConstraintData m_appliedImpulse(float setter);
	public native float m_dbgDrawSize(); public native b3TypedConstraintData m_dbgDrawSize(float setter);

	public native int m_disableCollisionsBetweenLinkedBodies(); public native b3TypedConstraintData m_disableCollisionsBetweenLinkedBodies(int setter);
	public native int m_overrideNumSolverIterations(); public native b3TypedConstraintData m_overrideNumSolverIterations(int setter);

	public native float m_breakingImpulseThreshold(); public native b3TypedConstraintData m_breakingImpulseThreshold(float setter);
	public native int m_isEnabled(); public native b3TypedConstraintData m_isEnabled(int setter);
	
}