// Targeted by JavaCPP version 1.5.8-SNAPSHOT: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletDynamics;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;
import org.bytedeco.bullet.BulletCollision.*;
import static org.bytedeco.bullet.global.BulletCollision.*;

import static org.bytedeco.bullet.global.BulletDynamics.*;


@Properties(inherit = org.bytedeco.bullet.presets.BulletDynamics.class)
public class btMultiBodyJacobianData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public btMultiBodyJacobianData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public btMultiBodyJacobianData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btMultiBodyJacobianData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public btMultiBodyJacobianData position(long position) {
        return (btMultiBodyJacobianData)super.position(position);
    }
    @Override public btMultiBodyJacobianData getPointer(long i) {
        return new btMultiBodyJacobianData((Pointer)this).offsetAddress(i);
    }

	public native @ByRef btAlignedObjectArray_btScalar m_jacobians(); public native btMultiBodyJacobianData m_jacobians(btAlignedObjectArray_btScalar setter);
	public native @ByRef btAlignedObjectArray_btScalar m_deltaVelocitiesUnitImpulse(); public native btMultiBodyJacobianData m_deltaVelocitiesUnitImpulse(btAlignedObjectArray_btScalar setter);  //holds the joint-space response of the corresp. tree to the test impulse in each constraint space dimension
	public native @ByRef btAlignedObjectArray_btScalar m_deltaVelocities(); public native btMultiBodyJacobianData m_deltaVelocities(btAlignedObjectArray_btScalar setter);             //holds joint-space vectors of all the constrained trees accumulating the effect of corrective impulses applied in SI
	public native @ByRef btAlignedObjectArray_btScalar scratch_r(); public native btMultiBodyJacobianData scratch_r(btAlignedObjectArray_btScalar setter);
	public native @ByRef btAlignedObjectArray_btVector3 scratch_v(); public native btMultiBodyJacobianData scratch_v(btAlignedObjectArray_btVector3 setter);
	public native @ByRef btAlignedObjectArray_btMatrix3x3 scratch_m(); public native btMultiBodyJacobianData scratch_m(btAlignedObjectArray_btMatrix3x3 setter);
	
	public native int m_fixedBodyId(); public native btMultiBodyJacobianData m_fixedBodyId(int setter);
}