// Targeted by JavaCPP version 1.5.7: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


// clang-format off

/**do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64 */
@Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btCollisionObjectDoubleData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public btCollisionObjectDoubleData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public btCollisionObjectDoubleData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btCollisionObjectDoubleData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public btCollisionObjectDoubleData position(long position) {
        return (btCollisionObjectDoubleData)super.position(position);
    }
    @Override public btCollisionObjectDoubleData getPointer(long i) {
        return new btCollisionObjectDoubleData((Pointer)this).offsetAddress(i);
    }

	public native Pointer m_broadphaseHandle(); public native btCollisionObjectDoubleData m_broadphaseHandle(Pointer setter);
	public native Pointer m_collisionShape(); public native btCollisionObjectDoubleData m_collisionShape(Pointer setter);
	public native btCollisionShapeData m_rootCollisionShape(); public native btCollisionObjectDoubleData m_rootCollisionShape(btCollisionShapeData setter);
	public native @Cast("char*") BytePointer m_name(); public native btCollisionObjectDoubleData m_name(BytePointer setter);

	public native @ByRef btTransformDoubleData m_worldTransform(); public native btCollisionObjectDoubleData m_worldTransform(btTransformDoubleData setter);
	public native @ByRef btTransformDoubleData m_interpolationWorldTransform(); public native btCollisionObjectDoubleData m_interpolationWorldTransform(btTransformDoubleData setter);
	public native @ByRef btVector3DoubleData m_interpolationLinearVelocity(); public native btCollisionObjectDoubleData m_interpolationLinearVelocity(btVector3DoubleData setter);
	public native @ByRef btVector3DoubleData m_interpolationAngularVelocity(); public native btCollisionObjectDoubleData m_interpolationAngularVelocity(btVector3DoubleData setter);
	public native @ByRef btVector3DoubleData m_anisotropicFriction(); public native btCollisionObjectDoubleData m_anisotropicFriction(btVector3DoubleData setter);
	public native double m_contactProcessingThreshold(); public native btCollisionObjectDoubleData m_contactProcessingThreshold(double setter);	
	public native double m_deactivationTime(); public native btCollisionObjectDoubleData m_deactivationTime(double setter);
	public native double m_friction(); public native btCollisionObjectDoubleData m_friction(double setter);
	public native double m_rollingFriction(); public native btCollisionObjectDoubleData m_rollingFriction(double setter);
	public native double m_contactDamping(); public native btCollisionObjectDoubleData m_contactDamping(double setter);
	public native double m_contactStiffness(); public native btCollisionObjectDoubleData m_contactStiffness(double setter);
	public native double m_restitution(); public native btCollisionObjectDoubleData m_restitution(double setter);
	public native double m_hitFraction(); public native btCollisionObjectDoubleData m_hitFraction(double setter); 
	public native double m_ccdSweptSphereRadius(); public native btCollisionObjectDoubleData m_ccdSweptSphereRadius(double setter);
	public native double m_ccdMotionThreshold(); public native btCollisionObjectDoubleData m_ccdMotionThreshold(double setter);
	public native int m_hasAnisotropicFriction(); public native btCollisionObjectDoubleData m_hasAnisotropicFriction(int setter);
	public native int m_collisionFlags(); public native btCollisionObjectDoubleData m_collisionFlags(int setter);
	public native int m_islandTag1(); public native btCollisionObjectDoubleData m_islandTag1(int setter);
	public native int m_companionId(); public native btCollisionObjectDoubleData m_companionId(int setter);
	public native int m_activationState1(); public native btCollisionObjectDoubleData m_activationState1(int setter);
	public native int m_internalType(); public native btCollisionObjectDoubleData m_internalType(int setter);
	public native int m_checkCollideWith(); public native btCollisionObjectDoubleData m_checkCollideWith(int setter);
	public native int m_collisionFilterGroup(); public native btCollisionObjectDoubleData m_collisionFilterGroup(int setter);
	public native int m_collisionFilterMask(); public native btCollisionObjectDoubleData m_collisionFilterMask(int setter);
	public native int m_uniqueId(); public native btCollisionObjectDoubleData m_uniqueId(int setter);//m_uniqueId is introduced for paircache. could get rid of this, by calculating the address offset etc.
}