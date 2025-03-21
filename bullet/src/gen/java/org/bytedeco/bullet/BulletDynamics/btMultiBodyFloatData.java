// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

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


/**do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64 */
@Properties(inherit = org.bytedeco.bullet.presets.BulletDynamics.class)
public class btMultiBodyFloatData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public btMultiBodyFloatData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public btMultiBodyFloatData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btMultiBodyFloatData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public btMultiBodyFloatData position(long position) {
        return (btMultiBodyFloatData)super.position(position);
    }
    @Override public btMultiBodyFloatData getPointer(long i) {
        return new btMultiBodyFloatData((Pointer)this).offsetAddress(i);
    }

	public native @ByRef btVector3FloatData m_baseWorldPosition(); public native btMultiBodyFloatData m_baseWorldPosition(btVector3FloatData setter);
	public native @ByRef btQuaternionFloatData m_baseWorldOrientation(); public native btMultiBodyFloatData m_baseWorldOrientation(btQuaternionFloatData setter);
	public native @ByRef btVector3FloatData m_baseLinearVelocity(); public native btMultiBodyFloatData m_baseLinearVelocity(btVector3FloatData setter);
	public native @ByRef btVector3FloatData m_baseAngularVelocity(); public native btMultiBodyFloatData m_baseAngularVelocity(btVector3FloatData setter);

	public native @ByRef btVector3FloatData m_baseInertia(); public native btMultiBodyFloatData m_baseInertia(btVector3FloatData setter);  // inertia of the base (in local frame; diagonal)
	public native float m_baseMass(); public native btMultiBodyFloatData m_baseMass(float setter);
	public native int m_numLinks(); public native btMultiBodyFloatData m_numLinks(int setter);

	public native @Cast("char*") BytePointer m_baseName(); public native btMultiBodyFloatData m_baseName(BytePointer setter);
	public native btMultiBodyLinkFloatData m_links(); public native btMultiBodyFloatData m_links(btMultiBodyLinkFloatData setter);
	public native btCollisionObjectFloatData m_baseCollider(); public native btMultiBodyFloatData m_baseCollider(btCollisionObjectFloatData setter);
}
