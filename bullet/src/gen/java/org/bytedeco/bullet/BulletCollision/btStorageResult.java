// Targeted by JavaCPP version 1.5.7: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btStorageResult extends btDiscreteCollisionDetectorInterface.Result {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btStorageResult(Pointer p) { super(p); }

	public native @ByRef btVector3 m_normalOnSurfaceB(); public native btStorageResult m_normalOnSurfaceB(btVector3 setter);
	public native @ByRef btVector3 m_closestPointInB(); public native btStorageResult m_closestPointInB(btVector3 setter);
	public native @Cast("btScalar") float m_distance(); public native btStorageResult m_distance(float setter);

	public native void addContactPoint(@Const @ByRef btVector3 normalOnBInWorld, @Const @ByRef btVector3 pointInWorld, @Cast("btScalar") float depth);
}