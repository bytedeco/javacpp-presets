// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


/**The btGhostObject can keep track of all objects that are overlapping
 * By default, this overlap is based on the AABB
 * This is useful for creating a character controller, collision sensors/triggers, explosions etc.
 * We plan on adding rayTest and other queries for the btGhostObject */
@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btGhostObject extends btCollisionObject {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btGhostObject(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public btGhostObject(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public btGhostObject position(long position) {
        return (btGhostObject)super.position(position);
    }
    @Override public btGhostObject getPointer(long i) {
        return new btGhostObject((Pointer)this).offsetAddress(i);
    }

	public btGhostObject() { super((Pointer)null); allocate(); }
	private native void allocate();

	public native void convexSweepTest(@Const btConvexShape castShape, @Const @ByRef btTransform convexFromWorld, @Const @ByRef btTransform convexToWorld, @ByRef btCollisionWorld.ConvexResultCallback resultCallback, @Cast("btScalar") double allowedCcdPenetration/*=0.f*/);
	public native void convexSweepTest(@Const btConvexShape castShape, @Const @ByRef btTransform convexFromWorld, @Const @ByRef btTransform convexToWorld, @ByRef btCollisionWorld.ConvexResultCallback resultCallback);

	public native void rayTest(@Const @ByRef btVector3 rayFromWorld, @Const @ByRef btVector3 rayToWorld, @ByRef btCollisionWorld.RayResultCallback resultCallback);

	/**this method is mainly for expert/internal use only. */
	public native void addOverlappingObjectInternal(btBroadphaseProxy otherProxy, btBroadphaseProxy thisProxy/*=0*/);
	public native void addOverlappingObjectInternal(btBroadphaseProxy otherProxy);
	/**this method is mainly for expert/internal use only. */
	public native void removeOverlappingObjectInternal(btBroadphaseProxy otherProxy, btDispatcher dispatcher, btBroadphaseProxy thisProxy/*=0*/);
	public native void removeOverlappingObjectInternal(btBroadphaseProxy otherProxy, btDispatcher dispatcher);

	public native int getNumOverlappingObjects();

	public native btCollisionObject getOverlappingObject(int index);

	public native @ByRef btCollisionObjectArray getOverlappingPairs();

	//
	// internal cast
	//

	public static native @Const btGhostObject upcast(@Const btCollisionObject colObj);
}
