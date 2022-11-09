// Targeted by JavaCPP version 1.5.8: DO NOT EDIT THIS FILE

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


//#define BTMBP2PCONSTRAINT_BLOCK_ANGULAR_MOTION_TEST

@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.BulletDynamics.class)
public class btMultiBodyPoint2Point extends btMultiBodyConstraint {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btMultiBodyPoint2Point(Pointer p) { super(p); }


	public btMultiBodyPoint2Point(btMultiBody body, int link, btRigidBody bodyB, @Const @ByRef btVector3 pivotInA, @Const @ByRef btVector3 pivotInB) { super((Pointer)null); allocate(body, link, bodyB, pivotInA, pivotInB); }
	private native void allocate(btMultiBody body, int link, btRigidBody bodyB, @Const @ByRef btVector3 pivotInA, @Const @ByRef btVector3 pivotInB);
	public btMultiBodyPoint2Point(btMultiBody bodyA, int linkA, btMultiBody bodyB, int linkB, @Const @ByRef btVector3 pivotInA, @Const @ByRef btVector3 pivotInB) { super((Pointer)null); allocate(bodyA, linkA, bodyB, linkB, pivotInA, pivotInB); }
	private native void allocate(btMultiBody bodyA, int linkA, btMultiBody bodyB, int linkB, @Const @ByRef btVector3 pivotInA, @Const @ByRef btVector3 pivotInB);

	public native void finalizeMultiDof();

	public native int getIslandIdA();
	public native int getIslandIdB();

	public native void createConstraintRows(@ByRef btMultiBodySolverConstraintArray constraintRows,
										  @ByRef btMultiBodyJacobianData data,
										  @Const @ByRef btContactSolverInfo infoGlobal);

	public native @Const @ByRef btVector3 getPivotInB();

	public native void setPivotInB(@Const @ByRef btVector3 pivotInB);

	public native void debugDraw(btIDebugDraw drawer);
}