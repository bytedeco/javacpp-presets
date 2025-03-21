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


/**btKinematicCharacterController is an object that supports a sliding motion in a world.
 * It uses a ghost object and convex sweep test to test for upcoming collisions. This is combined with discrete collision detection to recover from penetrations.
 * Interaction between btKinematicCharacterController and dynamic rigid bodies needs to be explicity implemented by the user. */
@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.BulletDynamics.class)
public class btKinematicCharacterController extends btCharacterControllerInterface {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btKinematicCharacterController(Pointer p) { super(p); }


	public btKinematicCharacterController(btPairCachingGhostObject ghostObject, btConvexShape convexShape, @Cast("btScalar") double stepHeight, @Const @ByRef(nullValue = "btVector3(1.0, 0.0, 0.0)") btVector3 up) { super((Pointer)null); allocate(ghostObject, convexShape, stepHeight, up); }
	private native void allocate(btPairCachingGhostObject ghostObject, btConvexShape convexShape, @Cast("btScalar") double stepHeight, @Const @ByRef(nullValue = "btVector3(1.0, 0.0, 0.0)") btVector3 up);
	public btKinematicCharacterController(btPairCachingGhostObject ghostObject, btConvexShape convexShape, @Cast("btScalar") double stepHeight) { super((Pointer)null); allocate(ghostObject, convexShape, stepHeight); }
	private native void allocate(btPairCachingGhostObject ghostObject, btConvexShape convexShape, @Cast("btScalar") double stepHeight);

	/**btActionInterface interface */
	public native void updateAction(btCollisionWorld collisionWorld, @Cast("btScalar") double deltaTime);

	/**btActionInterface interface */
	public native void debugDraw(btIDebugDraw debugDrawer);

	public native void setUp(@Const @ByRef btVector3 up);

	public native @Const @ByRef btVector3 getUp();

	/** This should probably be called setPositionIncrementPerSimulatorStep.
	 *  This is neither a direction nor a velocity, but the amount to
	 * 	increment the position each simulation iteration, regardless
	 * 	of dt.
	 *  This call will reset any velocity set by setVelocityForTimeInterval(). */
	public native void setWalkDirection(@Const @ByRef btVector3 walkDirection);

	/** Caller provides a velocity with which the character should move for
	 * 	the given time period.  After the time period, velocity is reset
	 * 	to zero.
	 *  This call will reset any walk direction set by setWalkDirection().
	 *  Negative time intervals will result in no motion. */
	public native void setVelocityForTimeInterval(@Const @ByRef btVector3 velocity,
												@Cast("btScalar") double timeInterval);

	public native void setAngularVelocity(@Const @ByRef btVector3 velocity);
	public native @Const @ByRef btVector3 getAngularVelocity();

	public native void setLinearVelocity(@Const @ByRef btVector3 velocity);
	public native @ByVal btVector3 getLinearVelocity();

	public native void setLinearDamping(@Cast("btScalar") double d);
	public native @Cast("btScalar") double getLinearDamping();
	public native void setAngularDamping(@Cast("btScalar") double d);
	public native @Cast("btScalar") double getAngularDamping();

	public native void reset(btCollisionWorld collisionWorld);
	public native void warp(@Const @ByRef btVector3 origin);

	public native void preStep(btCollisionWorld collisionWorld);
	public native void playerStep(btCollisionWorld collisionWorld, @Cast("btScalar") double dt);

	public native void setStepHeight(@Cast("btScalar") double h);
	public native @Cast("btScalar") double getStepHeight();
	public native void setFallSpeed(@Cast("btScalar") double fallSpeed);
	public native @Cast("btScalar") double getFallSpeed();
	public native void setJumpSpeed(@Cast("btScalar") double jumpSpeed);
	public native @Cast("btScalar") double getJumpSpeed();
	public native void setMaxJumpHeight(@Cast("btScalar") double maxJumpHeight);
	public native @Cast("bool") boolean canJump();

	public native void jump(@Const @ByRef(nullValue = "btVector3(0, 0, 0)") btVector3 v);
	public native void jump();

	public native void applyImpulse(@Const @ByRef btVector3 v);

	public native void setGravity(@Const @ByRef btVector3 gravity);
	public native @ByVal btVector3 getGravity();

	/** The max slope determines the maximum angle that the controller can walk up.
	 *  The slope angle is measured in radians. */
	public native void setMaxSlope(@Cast("btScalar") double slopeRadians);
	public native @Cast("btScalar") double getMaxSlope();

	public native void setMaxPenetrationDepth(@Cast("btScalar") double d);
	public native @Cast("btScalar") double getMaxPenetrationDepth();

	public native btPairCachingGhostObject getGhostObject();
	public native void setUseGhostSweepTest(@Cast("bool") boolean useGhostObjectSweepTest);

	public native @Cast("bool") boolean onGround();
	public native void setUpInterpolate(@Cast("bool") boolean value);
}
