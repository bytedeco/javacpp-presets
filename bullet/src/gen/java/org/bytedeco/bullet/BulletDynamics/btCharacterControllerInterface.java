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


@Properties(inherit = org.bytedeco.bullet.presets.BulletDynamics.class)
public class btCharacterControllerInterface extends btActionInterface {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btCharacterControllerInterface(Pointer p) { super(p); }


	public native void setWalkDirection(@Const @ByRef btVector3 walkDirection);
	public native void setVelocityForTimeInterval(@Const @ByRef btVector3 velocity, @Cast("btScalar") double timeInterval);
	public native void reset(btCollisionWorld collisionWorld);
	public native void warp(@Const @ByRef btVector3 origin);

	public native void preStep(btCollisionWorld collisionWorld);
	public native void playerStep(btCollisionWorld collisionWorld, @Cast("btScalar") double dt);
	public native @Cast("bool") boolean canJump();
	public native void jump(@Const @ByRef(nullValue = "btVector3(0, 0, 0)") btVector3 dir);
	public native void jump();

	public native @Cast("bool") boolean onGround();
	public native void setUpInterpolate(@Cast("bool") boolean value);
}
