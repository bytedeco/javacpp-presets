// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


/**The btPolyhedralConvexAabbCachingShape adds aabb caching to the btPolyhedralConvexShape */
@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btPolyhedralConvexAabbCachingShape extends btPolyhedralConvexShape {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btPolyhedralConvexAabbCachingShape(Pointer p) { super(p); }

	public native void getNonvirtualAabb(@Const @ByRef btTransform trans, @ByRef btVector3 aabbMin, @ByRef btVector3 aabbMax, @Cast("btScalar") double margin);

	public native void setLocalScaling(@Const @ByRef btVector3 scaling);

	public native void getAabb(@Const @ByRef btTransform t, @ByRef btVector3 aabbMin, @ByRef btVector3 aabbMax);

	public native void recalcLocalAabb();
}
