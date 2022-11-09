// Targeted by JavaCPP version 1.5.8: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.Bullet3Common;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.bullet.global.Bullet3Common.*;


/**The b3ConvexSeparatingDistanceUtil can help speed up convex collision detection
 * by conservatively updating a cached separating distance/vector instead of re-calculating the closest distance */
@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.Bullet3Common.class)
public class b3ConvexSeparatingDistanceUtil extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public b3ConvexSeparatingDistanceUtil(Pointer p) { super(p); }

	public b3ConvexSeparatingDistanceUtil(@Cast("b3Scalar") float boundingRadiusA, @Cast("b3Scalar") float boundingRadiusB) { super((Pointer)null); allocate(boundingRadiusA, boundingRadiusB); }
	private native void allocate(@Cast("b3Scalar") float boundingRadiusA, @Cast("b3Scalar") float boundingRadiusB);

	public native @Cast("b3Scalar") float getConservativeSeparatingDistance();

	public native void updateSeparatingDistance(@Const @ByRef b3Transform transA, @Const @ByRef b3Transform transB);

	public native void initSeparatingDistance(@Const @ByRef b3Vector3 separatingVector, @Cast("b3Scalar") float separatingDistance, @Const @ByRef b3Transform transA, @Const @ByRef b3Transform transB);
}