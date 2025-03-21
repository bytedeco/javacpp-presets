// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;

// #else
// #endif
/**Typically the conservative advancement reaches solution in a few iterations, clip it to 32 for degenerate cases.
 * See discussion about this here https://bulletphysics.orgphpBB2/viewtopic.php?t=565 */
//will need to digg deeper to make the algorithm more robust
//since, a large epsilon can cause an early termination with false
//positive results (ray intersections that shouldn't be there)

/** btConvexCast is an interface for Casting */
@Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btConvexCast extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btConvexCast(Pointer p) { super(p); }


	/**RayResult stores the closest result
	 *  alternatively, add a callback method to decide about closest/all results */
	@NoOffset public static class CastResult extends Pointer {
	    static { Loader.load(); }
	    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
	    public CastResult(Pointer p) { super(p); }
	    /** Native array allocator. Access with {@link Pointer#position(long)}. */
	    public CastResult(long size) { super((Pointer)null); allocateArray(size); }
	    private native void allocateArray(long size);
	    @Override public CastResult position(long position) {
	        return (CastResult)super.position(position);
	    }
	    @Override public CastResult getPointer(long i) {
	        return new CastResult((Pointer)this).offsetAddress(i);
	    }
	
		//virtual bool	addRayResult(const btVector3& normal,btScalar	fraction) = 0;

		public native void DebugDraw(@Cast("btScalar") double fraction);
		public native void drawCoordSystem(@Const @ByRef btTransform trans);
		public native void reportFailure(int errNo, int numIterations);
		public CastResult() { super((Pointer)null); allocate(); }
		private native void allocate();

		public native @ByRef btTransform m_hitTransformA(); public native CastResult m_hitTransformA(btTransform setter);
		public native @ByRef btTransform m_hitTransformB(); public native CastResult m_hitTransformB(btTransform setter);
		public native @ByRef btVector3 m_normal(); public native CastResult m_normal(btVector3 setter);
		public native @ByRef btVector3 m_hitPoint(); public native CastResult m_hitPoint(btVector3 setter);
		public native @Cast("btScalar") double m_fraction(); public native CastResult m_fraction(double setter);  //input and output
		public native btIDebugDraw m_debugDrawer(); public native CastResult m_debugDrawer(btIDebugDraw setter);
		public native @Cast("btScalar") double m_allowedPenetration(); public native CastResult m_allowedPenetration(double setter);
		
		public native int m_subSimplexCastMaxIterations(); public native CastResult m_subSimplexCastMaxIterations(int setter);
		public native @Cast("btScalar") double m_subSimplexCastEpsilon(); public native CastResult m_subSimplexCastEpsilon(double setter);

	}

	/** cast a convex against another convex object */
	public native @Cast("bool") boolean calcTimeOfImpact(
			@Const @ByRef btTransform fromA,
			@Const @ByRef btTransform toA,
			@Const @ByRef btTransform fromB,
			@Const @ByRef btTransform toB,
			@ByRef CastResult result);
}
