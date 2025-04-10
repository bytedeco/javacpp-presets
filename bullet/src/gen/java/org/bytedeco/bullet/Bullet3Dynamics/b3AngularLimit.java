// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.Bullet3Dynamics;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.Bullet3Common.*;
import static org.bytedeco.bullet.global.Bullet3Common.*;
import org.bytedeco.bullet.Bullet3Collision.*;
import static org.bytedeco.bullet.global.Bullet3Collision.*;

import static org.bytedeco.bullet.global.Bullet3Dynamics.*;


// clang-format on

/*B3_FORCE_INLINE	int	b3TypedConstraint::calculateSerializeBufferSize() const
{
	return sizeof(b3TypedConstraintData);
}
*/

@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.Bullet3Dynamics.class)
public class b3AngularLimit extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public b3AngularLimit(Pointer p) { super(p); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public b3AngularLimit(long size) { super((Pointer)null); allocateArray(size); }
    private native void allocateArray(long size);
    @Override public b3AngularLimit position(long position) {
        return (b3AngularLimit)super.position(position);
    }
    @Override public b3AngularLimit getPointer(long i) {
        return new b3AngularLimit((Pointer)this).offsetAddress(i);
    }

	/** Default constructor initializes limit as inactive, allowing free constraint movement */
	public b3AngularLimit() { super((Pointer)null); allocate(); }
	private native void allocate();

	/** Sets all limit's parameters.
	 *  When low > high limit becomes inactive.
	 *  When high - low > 2PI limit is ineffective too becouse no angle can exceed the limit */
	public native void set(@Cast("b3Scalar") float low, @Cast("b3Scalar") float high, @Cast("b3Scalar") float _softness/*=0.9f*/, @Cast("b3Scalar") float _biasFactor/*=0.3f*/, @Cast("b3Scalar") float _relaxationFactor/*=1.0f*/);
	public native void set(@Cast("b3Scalar") float low, @Cast("b3Scalar") float high);

	/** Checks conastaint angle against limit. If limit is active and the angle violates the limit
	 *  correction is calculated. */
	public native void test(@Cast("const b3Scalar") float angle);

	/** Returns limit's softness */
	public native @Cast("b3Scalar") float getSoftness();

	/** Returns limit's bias factor */
	public native @Cast("b3Scalar") float getBiasFactor();

	/** Returns limit's relaxation factor */
	public native @Cast("b3Scalar") float getRelaxationFactor();

	/** Returns correction value evaluated when test() was invoked */
	public native @Cast("b3Scalar") float getCorrection();

	/** Returns sign value evaluated when test() was invoked */
	public native @Cast("b3Scalar") float getSign();

	/** Gives half of the distance between min and max limit angle */
	public native @Cast("b3Scalar") float getHalfRange();

	/** Returns true when the last test() invocation recognized limit violation */
	public native @Cast("bool") boolean isLimit();

	/** Checks given angle against limit. If limit is active and angle doesn't fit it, the angle
	 *  returned is modified so it equals to the limit closest to given angle. */
	public native void fit(@Cast("b3Scalar*") @ByRef FloatPointer angle);
	public native void fit(@Cast("b3Scalar*") @ByRef FloatBuffer angle);
	public native void fit(@Cast("b3Scalar*") @ByRef float[] angle);

	/** Returns correction value multiplied by sign value */
	public native @Cast("b3Scalar") float getError();

	public native @Cast("b3Scalar") float getLow();

	public native @Cast("b3Scalar") float getHigh();
}
