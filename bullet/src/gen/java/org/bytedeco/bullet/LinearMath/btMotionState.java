// Targeted by JavaCPP version 1.5.7: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.LinearMath;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;

import static org.bytedeco.bullet.global.LinearMath.*;


/**The btMotionState interface class allows the dynamics world to synchronize and interpolate the updated world transforms with graphics
 * For optimizations, potentially only moving objects get synchronized (using setWorldPosition/setWorldOrientation) */
@Properties(inherit = org.bytedeco.bullet.presets.LinearMath.class)
public class btMotionState extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btMotionState(Pointer p) { super(p); }


	public native void getWorldTransform(@ByRef btTransform worldTrans);

	//Bullet only calls the update of worldtransform for active objects
	public native void setWorldTransform(@Const @ByRef btTransform worldTrans);
}