// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


/**btCollisionConfiguration allows to configure Bullet collision detection
 * stack allocator size, default collision algorithms and persistent manifold pool size
 * \todo: describe the meaning */
@Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btCollisionConfiguration extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btCollisionConfiguration(Pointer p) { super(p); }


	/**memory pools */
	public native btPoolAllocator getPersistentManifoldPool();

	public native btPoolAllocator getCollisionAlgorithmPool();

	public native btCollisionAlgorithmCreateFunc getCollisionAlgorithmCreateFunc(int proxyType0, int proxyType1);

	public native btCollisionAlgorithmCreateFunc getClosestPointsAlgorithmCreateFunc(int proxyType0, int proxyType1);
}
