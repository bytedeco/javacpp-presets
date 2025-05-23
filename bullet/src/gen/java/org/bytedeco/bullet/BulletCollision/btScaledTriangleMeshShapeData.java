// Targeted by JavaCPP version 1.5.11: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.BulletCollision;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.BulletCollision.*;


/**do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64 */
@Properties(inherit = org.bytedeco.bullet.presets.BulletCollision.class)
public class btScaledTriangleMeshShapeData extends Pointer {
    static { Loader.load(); }
    /** Default native constructor. */
    public btScaledTriangleMeshShapeData() { super((Pointer)null); allocate(); }
    /** Native array allocator. Access with {@link Pointer#position(long)}. */
    public btScaledTriangleMeshShapeData(long size) { super((Pointer)null); allocateArray(size); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public btScaledTriangleMeshShapeData(Pointer p) { super(p); }
    private native void allocate();
    private native void allocateArray(long size);
    @Override public btScaledTriangleMeshShapeData position(long position) {
        return (btScaledTriangleMeshShapeData)super.position(position);
    }
    @Override public btScaledTriangleMeshShapeData getPointer(long i) {
        return new btScaledTriangleMeshShapeData((Pointer)this).offsetAddress(i);
    }

	public native @ByRef btTriangleMeshShapeData m_trimeshShapeData(); public native btScaledTriangleMeshShapeData m_trimeshShapeData(btTriangleMeshShapeData setter);

	public native @ByRef btVector3FloatData m_localScaling(); public native btScaledTriangleMeshShapeData m_localScaling(btVector3FloatData setter);
}
