// Targeted by JavaCPP version 1.5.8: DO NOT EDIT THIS FILE

package org.bytedeco.bullet.Bullet3OpenCL;

import java.nio.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.annotation.*;

import static org.bytedeco.javacpp.presets.javacpp.*;
import org.bytedeco.bullet.Bullet3Common.*;
import static org.bytedeco.bullet.global.Bullet3Common.*;
import org.bytedeco.bullet.Bullet3Collision.*;
import static org.bytedeco.bullet.global.Bullet3Collision.*;
import org.bytedeco.bullet.Bullet3Dynamics.*;
import static org.bytedeco.bullet.global.Bullet3Dynamics.*;
import org.bytedeco.bullet.LinearMath.*;
import static org.bytedeco.bullet.global.LinearMath.*;

import static org.bytedeco.bullet.global.Bullet3OpenCL.*;


@NoOffset @Properties(inherit = org.bytedeco.bullet.presets.Bullet3OpenCL.class)
public class b3GpuNarrowPhase extends Pointer {
    static { Loader.load(); }
    /** Pointer cast constructor. Invokes {@link Pointer#Pointer(Pointer)}. */
    public b3GpuNarrowPhase(Pointer p) { super(p); }

	public b3GpuNarrowPhase(@Cast("cl_context") Pointer vtx, @Cast("cl_device_id") Pointer dev, @Cast("cl_command_queue") Pointer q, @Const @ByRef b3Config config) { super((Pointer)null); allocate(vtx, dev, q, config); }
	private native void allocate(@Cast("cl_context") Pointer vtx, @Cast("cl_device_id") Pointer dev, @Cast("cl_command_queue") Pointer q, @Const @ByRef b3Config config);

	public native int registerSphereShape(float radius);
	public native int registerPlaneShape(@Const @ByRef b3Vector3 planeNormal, float planeConstant);

	public native int registerCompoundShape(b3GpuChildShapeArray childShapes);
	public native int registerFace(@Const @ByRef b3Vector3 faceNormal, float faceConstant);

	public native int registerConcaveMesh(b3Vector3Array vertices, b3IntArray indices, @Const FloatPointer scaling);
	public native int registerConcaveMesh(b3Vector3Array vertices, b3IntArray indices, @Const FloatBuffer scaling);
	public native int registerConcaveMesh(b3Vector3Array vertices, b3IntArray indices, @Const float[] scaling);

	//do they need to be merged?

	public native int registerConvexHullShape(b3ConvexUtility utilPtr);
	public native int registerConvexHullShape(@Const FloatPointer vertices, int strideInBytes, int numVertices, @Const FloatPointer scaling);
	public native int registerConvexHullShape(@Const FloatBuffer vertices, int strideInBytes, int numVertices, @Const FloatBuffer scaling);
	public native int registerConvexHullShape(@Const float[] vertices, int strideInBytes, int numVertices, @Const float[] scaling);

	public native int registerRigidBody(int collidableIndex, float mass, @Const FloatPointer _position, @Const FloatPointer orientation, @Const FloatPointer aabbMin, @Const FloatPointer aabbMax, @Cast("bool") boolean writeToGpu);
	public native int registerRigidBody(int collidableIndex, float mass, @Const FloatBuffer _position, @Const FloatBuffer orientation, @Const FloatBuffer aabbMin, @Const FloatBuffer aabbMax, @Cast("bool") boolean writeToGpu);
	public native int registerRigidBody(int collidableIndex, float mass, @Const float[] _position, @Const float[] orientation, @Const float[] aabbMin, @Const float[] aabbMax, @Cast("bool") boolean writeToGpu);
	

	public native void writeAllBodiesToGpu();
	public native void reset();
	public native void readbackAllBodiesToCpu();
	public native @Cast("bool") boolean getObjectTransformFromCpu(FloatPointer _position, FloatPointer orientation, int bodyIndex);
	public native @Cast("bool") boolean getObjectTransformFromCpu(FloatBuffer _position, FloatBuffer orientation, int bodyIndex);
	public native @Cast("bool") boolean getObjectTransformFromCpu(float[] _position, float[] orientation, int bodyIndex);

	public native void setObjectTransformCpu(FloatPointer _position, FloatPointer orientation, int bodyIndex);
	public native void setObjectTransformCpu(FloatBuffer _position, FloatBuffer orientation, int bodyIndex);
	public native void setObjectTransformCpu(float[] _position, float[] orientation, int bodyIndex);
	public native void setObjectVelocityCpu(FloatPointer linVel, FloatPointer angVel, int bodyIndex);
	public native void setObjectVelocityCpu(FloatBuffer linVel, FloatBuffer angVel, int bodyIndex);
	public native void setObjectVelocityCpu(float[] linVel, float[] angVel, int bodyIndex);

	public native void computeContacts(@Cast("cl_mem") Pointer broadphasePairs, int numBroadphasePairs, @Cast("cl_mem") Pointer aabbsWorldSpace, int numObjects);

	public native @Cast("cl_mem") Pointer getBodiesGpu();
	public native @Const b3RigidBodyData getBodiesCpu();
	//struct b3RigidBodyData* getBodiesCpu();

	public native int getNumBodiesGpu();

	public native @Cast("cl_mem") Pointer getBodyInertiasGpu();
	public native int getNumBodyInertiasGpu();

	public native @Cast("cl_mem") Pointer getCollidablesGpu();
	public native @Const b3Collidable getCollidablesCpu();
	public native int getNumCollidablesGpu();

	public native @Const b3SapAabb getLocalSpaceAabbsCpu();

	public native @Const b3Contact4 getContactsCPU();

	public native @Cast("cl_mem") Pointer getContactsGpu();
	public native int getNumContactsGpu();

	public native @Cast("cl_mem") Pointer getAabbLocalSpaceBufferGpu();

	public native int getNumRigidBodies();

	public native int allocateCollidable();

	public native int getStatic0Index();
	public native @ByRef b3Collidable getCollidableCpu(int collidableIndex);

	public native b3GpuNarrowPhaseInternalData getInternalData();

	public native @Const @ByRef b3SapAabb getLocalSpaceAabb(int collidableIndex);
}