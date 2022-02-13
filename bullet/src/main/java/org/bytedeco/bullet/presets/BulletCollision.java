package org.bytedeco.bullet.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = LinearMath.class,
    value = {
        @Platform(
            include = {
                "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h",
                "BulletCollision/BroadphaseCollision/btDispatcher.h",
                "BulletCollision/BroadphaseCollision/btOverlappingPairCallback.h",
                "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h",
                "BulletCollision/BroadphaseCollision/btQuantizedBvh.h",
                "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h",
                "BulletCollision/BroadphaseCollision/btSimpleBroadphase.h",
                "BulletCollision/BroadphaseCollision/btAxisSweep3.h",
                "BulletCollision/BroadphaseCollision/btDbvtBroadphase.h",
                "BulletCollision/NarrowPhaseCollision/btManifoldPoint.h",
                "BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h",
                "BulletCollision/CollisionDispatch/btCollisionObject.h",
                "BulletCollision/CollisionDispatch/btCollisionCreateFunc.h",
                "BulletCollision/CollisionDispatch/btCollisionDispatcher.h",
                "BulletCollision/CollisionDispatch/btCollisionWorld.h",
                "BulletCollision/CollisionDispatch/btManifoldResult.h",
                "BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h",
                "BulletCollision/CollisionShapes/btCollisionShape.h",
                "BulletCollision/CollisionShapes/btPolyhedralConvexShape.h",
                "BulletCollision/CollisionShapes/btConvexInternalShape.h",
                "BulletCollision/CollisionShapes/btBoxShape.h",
                "BulletCollision/CollisionShapes/btSphereShape.h",
                "BulletCollision/CollisionShapes/btCapsuleShape.h",
                "BulletCollision/CollisionShapes/btCylinderShape.h",
                "BulletCollision/CollisionShapes/btConeShape.h",
                "BulletCollision/CollisionShapes/btConcaveShape.h",
                "BulletCollision/CollisionShapes/btTriangleCallback.h",
                "BulletCollision/CollisionShapes/btStaticPlaneShape.h",
                "BulletCollision/CollisionShapes/btConvexHullShape.h",
                "BulletCollision/CollisionShapes/btStridingMeshInterface.h",
                "BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h",
                "BulletCollision/CollisionShapes/btTriangleMesh.h",
                "BulletCollision/CollisionShapes/btConvexTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btOptimizedBvh.h",
                "BulletCollision/CollisionShapes/btTriangleInfoMap.h",
                "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btCompoundShape.h",
                "BulletCollision/CollisionShapes/btTetrahedronShape.h",
                "BulletCollision/CollisionShapes/btEmptyShape.h",
                "BulletCollision/CollisionShapes/btMultiSphereShape.h",
                "BulletCollision/CollisionShapes/btUniformScalingShape.h",
            },
            link = "BulletCollision"
        )
    },
    target = "org.bytedeco.bullet.BulletCollision"
)
public class BulletCollision implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("btCollisionObjectData").cppText("#define btCollisionObjectData btCollisionObjectFloatData"))
            .put(new Info( "btOverlappingPairCache::getOverlappingPairArray").skip())
            .put(new Info("btHashedOverlappingPairCache::getOverlappingPairArray").skip())
            .put(new Info("btSortedOverlappingPairCache::getOverlappingPairArray").skip())
            .put(new Info("btNullPairCache::getOverlappingPairArray").skip())
            .put(new Info("btCollisionWorld::AllHitsRayResultCallback::m_collisionObjects").skip())
            .put(new Info("PFX_USE_FREE_VECTORMATH").define(false))
            .put(new Info("__SPU__").define(false))
            .put(new Info("btQuantizedBvhData").cppText("#define btQuantizedBvhData btQuantizedBvhFloatData"))
            .put(new Info("btOptimizedBvhNodeData").cppText("#define btOptimizedBvhNodeData btOptimizedBvhNodeFloatData"))
            .put(new Info("btCompoundShapeChild::m_node").skip())
            .put(new Info("btSphereSphereCollisionAlgorithm::getAllContactManifolds").skip())
            .put(new Info("btAxisSweep3").base("btBroadphaseInterface"))
            .put(new Info("bt32BitAxisSweep3").base("btBroadphaseInterface"))
            .put(new Info("btDbvtBroadphase::m_rayTestStacks").skip())
            .put(new Info("btDbvtProxy").skip())
            .put(new Info("DBVT_BP_PROFILE").define(false))
            ;
    }
}
