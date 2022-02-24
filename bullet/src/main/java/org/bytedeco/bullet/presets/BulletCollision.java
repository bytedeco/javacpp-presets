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
                "LinearMath/btAlignedObjectArray.h",
                "BulletCollision/BroadphaseCollision/btDbvt.h",
                "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h",
                "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h",
                "BulletCollision/BroadphaseCollision/btDispatcher.h",
                "BulletCollision/BroadphaseCollision/btOverlappingPairCallback.h",
                "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h",
                "BulletCollision/BroadphaseCollision/btQuantizedBvh.h",
                "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h",
                "BulletCollision/BroadphaseCollision/btSimpleBroadphase.h",
                "BulletCollision/BroadphaseCollision/btAxisSweep3.h",
                "BulletCollision/BroadphaseCollision/btDbvtBroadphase.h",
                "BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h",
                "BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h",
                "BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h",
                "BulletCollision/NarrowPhaseCollision/btManifoldPoint.h",
                "BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h",
                "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h",
                "BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h",
                "BulletCollision/CollisionDispatch/btCollisionConfiguration.h",
                "BulletCollision/CollisionDispatch/btCollisionObject.h",
                "BulletCollision/CollisionDispatch/btCollisionCreateFunc.h",
                "BulletCollision/CollisionDispatch/btCollisionDispatcher.h",
                "BulletCollision/CollisionDispatch/btCollisionWorld.h",
                "BulletCollision/CollisionDispatch/btManifoldResult.h",
                "BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h",
                "BulletCollision/CollisionDispatch/btSimulationIslandManager.h",
                "BulletCollision/CollisionDispatch/btUnionFind.h",
                "BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h",
                "BulletCollision/CollisionShapes/btCollisionShape.h",
                "BulletCollision/CollisionShapes/btConvexShape.h",
                "BulletCollision/CollisionShapes/btConvexPolyhedron.h",
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
    target = "org.bytedeco.bullet.BulletCollision",
    global = "org.bytedeco.bullet.global.BulletCollision"
)
public class BulletCollision implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("bt32BitAxisSweep3").base("btBroadphaseInterface"))
            .put(new Info("btAxisSweep3").base("btBroadphaseInterface"))

            .put(new Info("BT_DECLARE_STACK_ONLY_OBJECT").cppText("#define BT_DECLARE_STACK_ONLY_OBJECT"))
            .put(new Info("btCollisionObjectData").cppText("#define btCollisionObjectData btCollisionObjectFloatData"))
            .put(new Info("btOptimizedBvhNodeData").cppText("#define btOptimizedBvhNodeData btOptimizedBvhNodeFloatData"))
            .put(new Info("btPersistentManifoldData").cppText("#define btPersistentManifoldData btPersistentManifoldFloatData"))
            .put(new Info("btQuantizedBvhData").cppText("#define btQuantizedBvhData btQuantizedBvhFloatData"))

            .put(new Info("DBVT_INLINE").cppTypes().annotations())

            .put(new Info(
                    "DBVT_BP_PROFILE",
                    "DEBUG_PERSISTENCY",
                    "NO_VIRTUAL_INTERFACE",
                    "PFX_USE_FREE_VECTORMATH",
                    "__SPU__"
                ).define(false))

            .put(new Info("btAlignedObjectArray<btBvhSubtreeInfo>").pointerTypes("btBvhSubtreeInfoArray"))
            .put(new Info("btAlignedObjectArray<btCollisionObject*>").pointerTypes("btCollisionObjectArray"))
            .put(new Info("btAlignedObjectArray<btIndexedMesh>").pointerTypes("btIndexedMeshArray"))
            .put(new Info("btAlignedObjectArray<btPersistentManifold*>").pointerTypes("btPersistentManifoldArray"))
            .put(new Info("btAlignedObjectArray<btQuantizedBvhNode>").pointerTypes("btQuantizedBvhNodeArray"))

            .put(new Info("btDispatcher.h").linePatterns("class btRigidBody;").skip())
            .put(new Info("btPersistentManifold.h").linePatterns("struct btCollisionResult;").skip())

            .put(new Info(
                    "DBVT_CHECKTYPE",
                    "DBVT_IPOLICY",
                    "DBVT_PREFIX",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::findBinarySearch",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::findLinearSearch",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::findLinearSearch2",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::remove",
                    "btAlignedObjectArray<btIndexedMesh>::findBinarySearch",
                    "btAlignedObjectArray<btIndexedMesh>::findLinearSearch",
                    "btAlignedObjectArray<btIndexedMesh>::findLinearSearch2",
                    "btAlignedObjectArray<btIndexedMesh>::remove",
                    "btAlignedObjectArray<btQuantizedBvhNode>::findBinarySearch",
                    "btAlignedObjectArray<btQuantizedBvhNode>::findLinearSearch",
                    "btAlignedObjectArray<btQuantizedBvhNode>::findLinearSearch2",
                    "btAlignedObjectArray<btQuantizedBvhNode>::remove",
                    "btCollisionWorld::AllHitsRayResultCallback::m_collisionObjects",
                    "btCompoundShapeChild::m_node",
                    "btConvexPolyhedron::m_faces",
                    "btDbvt::allocate",
                    "btDbvt::extractLeaves",
                    "btDbvt::m_stkStack",
                    "btDbvt::rayTestInternal",
                    "btDbvtBroadphase::m_rayTestStacks",
                    "btDbvtProxy",
                    "btHashedOverlappingPairCache::getOverlappingPairArray",
                    "btNullPairCache::getOverlappingPairArray",
                    "btOverlappingPairCache::getOverlappingPairArray",
                    "btSortedOverlappingPairCache::getOverlappingPairArray",
                    "btSphereSphereCollisionAlgorithm::getAllContactManifolds",
                    "gContactDestroyedCallback",
                    "gContactEndedCallback",
                    "gContactProcessedCallback",
                    "gContactStartedCallback"
                ).skip())
            ;
    }
}
