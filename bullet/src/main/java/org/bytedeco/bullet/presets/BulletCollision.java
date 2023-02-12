/*
 * Copyright (C) 2022 Andrey Krainyak
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.bullet.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

/**
 *
 * @author Andrey Krainyak
 */
@Properties(
    inherit = LinearMath.class,
    value = {
        @Platform(
            include = {
                "LinearMath/btAlignedObjectArray.h",
                "BulletCollision/BroadphaseCollision/btAxisSweep3.h",
                "BulletCollision/BroadphaseCollision/btBroadphaseInterface.h",
                "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h",
                "BulletCollision/BroadphaseCollision/btCollisionAlgorithm.h",
                "BulletCollision/BroadphaseCollision/btDbvt.h",
                "BulletCollision/BroadphaseCollision/btDbvtBroadphase.h",
                "BulletCollision/BroadphaseCollision/btDispatcher.h",
                "BulletCollision/BroadphaseCollision/btOverlappingPairCache.h",
                "BulletCollision/BroadphaseCollision/btOverlappingPairCallback.h",
                "BulletCollision/BroadphaseCollision/btQuantizedBvh.h",
                "BulletCollision/BroadphaseCollision/btSimpleBroadphase.h",
                "BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h",
                "BulletCollision/NarrowPhaseCollision/btContinuousConvexCollision.h",
                "BulletCollision/NarrowPhaseCollision/btConvexCast.h",
                "BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h",
                "BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h",
                "BulletCollision/NarrowPhaseCollision/btGjkCollisionDescription.h",
                "BulletCollision/NarrowPhaseCollision/btGjkConvexCast.h",
                "BulletCollision/NarrowPhaseCollision/btGjkEpa2.h",
                "BulletCollision/NarrowPhaseCollision/btGjkEpa3.h",
                "BulletCollision/NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h",
                "BulletCollision/NarrowPhaseCollision/btGjkPairDetector.h",
                "BulletCollision/NarrowPhaseCollision/btManifoldPoint.h",
                "BulletCollision/NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h",
                "BulletCollision/NarrowPhaseCollision/btMprPenetration.h",
                "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h",
                "BulletCollision/NarrowPhaseCollision/btPointCollector.h",
                "BulletCollision/NarrowPhaseCollision/btPolyhedralContactClipping.h",
                "BulletCollision/NarrowPhaseCollision/btRaycastCallback.h",
                "BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h",
                "BulletCollision/NarrowPhaseCollision/btSubSimplexConvexCast.h",
                "BulletCollision/NarrowPhaseCollision/btVoronoiSimplexSolver.h",
                "BulletCollision/CollisionDispatch/btActivatingCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btBoxBoxCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btBoxBoxDetector.h",
                "BulletCollision/CollisionDispatch/btCollisionConfiguration.h",
                "BulletCollision/CollisionDispatch/btCollisionCreateFunc.h",
                "BulletCollision/CollisionDispatch/btCollisionDispatcher.h",
                "BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h",
                "BulletCollision/CollisionDispatch/btCollisionObject.h",
                "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h",
                "BulletCollision/CollisionDispatch/btCollisionWorld.h",
                "BulletCollision/CollisionDispatch/btCollisionWorldImporter.h",
                "BulletCollision/CollisionDispatch/btCompoundCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btCompoundCompoundCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btConvex2dConvex2dAlgorithm.h",
                "BulletCollision/CollisionDispatch/btConvexConcaveCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btConvexConvexAlgorithm.h",
                "BulletCollision/CollisionDispatch/btConvexPlaneCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btDefaultCollisionConfiguration.h",
                "BulletCollision/CollisionDispatch/btEmptyCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btGhostObject.h",
                "BulletCollision/CollisionDispatch/btHashedSimplePairCache.h",
                "BulletCollision/CollisionDispatch/btInternalEdgeUtility.h",
                "BulletCollision/CollisionDispatch/btManifoldResult.h",
                "BulletCollision/CollisionDispatch/btSimulationIslandManager.h",
                "BulletCollision/CollisionDispatch/btSphereBoxCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btSphereSphereCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btSphereTriangleCollisionAlgorithm.h",
                "BulletCollision/CollisionDispatch/btUnionFind.h",
                "BulletCollision/CollisionDispatch/SphereTriangleDetector.h",
                "BulletCollision/CollisionShapes/btBox2dShape.h",
                "BulletCollision/CollisionShapes/btBoxShape.h",
                "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btCapsuleShape.h",
                "BulletCollision/CollisionShapes/btCollisionMargin.h",
                "BulletCollision/CollisionShapes/btCollisionShape.h",
                "BulletCollision/CollisionShapes/btCompoundShape.h",
                "BulletCollision/CollisionShapes/btConcaveShape.h",
                "BulletCollision/CollisionShapes/btConeShape.h",
                "BulletCollision/CollisionShapes/btConvex2dShape.h",
                "BulletCollision/CollisionShapes/btConvexHullShape.h",
                "BulletCollision/CollisionShapes/btConvexInternalShape.h",
                "BulletCollision/CollisionShapes/btConvexPointCloudShape.h",
                "BulletCollision/CollisionShapes/btConvexPolyhedron.h",
                "BulletCollision/CollisionShapes/btConvexShape.h",
                "BulletCollision/CollisionShapes/btConvexTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btCylinderShape.h",
                "BulletCollision/CollisionShapes/btEmptyShape.h",
                "BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h",
                "BulletCollision/CollisionShapes/btMaterial.h",
                "BulletCollision/CollisionShapes/btMiniSDF.h",
                "BulletCollision/CollisionShapes/btMinkowskiSumShape.h",
                "BulletCollision/CollisionShapes/btMultimaterialTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btMultiSphereShape.h",
                "BulletCollision/CollisionShapes/btOptimizedBvh.h",
                "BulletCollision/CollisionShapes/btPolyhedralConvexShape.h",
                "BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btSdfCollisionShape.h",
                "BulletCollision/CollisionShapes/btShapeHull.h",
                "BulletCollision/CollisionShapes/btSphereShape.h",
                "BulletCollision/CollisionShapes/btStaticPlaneShape.h",
                "BulletCollision/CollisionShapes/btStridingMeshInterface.h",
                "BulletCollision/CollisionShapes/btTetrahedronShape.h",
                "BulletCollision/CollisionShapes/btTriangleBuffer.h",
                "BulletCollision/CollisionShapes/btTriangleCallback.h",
                "BulletCollision/CollisionShapes/btTriangleIndexVertexArray.h",
                "BulletCollision/CollisionShapes/btTriangleIndexVertexMaterialArray.h",
                "BulletCollision/CollisionShapes/btTriangleInfoMap.h",
                "BulletCollision/CollisionShapes/btTriangleMesh.h",
                "BulletCollision/CollisionShapes/btTriangleMeshShape.h",
                "BulletCollision/CollisionShapes/btTriangleShape.h",
                "BulletCollision/CollisionShapes/btUniformScalingShape.h",
                "BulletCollision/Gimpact/btBoxCollision.h",
                "BulletCollision/Gimpact/btClipPolygon.h",
                "BulletCollision/Gimpact/btCompoundFromGimpact.h",
                "BulletCollision/Gimpact/btContactProcessing.h",
                "BulletCollision/Gimpact/btContactProcessingStructs.h",
                "BulletCollision/Gimpact/btGImpactBvh.h",
                "BulletCollision/Gimpact/btGImpactBvhStructs.h",
                "BulletCollision/Gimpact/btGImpactCollisionAlgorithm.h",
                "BulletCollision/Gimpact/btGImpactMassUtil.h",
                "BulletCollision/Gimpact/btGImpactQuantizedBvh.h",
                "BulletCollision/Gimpact/btGImpactQuantizedBvhStructs.h",
                "BulletCollision/Gimpact/btGImpactShape.h",
                "BulletCollision/Gimpact/btGenericPoolAllocator.h",
                "BulletCollision/Gimpact/btGeometryOperations.h",
                "BulletCollision/Gimpact/btQuantization.h",
                "BulletCollision/Gimpact/btTriangleShapeEx.h",
                "BulletCollision/Gimpact/gim_pair.h",
            },
            link = "BulletCollision@.3.25"
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

            .put(new Info(
                    "BT_DEBUG_COLLISION_PAIRS",
                    "BT_DECLARE_STACK_ONLY_OBJECT",
                    "BT_UINT_MAX",
                    "SUPPORT_GIMPACT_SHAPE_IMPORT",
                    "TRI_COLLISION_PROFILING",
                    "btCollisionObjectData",
                    "btOptimizedBvhNodeData",
                    "btPersistentManifoldData",
                    "btQuantizedBvhData"
                ).cppTypes().translate(false))

            .put(new Info("DBVT_INLINE").cppTypes().annotations())

            .put(new Info(
                    "BT_DECLARE_STACK_ONLY_OBJECT",
                    "BT_INTERNAL_EDGE_DEBUG_DRAW",
                    "DBVT_BP_PROFILE",
                    "DEBUG_MPR",
                    "DEBUG_PERSISTENCY",
                    "NO_VIRTUAL_INTERFACE",
                    "PFX_USE_FREE_VECTORMATH",
                    "__SPU__"
                ).define(false))

            .put(new Info("btAlignedObjectArray<BT_QUANTIZED_BVH_NODE>").pointerTypes("BT_QUANTIZED_BVH_NODE_Array"))
            .put(new Info("btAlignedObjectArray<GIM_BVH_DATA>").pointerTypes("GIM_BVH_DATA_Array_"))
            .put(new Info("btAlignedObjectArray<GIM_BVH_TREE_NODE>").pointerTypes("GIM_BVH_TREE_NODE_Array_"))
            .put(new Info("btAlignedObjectArray<GIM_CONTACT>").pointerTypes("GIM_CONTACT_Array_"))
            .put(new Info("btAlignedObjectArray<GIM_PAIR>").pointerTypes("GIM_PAIR_Array_"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<btCell32> >").pointerTypes("btCell32ArrayArray"))
            .put(new Info("btAlignedObjectArray<btBvhSubtreeInfo>").pointerTypes("btBvhSubtreeInfoArray"))
            .put(new Info("btAlignedObjectArray<btCell32>").pointerTypes("btCell32Array"))
            .put(new Info("btAlignedObjectArray<btCollisionObject*>").pointerTypes("btCollisionObjectArray"))
            .put(new Info("btAlignedObjectArray<btDbvt::sStkNN>").pointerTypes("btDbvtStkNNArray"))
            .put(new Info("btAlignedObjectArray<btDbvt::sStkNPS>").pointerTypes("btDbvtStkNPSArray"))
            .put(new Info("btAlignedObjectArray<btFace>").pointerTypes("btFaceArray"))
            .put(new Info("btAlignedObjectArray<btIndexedMesh>").pointerTypes("btIndexedMeshArray"))
            .put(new Info("btAlignedObjectArray<btPersistentManifold*>").pointerTypes("btPersistentManifoldArray"))
            .put(new Info("btAlignedObjectArray<btQuantizedBvhNode>").pointerTypes("btQuantizedBvhNodeArray"))
            .put(new Info("btDbvt::sStkNN").pointerTypes("btDbvt.sStkNN"))
            .put(new Info("btDbvt::sStkNPS").pointerTypes("btDbvt.sStkNPS"))

            .put(new Info("btCollisionObjectWrapper").purify(true))

            .put(new Info("btCollisionWorldImporter.h").linePatterns("struct btContactSolverInfo;").skip())
            .put(new Info("btDispatcher.h").linePatterns("class btRigidBody;").skip())
            .put(new Info("btPersistentManifold.h").linePatterns("struct btCollisionResult;").skip())

            .put(new Info(
                    "BT_MPR_FABS",
                    "BT_MPR_SQRT",
                    "DBVT_CHECKTYPE",
                    "DBVT_IPOLICY",
                    "DBVT_PREFIX",
                    "MAX_CONVEX_CAST_ITERATIONS",
                    "MAX_CONVEX_CAST_EPSILON",
                    "btAABB",
                    "btAlignedObjectArray<BT_QUANTIZED_BVH_NODE>::findBinarySearch",
                    "btAlignedObjectArray<BT_QUANTIZED_BVH_NODE>::findLinearSearch",
                    "btAlignedObjectArray<BT_QUANTIZED_BVH_NODE>::findLinearSearch2",
                    "btAlignedObjectArray<BT_QUANTIZED_BVH_NODE>::remove",
                    "btAlignedObjectArray<GIM_BVH_DATA>::findBinarySearch",
                    "btAlignedObjectArray<GIM_BVH_DATA>::findLinearSearch",
                    "btAlignedObjectArray<GIM_BVH_DATA>::findLinearSearch2",
                    "btAlignedObjectArray<GIM_BVH_DATA>::remove",
                    "btAlignedObjectArray<GIM_BVH_TREE_NODE>::findBinarySearch",
                    "btAlignedObjectArray<GIM_BVH_TREE_NODE>::findLinearSearch",
                    "btAlignedObjectArray<GIM_BVH_TREE_NODE>::findLinearSearch2",
                    "btAlignedObjectArray<GIM_BVH_TREE_NODE>::remove",
                    "btAlignedObjectArray<GIM_CONTACT>::findBinarySearch",
                    "btAlignedObjectArray<GIM_CONTACT>::findLinearSearch",
                    "btAlignedObjectArray<GIM_CONTACT>::findLinearSearch2",
                    "btAlignedObjectArray<GIM_CONTACT>::remove",
                    "btAlignedObjectArray<GIM_PAIR>::findBinarySearch",
                    "btAlignedObjectArray<GIM_PAIR>::findLinearSearch",
                    "btAlignedObjectArray<GIM_PAIR>::findLinearSearch2",
                    "btAlignedObjectArray<GIM_PAIR>::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<btCell32> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btCell32> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btCell32> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<btCell32> >::remove",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::findBinarySearch",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::findLinearSearch",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::findLinearSearch2",
                    "btAlignedObjectArray<btBvhSubtreeInfo>::remove",
                    "btAlignedObjectArray<btCell32>::findBinarySearch",
                    "btAlignedObjectArray<btCell32>::findLinearSearch",
                    "btAlignedObjectArray<btCell32>::findLinearSearch2",
                    "btAlignedObjectArray<btCell32>::remove",
                    "btAlignedObjectArray<btDbvt::sStkNN>::findBinarySearch",
                    "btAlignedObjectArray<btDbvt::sStkNN>::findLinearSearch",
                    "btAlignedObjectArray<btDbvt::sStkNN>::findLinearSearch2",
                    "btAlignedObjectArray<btDbvt::sStkNN>::remove",
                    "btAlignedObjectArray<btDbvt::sStkNPS>::findBinarySearch",
                    "btAlignedObjectArray<btDbvt::sStkNPS>::findLinearSearch",
                    "btAlignedObjectArray<btDbvt::sStkNPS>::findLinearSearch2",
                    "btAlignedObjectArray<btDbvt::sStkNPS>::remove",
                    "btAlignedObjectArray<btFace>::findBinarySearch",
                    "btAlignedObjectArray<btFace>::findLinearSearch",
                    "btAlignedObjectArray<btFace>::findLinearSearch2",
                    "btAlignedObjectArray<btFace>::remove",
                    "btAlignedObjectArray<btIndexedMesh>::findBinarySearch",
                    "btAlignedObjectArray<btIndexedMesh>::findLinearSearch",
                    "btAlignedObjectArray<btIndexedMesh>::findLinearSearch2",
                    "btAlignedObjectArray<btIndexedMesh>::remove",
                    "btAlignedObjectArray<btQuantizedBvhNode>::findBinarySearch",
                    "btAlignedObjectArray<btQuantizedBvhNode>::findLinearSearch",
                    "btAlignedObjectArray<btQuantizedBvhNode>::findLinearSearch2",
                    "btAlignedObjectArray<btQuantizedBvhNode>::remove",
                    "btCollisionWorld::AllHitsRayResultCallback::m_collisionObjects",
                    "btDbvt::extractLeaves",
                    "btDbvtBroadphase::m_rayTestStacks",
                    "gContactDestroyedCallback",
                    "gContactEndedCallback",
                    "gContactProcessedCallback",
                    "gContactStartedCallback"
                ).skip())
            ;
    }
}
