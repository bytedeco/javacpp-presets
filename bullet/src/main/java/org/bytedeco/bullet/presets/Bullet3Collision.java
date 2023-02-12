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
    inherit = Bullet3Common.class,
    value = {
        @Platform(
            include = {
                "Bullet3Common/b3AlignedObjectArray.h",
                "Bullet3Collision/BroadPhaseCollision/b3DynamicBvh.h",
                "Bullet3Collision/BroadPhaseCollision/b3OverlappingPair.h",
                "Bullet3Collision/BroadPhaseCollision/b3OverlappingPairCache.h",
                "Bullet3Collision/BroadPhaseCollision/b3BroadphaseCallback.h",
                "Bullet3Collision/BroadPhaseCollision/b3DynamicBvhBroadphase.h",
                "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h",
                "Bullet3Collision/NarrowPhaseCollision/b3Config.h",
                "Bullet3Collision/NarrowPhaseCollision/b3Contact4.h",
                "Bullet3Collision/NarrowPhaseCollision/b3ConvexUtility.h",
                "Bullet3Collision/NarrowPhaseCollision/b3CpuNarrowPhase.h",
                "Bullet3Collision/NarrowPhaseCollision/b3RaycastInfo.h",
                "Bullet3Collision/NarrowPhaseCollision/b3RigidBodyCL.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3BvhSubtreeInfoData.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3ContactConvexConvexSAT.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3FindSeparatingAxis.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3MprPenetration.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3QuantizedBvhNodeData.h",
                "Bullet3Collision/NarrowPhaseCollision/shared/b3ReduceContacts.h",
            },
            link = {"Bullet3Geometry@.3.25", "Bullet3Collision@.3.25"}
        ),
    },
    target = "org.bytedeco.bullet.Bullet3Collision",
    global = "org.bytedeco.bullet.global.Bullet3Collision"
)
public class Bullet3Collision implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info(
                    "B3_DBVT_BP_PROFILE",
                    "B3_DBVT_INLINE",
                    "B3_DBVT_IPOLICY",
                    "B3_DBVT_USE_TEMPLATE",
                    "B3_DBVT_VIRTUAL",
                    "B3_MPR_FABS",
                    "B3_MPR_SQRT"
                ).cppTypes().translate(false))

            .put(new Info("_b3MprSimplex_t").pointerTypes("b3MprSimplex_t"))
            .put(new Info("_b3MprSupport_t").pointerTypes("b3MprSupport_t"))
            .put(new Info("b3Aabb_t").pointerTypes("b3Aabb"))
            .put(new Info("b3AlignedObjectArray<b3DynamicBvh::sStkNN>").pointerTypes("sStkNNArray"))
            .put(new Info("b3AlignedObjectArray<b3DynamicBvh::sStkNPS>").pointerTypes("sStkNPSArray"))
            .put(new Info("b3BroadphasePair").pointerTypes("b3Int4"))
            .put(new Info("b3BvhSubtreeInfoData_t").pointerTypes("b3BvhSubtreeInfoData"))
            .put(new Info("b3Collidable_t").pointerTypes("b3Collidable"))
            .put(new Info("b3Contact4Data_t").pointerTypes("b3Contact4Data"))
            .put(new Info("b3ConvexPolyhedronData_t").pointerTypes("b3ConvexPolyhedronData"))
            .put(new Info("b3DynamicBvh::sStkNN").pointerTypes("b3DynamicBvh.sStkNN"))
            .put(new Info("b3DynamicBvh::sStkNPS").pointerTypes("b3DynamicBvh.sStkNPS"))
            .put(new Info("b3GpuChildShape_t").pointerTypes("b3GpuChildShape"))
            .put(new Info("b3GpuFace_t").pointerTypes("b3GpuFace"))
            .put(new Info("b3InertiaData_t").pointerTypes("b3InertiaData"))
            .put(new Info("b3QuantizedBvhNodeData_t").pointerTypes("b3QuantizedBvhNodeData"))
            .put(new Info("b3RigidBodyData_t").pointerTypes("b3RigidBodyData"))

            .put(new Info("b3BroadphaseRayCallback").purify(true))

            .put(new Info(
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNN>::findBinarySearch",
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNN>::findLinearSearch",
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNN>::findLinearSearch2",
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNN>::remove",
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNPS>::findBinarySearch",
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNPS>::findLinearSearch",
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNPS>::findLinearSearch2",
                    "b3AlignedObjectArray<b3DynamicBvh::sStkNPS>::remove",
                    "b3ConvexUtility::testContainment",
                    "b3CpuNarrowPhase::getBodiesCpu",
                    "b3CpuNarrowPhase::getCollidablesCpu",
                    "b3CpuNarrowPhase::getInternalData",
                    "b3CpuNarrowPhase::getNumBodiesGpu",
                    "b3CpuNarrowPhase::getNumBodyInertiasGpu",
                    "b3CpuNarrowPhase::getNumCollidablesGpu",
                    "b3CpuNarrowPhase::getNumRigidBodies",
                    "b3CpuNarrowPhase::getObjectTransformFromCpu",
                    "b3CpuNarrowPhase::readbackAllBodiesToCpu",
                    "b3CpuNarrowPhase::registerCompoundShape",
                    "b3CpuNarrowPhase::registerConcaveMesh",
                    "b3CpuNarrowPhase::registerFace",
                    "b3CpuNarrowPhase::registerPlaneShape",
                    "b3CpuNarrowPhase::registerSphereShape",
                    "b3CpuNarrowPhase::reset",
                    "b3CpuNarrowPhase::setObjectTransform",
                    "b3CpuNarrowPhase::setObjectTransformCpu",
                    "b3CpuNarrowPhase::setObjectVelocityCpu",
                    "b3CpuNarrowPhase::writeAllBodiesToGpu",
                    "b3DynamicBvh::extractLeaves",
                    "b3DynamicBvh::m_rayTestStack"
                ).skip())
            ;

        Bullet3Common.mapArrays(infoMap,
            "b3Aabb",
            "b3Collidable",
            "b3Contact4Data",
            "b3ConvexPolyhedronData",
            "b3DbvtProxy",
            "b3GpuChildShape",
            "b3GpuFace",
            "b3MyFace",
            "b3RayHit",
            "b3RayInfo",
            "b3RigidBodyData"
        );
    }
}
