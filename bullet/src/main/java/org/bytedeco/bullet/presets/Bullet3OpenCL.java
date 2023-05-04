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
import org.bytedeco.javacpp.Pointer;
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
    inherit = {Bullet3Dynamics.class, LinearMath.class},
    value = {
        @Platform(
            define = {
                "B3_USE_CLEW",
            },
            include = {
                "Bullet3Common/b3AlignedObjectArray.h",
                "Bullet3OpenCL/Initialize/b3OpenCLUtils.h",
                "Bullet3OpenCL/ParallelPrimitives/b3BufferInfoCL.h",
                "Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h",
                "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h",
                "Bullet3OpenCL/ParallelPrimitives/b3FillCL.h",
                "Bullet3OpenCL/ParallelPrimitives/b3PrefixScanFloat4CL.h",
                "Bullet3OpenCL/ParallelPrimitives/b3RadixSort32CL.h",
                "Bullet3OpenCL/ParallelPrimitives/b3BoundSearchCL.h",
                "Bullet3OpenCL/BroadphaseCollision/b3SapAabb.h",
                "Bullet3OpenCL/BroadphaseCollision/b3GpuBroadphaseInterface.h",
                "Bullet3OpenCL/BroadphaseCollision/b3GpuGridBroadphase.h",
                "Bullet3OpenCL/BroadphaseCollision/b3GpuParallelLinearBvhBroadphase.h",
                "Bullet3OpenCL/BroadphaseCollision/b3GpuParallelLinearBvh.h",
                "Bullet3OpenCL/BroadphaseCollision/b3GpuSapBroadphase.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3BvhInfo.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3ConvexHullContact.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3ConvexPolyhedronCL.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3GjkEpa.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3OptimizedBvh.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3QuantizedBvh.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3StridingMeshInterface.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3VectorFloat4.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3SupportMappings.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3TriangleCallback.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3TriangleIndexVertexArray.h",
                "Bullet3OpenCL/NarrowphaseCollision/b3VoronoiSimplexSolver.h",
                "Bullet3OpenCL/RigidBody/b3GpuConstraint4.h",
                "Bullet3OpenCL/RigidBody/b3GpuGenericConstraint.h",
                "Bullet3OpenCL/RigidBody/b3GpuJacobiContactSolver.h",
                "Bullet3OpenCL/RigidBody/b3GpuNarrowPhase.h",
                "Bullet3OpenCL/RigidBody/b3GpuNarrowPhaseInternalData.h",
                "Bullet3OpenCL/RigidBody/b3GpuPgsConstraintSolver.h",
                "Bullet3OpenCL/RigidBody/b3GpuPgsContactSolver.h",
                "Bullet3OpenCL/RigidBody/b3GpuRigidBodyPipeline.h",
                "Bullet3OpenCL/Raycast/b3GpuRaycast.h",
                "Bullet3OpenCL/RigidBody/b3GpuRigidBodyPipelineInternalData.h",
                "Bullet3OpenCL/RigidBody/b3GpuSolverBody.h",
                "Bullet3OpenCL/RigidBody/b3GpuSolverConstraint.h",
                "Bullet3OpenCL/RigidBody/b3Solver.h",

            },
            link = "Bullet3OpenCL_clew@.3.25"
        )
    },
    target = "org.bytedeco.bullet.Bullet3OpenCL",
    global = "org.bytedeco.bullet.global.Bullet3OpenCL"
)
public class Bullet3OpenCL implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    private static void mapOCLArrays(InfoMap infoMap, String... typeNames) {
        for (String typeName: typeNames) {
            String cppName = "b3OpenCLArray<" + typeName + ">";
            String javaName = typeName + "OCLArray";
            infoMap.put(new Info(cppName).pointerTypes(javaName));
        }
    }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info(
                    "cl_int",
                    "cl_uint",
                    "cl_bool",
                    "cl_device_local_mem_type"
                ).cast().valueTypes("int").pointerTypes("IntPointer", "IntBuffer", "int[]"))

            .put(new Info(
                    "cl_ulong",
                    "cl_device_type",
                    "cl_command_queue_properties"
                ).cast().valueTypes("long").pointerTypes("LongPointer", "LongBuffer", "long[]"))

            .put(new Info(
                    "cl_platform_id",
                    "cl_device_id",
                    "cl_context",
                    "cl_command_queue",
                    "cl_mem",
                    "cl_program",
                    "cl_kernel"
                ).cast().valueTypes("Pointer").pointerTypes("PointerPointer"))

            .put(new Info(
                    "DEBUG_CHECK_DEQUANTIZATION",
                    "b3Int64",
                    "b3OptimizedBvhNodeData",
                    "b3QuantizedBvhData",
                    "float4"
                ).cppTypes().translate(false))

            .put(new Info("b3AlignedObjectArray<b3ConvexUtility*>").pointerTypes("b3ConvexUtilityArray"))
            .put(new Info("b3AlignedObjectArray<b3OpenCLArray<unsigned char>*>").pointerTypes("b3UnsignedCharOCLArrayArray"))
            .put(new Info("b3AlignedObjectArray<b3OptimizedBvh*>").pointerTypes("b3OptimizedBvhArray"))
            .put(new Info("b3AlignedObjectArray<b3TriangleIndexVertexArray*>").pointerTypes("b3TriangleIndexVertexArrayArray"))
            .put(new Info("b3OpenCLArray<unsigned char>").pointerTypes("b3UnsignedCharOCLArray"))
            .put(new Info("b3OpenCLArray<int>").pointerTypes("b3IntOCLArray"))
            .put(new Info("b3OpenCLArray<unsigned int>").pointerTypes("b3UnsignedIntOCLArray"))
            .put(new Info("b3OpenCLArray<float>").pointerTypes("b3FloatOCLArray"))
            .put(new Info("b3OpenCLArray<b3BroadphasePair>").pointerTypes("b3Int4OCLArray"))

            .put(new Info("b3GpuBroadphaseInterface.h").linePatterns(".*typedef.*CreateFunc.*").skip())

            .put(new Info(
                    "GpuSatCollision::m_concaveHasSeparatingNormals",
                    "GpuSatCollision::m_concaveSepNormals",
                    "GpuSatCollision::m_dmins",
                    "GpuSatCollision::m_gpuCompoundPairs",
                    "GpuSatCollision::m_gpuCompoundSepNormals",
                    "GpuSatCollision::m_gpuHasCompoundSepNormals",
                    "GpuSatCollision::m_hasSeparatingNormals",
                    "GpuSatCollision::m_numCompoundPairsOut",
                    "GpuSatCollision::m_numConcavePairsOut",
                    "GpuSatCollision::m_sepNormals",
                    "GpuSatCollision::m_totalContactsOut",
                    "GpuSatCollision::m_unitSphereDirections",
                    "b3GpuNarrowPhase::setObjectTransform",
                    "b3GpuPgsConstraintSolver::sortConstraintByBatch3",
                    "b3GpuRigidBodyPipeline::registerConvexPolyhedron",
                    "b3GpuSapBroadphase::m_allAabbsGPU",
                    "b3GpuSapBroadphase::m_dst",
                    "b3GpuSapBroadphase::m_gpuSmallSortData",
                    "b3GpuSapBroadphase::m_gpuSmallSortedAabbs",
                    "b3GpuSapBroadphase::m_largeAabbsMappingGPU",
                    "b3GpuSapBroadphase::m_overlappingPairs",
                    "b3GpuSapBroadphase::m_pairCount",
                    "b3GpuSapBroadphase::m_smallAabbsMappingGPU",
                    "b3GpuSapBroadphase::m_sum",
                    "b3GpuSapBroadphase::m_sum2",
                    "b3LauncherCL::validateResults",
                    "b3Solver::m_batchSizes",
                    "b3Solver::m_scan"
                ).skip())
            ;

        Bullet3Common.mapArrays(infoMap,
            "b3BvhInfo",
            "b3BvhSubtreeInfo",
            "b3CompoundOverlappingPair",
            "b3Contact4",
            "b3GpuConstraint4",
            "b3GpuGenericConstraint",
            "b3InertiaData",
            "b3QuantizedBvhNode",
            "b3SapAabb",
            "b3SortData"
        );

        mapOCLArrays(infoMap,
            "b3Aabb",
            "b3BvhInfo",
            "b3BvhSubtreeInfo",
            "b3Collidable",
            "b3CompoundOverlappingPair",
            "b3Contact4",
            "b3ConvexPolyhedronData",
            "b3GpuChildShape",
            "b3GpuConstraint4",
            "b3GpuFace",
            "b3GpuGenericConstraint",
            "b3InertiaData",
            "b3Int2",
            "b3Int4",
            "b3QuantizedBvhNode",
            "b3RayInfo",
            "b3RigidBodyData",
            "b3SapAabb",
            "b3SortData",
            "b3Vector3"
        );
    }
}
