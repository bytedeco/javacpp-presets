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
    inherit = javacpp.class,
    value = {
        @Platform(
            define = {
                "BT_USE_DOUBLE_PRECISION",
            },
            include = {
                "LinearMath/btScalar.h",
                "LinearMath/btVector3.h",
                "LinearMath/btQuadWord.h",
                "LinearMath/btQuaternion.h",
                "LinearMath/btMatrix3x3.h",
                "LinearMath/btTransform.h",
                "LinearMath/btAlignedObjectArray.h",
                "LinearMath/btHashMap.h",
                "LinearMath/btSerializer.h",
                "LinearMath/btIDebugDraw.h",
                "LinearMath/btQuickprof.h",
                "LinearMath/btMotionState.h",
                "LinearMath/btDefaultMotionState.h",
                "LinearMath/btSpatialAlgebra.h",
                "LinearMath/btPoolAllocator.h",
                "LinearMath/btStackAlloc.h",
                "LinearMath/TaskScheduler/btThreadSupportInterface.h",
                "LinearMath/btThreads.h",
                "LinearMath/btAlignedAllocator.h",
                "LinearMath/btConvexHull.h",
                "LinearMath/btConvexHullComputer.h",
                "LinearMath/btGeometryUtil.h",
                "LinearMath/btMinMax.h",
                "LinearMath/btTransformUtil.h",
                "LinearMath/btMatrixX.h",
            },
            link = "LinearMath@.3.25",
            preload = {"gomp@.1", "iomp5", "omp", "tbb@.2"}
        )
    },
    target = "org.bytedeco.bullet.LinearMath",
    global = "org.bytedeco.bullet.global.LinearMath"
)
public class LinearMath implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("ATTRIBUTE_ALIGNED16").cppText("#define ATTRIBUTE_ALIGNED16(x) x"))
            .put(new Info("BT_OVERRIDE").cppText("#define BT_OVERRIDE"))
            .put(new Info("btMatrix3x3Data").cppText("#define btMatrix3x3Data btMatrix3x3DoubleData"))
            .put(new Info("btQuaternionData").cppText("#define btQuaternionData btQuaternionDoubleData"))
            .put(new Info("btTransformData").cppText("#define btTransformData btTransformDoubleData"))
            .put(new Info("btVector3Data").cppText("#define btVector3Data btVector3DoubleData"))

            .put(new Info("SIMD_FORCE_INLINE").cppTypes().annotations())

            .put(new Info(
                    "(defined (__APPLE__) && (!defined (BT_USE_DOUBLE_PRECISION)))",
                    "(defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)) || defined(BT_USE_NEON)",
                    "BT_DEBUG_MEMORY_ALLOCATIONS",
                    "BT_DEBUG_OSTREAM",
                    "BT_USE_NEON",
                    "BT_USE_SSE",
                    "ENABLE_INMEMORY_SERIALIZER",
                    "USE_LIBSPE2",
                    "USE_SIMD",
                    "_WIN32",
                    "defined BT_USE_SSE",
                    "defined(BT_USE_NEON)",
                    "defined(BT_USE_SSE) || defined(BT_USE_NEON)",
                    "defined(BT_USE_SSE)",
                    "defined(DEBUG) || defined (_DEBUG)",
                    "defined(_MSC_VER)",
                    "defined(__SPU__) && defined(__CELLOS_LV2__)",
                    "defined\t(__CELLOS_LV2__)"
                ).define(false))
            .put(new Info(
                    "BT_USE_DOUBLE_PRECISION",
                    "defined(BT_USE_DOUBLE_PRECISION)"
                ).define(true))

            .put(new Info("btDefaultSerializer").immutable(true))

            .put(new Info("btAlignedObjectArray<bool>").pointerTypes("btBoolArray"))
            .put(new Info("btAlignedObjectArray<char>").pointerTypes("btCharArray"))
            .put(new Info("btAlignedObjectArray<int>").pointerTypes("btIntArray"))
            .put(new Info("btAlignedObjectArray<unsigned int>").pointerTypes("btUIntArray"))
            .put(new Info("btAlignedObjectArray<float>").pointerTypes("btFloatArray"))
            .put(new Info("btAlignedObjectArray<double>").pointerTypes("btDoubleArray"))
            .put(new Info("btAlignedObjectArray<double>").pointerTypes("btScalarArray"))
            .put(new Info("btAlignedObjectArray<btScalar>").pointerTypes("btScalarArray"))
            .put(new Info("btAlignedObjectArray<btMatrix3x3>").pointerTypes("btMatrix3x3Array"))
            .put(new Info("btAlignedObjectArray<btQuaternion>").pointerTypes("btQuaternionArray"))
            .put(new Info("btAlignedObjectArray<btVector3>").pointerTypes("btVector3Array"))
            .put(new Info("btAlignedObjectArray<btVector4>").pointerTypes("btVector4Array"))
            .put(new Info("btAlignedObjectArray<btPlane>").pointerTypes("btPlaneArray"))
            .put(new Info("btAlignedObjectArray<ConvexH::HalfEdge>").pointerTypes("btConvexHHalfEdgeArray"))
            .put(new Info("ConvexH::HalfEdge").pointerTypes("ConvexH.HalfEdge"))
            .put(new Info("btAlignedObjectArray<btConvexHullComputer::Edge>").pointerTypes("btConvexHullComputerEdgeArray"))
            .put(new Info("btConvexHullComputer::Edge").pointerTypes("btConvexHullComputer.Edge"))
            .put(new Info("btHashMap<btHashPtr,void*>").pointerTypes("btHashMap_btHashPtr_voidPointer"))
            .put(new Info("btHashMap<btHashInt,btAlignedObjectArray<btVector3> >").pointerTypes("btHashMap_btHashInt_btVector3Array"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<int> >").javaNames("btIntArrayArray"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<unsigned int> >").javaNames("btUnsignedIntArrayArray"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<double> >").javaNames("btDoubleArrayArray"))
            .put(new Info("int4").pointerTypes("Int4"))
            .put(new Info("btVectorX<float>").pointerTypes("btVectorXf"))
            .put(new Info("btVectorX<double>").pointerTypes("btVectorXd"))
            .put(new Info("btMatrixX<float>").pointerTypes("btMatrixXf"))
            .put(new Info("btMatrixX<double>").pointerTypes("btMatrixXd"))
            .put(new Info("btVectorXu").cppText("#define btVectorXu btVectorXd"))
            .put(new Info("btMatrixXu").cppText("#define btMatrixXu btMatrixXd"))

            .put(new Info("btAlignedObjectArray.h").linePatterns("\tclass less", "\t};").skip())

            .put(new Info("btIDebugDraw", "btMotionState", "btDefaultMotionState").virtualize())

            .put(new Info(
                    "BT_DECLARE_ALIGNED_ALLOCATOR",
                    "BT_INFINITY",
                    "BT_NAN",
                    "SIMD_EPSILON",
                    "SIMD_INFINITY",
                    "btAlignedObjectArray<ConvexH::HalfEdge>::findBinarySearch",
                    "btAlignedObjectArray<ConvexH::HalfEdge>::findLinearSearch",
                    "btAlignedObjectArray<ConvexH::HalfEdge>::findLinearSearch2",
                    "btAlignedObjectArray<ConvexH::HalfEdge>::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<double> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<double> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<double> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<double> >::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<int> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<int> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<int> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<int> >::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<unsigned int> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<unsigned int> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<unsigned int> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<unsigned int> >::remove",
                    "btAlignedObjectArray<btConvexHullComputer::Edge>::findBinarySearch",
                    "btAlignedObjectArray<btConvexHullComputer::Edge>::findLinearSearch",
                    "btAlignedObjectArray<btConvexHullComputer::Edge>::findLinearSearch2",
                    "btAlignedObjectArray<btConvexHullComputer::Edge>::remove",
                    "btAlignedObjectArray<btMatrix3x3>::findBinarySearch",
                    "btAlignedObjectArray<btPlane>::findBinarySearch",
                    "btAlignedObjectArray<btPlane>::findLinearSearch",
                    "btAlignedObjectArray<btPlane>::findLinearSearch2",
                    "btAlignedObjectArray<btPlane>::remove",
                    "btBulletSerializedArrays",
                    "btGeometryUtil::isInside",
                    "btGetInfinityMask",
                    "btInfMaskConverter",
                    "btThreadSupportInterface::create"
                ).skip())
            ;
    }
}
