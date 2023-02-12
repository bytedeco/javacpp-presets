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
            include = {
                "Bullet3Common/b3AlignedObjectArray.h",
                "Bullet3Common/b3CommandLineArgs.h",
                "Bullet3Common/b3FileUtils.h",
                "Bullet3Common/b3HashMap.h",
                "Bullet3Common/b3Logging.h",
                "Bullet3Common/b3Scalar.h",
                "Bullet3Common/b3Vector3.h",
                "Bullet3Common/b3QuadWord.h",
                "Bullet3Common/b3Quaternion.h",
                "Bullet3Common/b3Matrix3x3.h",
                "Bullet3Common/b3MinMax.h",
                "Bullet3Common/b3ResizablePool.h",
                "Bullet3Common/b3Transform.h",
                "Bullet3Common/b3TransformUtil.h",
                "Bullet3Common/shared/b3Float4.h",
                "Bullet3Common/shared/b3Int2.h",
                "Bullet3Common/shared/b3Int4.h",
                "Bullet3Common/shared/b3Quat.h",
                "Bullet3Common/shared/b3Mat3x3.h",
                "Bullet3Common/shared/b3PlatformDefinitions.h",
            },
            link = "Bullet3Common@.3.25",
            preload = {"gomp@.1", "iomp5", "omp", "tbb@.2"}
        )
    },
    target = "org.bytedeco.bullet.Bullet3Common",
    global = "org.bytedeco.bullet.global.Bullet3Common"
)
public class Bullet3Common implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public static void mapArrays(InfoMap infoMap, String... typeNames) {
        for (String typeName: typeNames) {
            String cppName = "b3AlignedObjectArray<" + typeName + ">";
            String javaName = typeName + "Array";
            infoMap.put(new Info(cppName).pointerTypes(javaName));
            infoMap.put(new Info(
                    cppName + "::findBinarySearch",
                    cppName + "::findLinearSearch",
                    cppName + "::findLinearSearch2",
                    cppName + "::remove"
                ).skip());
        }
    }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("B3_ATTRIBUTE_ALIGNED16").cppText("#define B3_ATTRIBUTE_ALIGNED16(x) x"))
            .put(new Info("B3_DECLARE_ALIGNED_ALLOCATOR").cppText("#define B3_DECLARE_ALIGNED_ALLOCATOR()"))
            .put(new Info("B3_FORCE_INLINE").cppText("#define B3_FORCE_INLINE"))

            .put(new Info("B3_EPSILON").cppTypes("float"))
            .put(new Info("B3_INFINITY").cppTypes("float"))
            .put(new Info("B3_LARGE_FLOAT").cppTypes("float"))

            .put(new Info(
                    "__global",
                    "__inline"
                ).cppTypes().annotations())

            .put(new Info(
                    "B3_STATIC",
                    "USE_SIMD",
                    "b3Cross3",
                    "b3Dot3F4",
                    "b3Float4",
                    "b3Float4ConstArg",
                    "b3MakeFloat4",
                    "b3Mat3x3",
                    "b3Mat3x3ConstArg",
                    "b3Matrix3x3Data",
                    "b3Quat",
                    "b3QuatConstArg",
                    "b3SimdScalar",
                    "b3TransformData",
                    "b3Vector3Data"
                ).cppTypes().translate(false))

            .put(new Info(
                    "(defined(B3_USE_SSE_IN_API) && defined(B3_USE_SSE)) || defined(B3_USE_NEON)",
                    "B3_USE_DOUBLE_PRECISION",
                    "B3_USE_NEON",
                    "B3_USE_SSE",
                    "_WIN32",
                    "__clang__",
                    "defined B3_USE_SSE",
                    "defined(B3_USE_DOUBLE_PRECISION) || defined(B3_FORCE_DOUBLE_FUNCTIONS)",
                    "defined(B3_USE_DOUBLE_PRECISION)",
                    "defined(B3_USE_SSE) || defined(B3_USE_NEON)",
                    "defined(B3_USE_SSE)",
                    "defined(B3_USE_SSE_IN_API) && defined(B3_USE_SSE)",
                    "defined(__SPU__) && defined(__CELLOS_LV2__)"
                ).define(false))
            .put(new Info(
                    "__cplusplus"
                ).define(true))

            .put(new Info("b3AlignedObjectArray<unsigned char>").pointerTypes("b3UnsignedCharArray"))
            .put(new Info("b3AlignedObjectArray<int>").pointerTypes("b3IntArray"))
            .put(new Info("b3AlignedObjectArray<unsigned int>").pointerTypes("b3UnsignedIntArray"))
            .put(new Info("b3AlignedObjectArray<float>").pointerTypes("b3FloatArray"))
            .put(new Info("b3AlignedObjectArray<b3Int2>").pointerTypes("b3Int2Array"))
            .put(new Info("b3AlignedObjectArray<b3Int4>").pointerTypes("b3Int4Array"))
            .put(new Info("b3AlignedObjectArray<b3Vector3>").pointerTypes("b3Vector3Array"))
            .put(new Info("b3AlignedObjectArray<b3BroadphasePair>").pointerTypes("b3Int4Array"))

            .put(new Info("b3AlignedObjectArray.h").linePatterns("\tclass less", "\t};").skip())

            .put(new Info(
                    "b3AlignedObjectArray<b3Int2>::findBinarySearch",
                    "b3AlignedObjectArray<b3Int2>::findLinearSearch",
                    "b3AlignedObjectArray<b3Int2>::findLinearSearch2",
                    "b3AlignedObjectArray<b3Int2>::remove",
                    "b3AlignedObjectArray<b3Int4>::findBinarySearch",
                    "b3AlignedObjectArray<b3Int4>::findLinearSearch",
                    "b3AlignedObjectArray<b3Int4>::findLinearSearch2",
                    "b3AlignedObjectArray<b3Int4>::remove"
                ).skip())
            ;
    }
}
