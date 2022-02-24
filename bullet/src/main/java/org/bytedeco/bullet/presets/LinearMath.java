package org.bytedeco.bullet.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = javacpp.class,
    value = {
        @Platform(
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
            },
            link = "LinearMath"
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
            .put(new Info("btMatrix3x3Data").cppText("#define btMatrix3x3Data btMatrix3x3FloatData"))
            .put(new Info("btQuaternionData").cppText("#define btQuaternionData btQuaternionFloatData"))
            .put(new Info("btTransformData").cppText("#define btTransformData btTransformFloatData"))
            .put(new Info("btVector3Data").cppText("#define btVector3Data btVector3FloatData"))

            .put(new Info("SIMD_FORCE_INLINE").cppTypes().annotations())

            .put(new Info(
                    "(defined (__APPLE__) && (!defined (BT_USE_DOUBLE_PRECISION)))",
                    "(defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)) || defined(BT_USE_NEON)",
                    "BT_USE_DOUBLE_PRECISION",
                    "BT_USE_NEON",
                    "BT_USE_SSE",
                    "ENABLE_INMEMORY_SERIALIZER",
                    "USE_LIBSPE2",
                    "_WIN32",
                    "defined BT_USE_SSE",
                    "defined(BT_USE_DOUBLE_PRECISION)",
                    "defined(BT_USE_NEON)",
                    "defined(BT_USE_SSE) || defined(BT_USE_NEON)",
                    "defined(BT_USE_SSE)",
                    "defined(DEBUG) || defined (_DEBUG)",
                    "defined(_MSC_VER)",
                    "defined(__SPU__) && defined(__CELLOS_LV2__)",
                    "defined\t(__CELLOS_LV2__)"
                ).define(false))

            .put(new Info("btDefaultSerializer").immutable(true))

            .put(new Info("btAlignedObjectArray<bool>").pointerTypes("btBoolArray"))
            .put(new Info("btAlignedObjectArray<char>").pointerTypes("btCharArray"))
            .put(new Info("btAlignedObjectArray<int>").pointerTypes("btIntArray"))
            .put(new Info("btAlignedObjectArray<btScalar>").pointerTypes("btScalarArray"))
            .put(new Info("btAlignedObjectArray<btMatrix3x3>").pointerTypes("btMatrix3x3Array"))
            .put(new Info("btAlignedObjectArray<btQuaternion>").pointerTypes("btQuaternionArray"))
            .put(new Info("btAlignedObjectArray<btVector3>").pointerTypes("btVector3Array"))
            .put(new Info("btAlignedObjectArray<btVector4>").pointerTypes("btVector4Array"))
            .put(new Info("btHashMap<btHashPtr,void*>").pointerTypes("btHashMap_btHashPtr_voidPointer"))

            .put(new Info("btAlignedObjectArray.h").linePatterns("\tclass less", "\t};").skip())

            .put(new Info(
                    "BT_DECLARE_ALIGNED_ALLOCATOR",
                    "BT_INFINITY",
                    "BT_NAN",
                    "SIMD_EPSILON",
                    "SIMD_INFINITY",
                    "btAlignedObjectArray<btMatrix3x3>::findBinarySearch",
                    "btBulletSerializedArrays",
                    "btInfMaskConverter"
                ).skip())
            ;
    }
}
