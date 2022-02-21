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
            // btScalar.h
            .put(new Info("_WIN32").define(false))
            .put(new Info("defined(_MSC_VER)").define(false))
            .put(new Info("defined\t(__CELLOS_LV2__)").define(false))
            .put(new Info("USE_LIBSPE2").define(false))
            .put(new Info("(defined (__APPLE__) && (!defined (BT_USE_DOUBLE_PRECISION)))").define(false))
            .put(new Info("defined(DEBUG) || defined (_DEBUG)").define(false))
            .put(new Info("defined(BT_USE_SSE) || defined(BT_USE_NEON)").define(false))
            .put(new Info("BT_USE_NEON").define(false))
            .put(new Info("defined(__SPU__) && defined(__CELLOS_LV2__)").define(false))
            .put(new Info("SIMD_FORCE_INLINE").cppTypes().annotations())
            .put(new Info("BT_INFINITY").skip())
            .put(new Info("BT_NAN").skip())
            .put(new Info("BT_USE_DOUBLE_PRECISION").define(false))
            .put(new Info("defined(BT_USE_DOUBLE_PRECISION)").define(false))
            .put(new Info("SIMD_EPSILON").skip())
            .put(new Info("SIMD_INFINITY").skip())
            .put(new Info("ATTRIBUTE_ALIGNED16").cppText("#define ATTRIBUTE_ALIGNED16(x) x"))
            .put(new Info("btInfMaskConverter").skip())

            // btVector3.h
            .put(new Info("BT_DECLARE_ALIGNED_ALLOCATOR").skip())
            .put(new Info("(defined(BT_USE_SSE_IN_API) && defined(BT_USE_SSE)) || defined(BT_USE_NEON)").define(false))
            .put(new Info("btVector3Data").cppText("#define btVector3Data btVector3FloatData"))
            .put(new Info("defined BT_USE_SSE").define(false))

            // btQuaternion.h
            .put(new Info("BT_USE_SSE").define(false))
            .put(new Info("defined(BT_USE_SSE)").define(false))
            .put(new Info("defined(BT_USE_NEON)").define(false))
            .put(new Info("btQuaternionData").cppText("#define btQuaternionData btQuaternionFloatData"))

            // btMatrix3x3.h
            .put(new Info("btMatrix3x3Data").cppText("#define btMatrix3x3Data btMatrix3x3FloatData"))

            // btTransform.h
            .put(new Info("btTransformData").cppText("#define btTransformData btTransformFloatData"))

            // btSerializer.h
            .put(new Info("btBulletSerializedArrays").skip())
            .put(new Info("btHashMap<btHashPtr,void*>").pointerTypes("btHashMap_btHashPtr_voidPointer"))
            .put(new Info("btDefaultSerializer").immutable(true))
            .put(new Info("ENABLE_INMEMORY_SERIALIZER").define(false))

            // btAlignedObjectArray.h
            .put(new Info("btAlignedObjectArray.h").linePatterns("\tclass less", "\t};").skip())
            .put(new Info("btAlignedObjectArray<bool>").pointerTypes("btAlignedObjectArray_bool"))
            .put(new Info("btAlignedObjectArray<char>").pointerTypes("btAlignedObjectArray_char"))
            .put(new Info("btAlignedObjectArray<int>").pointerTypes("btAlignedObjectArray_int"))
            .put(new Info("btAlignedObjectArray<btScalar>").pointerTypes("btAlignedObjectArray_btScalar"))
            .put(new Info("btAlignedObjectArray<btVector3>").pointerTypes("btAlignedObjectArray_btVector3"))
            .put(new Info("btAlignedObjectArray<btVector4>").pointerTypes("btAlignedObjectArray_btVector4"))
            .put(new Info("btAlignedObjectArray<btMatrix3x3>").pointerTypes("btAlignedObjectArray_btMatrix3x3"))
            .put(new Info("btAlignedObjectArray<btMatrix3x3>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btQuaternion>").pointerTypes("btAlignedObjectArray_btQuaternion"))
            ;
    }
}
