package org.bytedeco.bullet.presets;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;

@Properties(
    inherit = BulletCollision.class,
    value = {
        @Platform(
            include = {
                "LinearMath/btAlignedObjectArray.h",
                "BulletDynamics/Dynamics/btRigidBody.h",
                "BulletDynamics/Dynamics/btDynamicsWorld.h",
                "BulletDynamics/ConstraintSolver/btContactSolverInfo.h",
                "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h",
                "BulletDynamics/Dynamics/btSimpleDynamicsWorld.h",
                "BulletDynamics/ConstraintSolver/btPoint2PointConstraint.h",
                "BulletDynamics/ConstraintSolver/btHingeConstraint.h",
                "BulletDynamics/ConstraintSolver/btConeTwistConstraint.h",
                "BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.h",
                "BulletDynamics/ConstraintSolver/btSliderConstraint.h",
                "BulletDynamics/ConstraintSolver/btGeneric6DofSpringConstraint.h",
                "BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h",
                "BulletDynamics/ConstraintSolver/btUniversalConstraint.h",
                "BulletDynamics/ConstraintSolver/btHinge2Constraint.h",
                "BulletDynamics/ConstraintSolver/btGearConstraint.h",
                "BulletDynamics/ConstraintSolver/btFixedConstraint.h",
                "BulletDynamics/ConstraintSolver/btConstraintSolver.h",
                "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h",
                "BulletDynamics/Vehicle/btVehicleRaycaster.h",
                "BulletDynamics/Vehicle/btWheelInfo.h",
                "BulletDynamics/Vehicle/btRaycastVehicle.h",
            },
            link = "BulletDynamics"
        )
    },
    target = "org.bytedeco.bullet.BulletDynamics",
    global = "org.bytedeco.bullet.global.BulletDynamics"
)
public class BulletDynamics implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("btRigidBodyData")
                .cppText("#define btRigidBodyData btRigidBodyFloatData"))
            .put(new Info("defined(BT_CLAMP_VELOCITY_TO) && BT_CLAMP_VELOCITY_TO > 0")
                .define(false))
            .put(new Info("btAlignedObjectArray<btRigidBody*>")
                .pointerTypes("btAlignedObjectArray_btRigidBodyPointer"))
            .put(new Info("IN_PARALLELL_SOLVER").define(false))
            .put(new Info("BT_BACKWARDS_COMPATIBLE_SERIALIZATION").define(true))
            .put(new Info("btConstraintInfo1").skip())
            .put(new Info("btConstraintInfo2").skip())
            .put(new Info("btPoint2PointConstraintFloatData::m_typeConstraintData").skip())
            .put(new Info("btPoint2PointConstraintDoubleData::m_typeConstraintData").skip())
            .put(new Info("btPoint2PointConstraintDoubleData2::m_typeConstraintData").skip())
            .put(new Info("btPoint2PointConstraintData2")
                .cppText("#define btPoint2PointConstraintData2 " +
                    "btPoint2PointConstraintDoubleData2"))
            .put(new Info("btHingeConstraintFloatData::m_typeConstraintData").skip())
            .put(new Info("btHingeConstraintDoubleData::m_typeConstraintData").skip())
            .put(new Info("btHingeConstraintDoubleData2::m_typeConstraintData").skip())
            .put(new Info("btHingeConstraintData")
                .cppText("#define btHingeConstraintData " +
                    "btHingeConstraintFloatData"))
            .put(new Info("btConeTwistConstraint::solveConstraintObsolete").skip())
            .put(new Info("btConeTwistConstraintData::m_typeConstraintData").skip())
            .put(new Info("btConeTwistConstraintDoubleData::m_typeConstraintData").skip())
            .put(new Info("btConeTwistConstraintData2")
                .cppText("#define btConeTwistConstraintData2 " +
                    "btConeTwistConstraintData"))
            .put(new Info("btGeneric6DofConstraintData::m_typeConstraintData").skip())
            .put(new Info("btGeneric6DofConstraintDoubleData2::m_typeConstraintData").skip())
            .put(new Info("btGeneric6DofConstraintData2")
                .cppText("#define btGeneric6DofConstraintData2 " +
                    "btGeneric6DofConstraintDoubleData2"))
            .put(new Info("btSliderConstraintData::m_typeConstraintData").skip())
            .put(new Info("btSliderConstraintDoubleData::m_typeConstraintData").skip())
            .put(new Info("btSliderConstraintData2")
                .cppText("#define btSliderConstraintData2 " +
                    "btSliderConstraintDoubleData2"))
            .put(new Info("btGeneric6DofSpringConstraintData2")
                .cppText("#define btGeneric6DofSpringConstraintData2 " +
                    "btGeneric6DofSpringConstraintData"))
            .put(new Info("btGeneric6DofSpring2ConstraintData::m_typeConstraintData").skip())
            .put(new Info("btGeneric6DofSpring2ConstraintDoubleData2::" +
                "m_typeConstraintData").skip())
            .put(new Info("btGeneric6DofSpring2ConstraintData2")
                .cppText("#define btGeneric6DofSpring2ConstraintData2 " +
                    "btGeneric6DofSpring2ConstraintData"))
            .put(new Info("btGearConstraintFloatData::m_typeConstraintData").skip())
            .put(new Info("btGearConstraintDoubleData::m_typeConstraintData").skip())
            .put(new Info("btGearConstraintData")
                .cppText("#define btGearConstraintData btGearConstraintFloatData"))
            .put(new Info("btSingleConstraintRowSolver").skip())
            .put(new Info("btRaycastVehicle::m_wheelInfo").skip())
            ;
    }
}
