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
    inherit = BulletCollision.class,
    value = {
        @Platform(
            include = {
                "LinearMath/btAlignedObjectArray.h",
                "BulletDynamics/Dynamics/btActionInterface.h",
                "BulletDynamics/Dynamics/btRigidBody.h",
                "BulletDynamics/Dynamics/btDynamicsWorld.h",
                "BulletDynamics/ConstraintSolver/btContactSolverInfo.h",
                "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h",
                "BulletDynamics/Dynamics/btSimpleDynamicsWorld.h",
                "BulletDynamics/ConstraintSolver/btConstraintSolver.h",
                "BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h",
                "BulletDynamics/Dynamics/btSimulationIslandManagerMt.h",
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
                "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h",
                "BulletDynamics/ConstraintSolver/btSolverConstraint.h",
                "BulletDynamics/ConstraintSolver/btSolverBody.h",
                "BulletDynamics/ConstraintSolver/btTypedConstraint.h",
                "BulletDynamics/ConstraintSolver/btBatchedConstraints.h",
                "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h",
                "BulletDynamics/Vehicle/btVehicleRaycaster.h",
                "BulletDynamics/Vehicle/btWheelInfo.h",
                "BulletDynamics/Vehicle/btRaycastVehicle.h",
                "BulletDynamics/Featherstone/btMultiBodyJointFeedback.h",
                "BulletDynamics/Featherstone/btMultiBodyLink.h",
                "BulletDynamics/Featherstone/btMultiBody.h",
                "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h",
                "BulletDynamics/Featherstone/btMultiBodyConstraint.h",
                "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h",
                "BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h",
            },
            link = "BulletDynamics@.3.20"
        )
    },
    target = "org.bytedeco.bullet.BulletDynamics",
    global = "org.bytedeco.bullet.global.BulletDynamics"
)
public class BulletDynamics implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("btConeTwistConstraintData2").cppText("#define btConeTwistConstraintData2 btConeTwistConstraintData"))
            .put(new Info("btGearConstraintData").cppText("#define btGearConstraintData btGearConstraintFloatData"))
            .put(new Info("btGeneric6DofConstraintData2").cppText("#define btGeneric6DofConstraintData2 btGeneric6DofConstraintDoubleData2"))
            .put(new Info("btGeneric6DofSpring2ConstraintData2").cppText("#define btGeneric6DofSpring2ConstraintData2 btGeneric6DofSpring2ConstraintData"))
            .put(new Info("btGeneric6DofSpringConstraintData2").cppText("#define btGeneric6DofSpringConstraintData2 btGeneric6DofSpringConstraintData"))
            .put(new Info("btHingeConstraintData").cppText("#define btHingeConstraintData btHingeConstraintFloatData"))
            .put(new Info("btMultiBodyData").cppText("#define btMultiBodyData btMultiBodyFloatData"))
            .put(new Info("btMultiBodyLinkColliderData").cppText("#define btMultiBodyLinkColliderData btMultiBodyLinkColliderFloatData"))
            .put(new Info("btMultiBodyLinkData").cppText("#define btMultiBodyLinkData btMultiBodyLinkFloatData"))
            .put(new Info("btPoint2PointConstraintData2").cppText("#define btPoint2PointConstraintData2 btPoint2PointConstraintDoubleData2"))
            .put(new Info("btRigidBodyData").cppText("#define btRigidBodyData btRigidBodyFloatData"))
            .put(new Info("btSimdScalar").cppText("#define btSimdScalar btScalar"))
            .put(new Info("btSliderConstraintData2").cppText("#define btSliderConstraintData2 btSliderConstraintDoubleData2"))
            .put(new Info("btTypedConstraintData2").cppText("#define btTypedConstraintData2 btTypedConstraintFloatData"))

            .put(new Info(
                    "IN_PARALLELL_SOLVER",
                    "USE_SIMD",
                    "defined(BT_CLAMP_VELOCITY_TO) && BT_CLAMP_VELOCITY_TO > 0"
                ).define(false))

            .put(new Info(
                    "BT_BACKWARDS_COMPATIBLE_SERIALIZATION"
                ).define(true))

            .put(new Info("btAlignedObjectArray<btRigidBody*>").pointerTypes("btRigidBodyArray"))

            .put(new Info(
                    "DeformableBodyInplaceSolverIslandCallback",
                    "InplaceSolverIslandCallback",
                    "MultiBodyInplaceSolverIslandCallback",
                    "btBatchedConstraints::m_batches",
                    "btBatchedConstraints::m_phases",
                    "btConeTwistConstraint::solveConstraintObsolete",
                    "btConeTwistConstraintData::m_typeConstraintData",
                    "btConeTwistConstraintDoubleData::m_typeConstraintData",
                    "btConstraintArray",
                    "btConstraintInfo1",
                    "btConstraintInfo2",
                    "btGearConstraintDoubleData::m_typeConstraintData",
                    "btGearConstraintFloatData::m_typeConstraintData",
                    "btGeneric6DofConstraintData::m_typeConstraintData",
                    "btGeneric6DofConstraintDoubleData2::m_typeConstraintData",
                    "btGeneric6DofSpring2ConstraintData::m_typeConstraintData",
                    "btGeneric6DofSpring2ConstraintDoubleData2::m_typeConstraintData",
                    "btHingeConstraintDoubleData2::m_typeConstraintData",
                    "btHingeConstraintDoubleData::m_typeConstraintData",
                    "btHingeConstraintFloatData::m_typeConstraintData",
                    "btMultiBodyConstraint::createConstraintRows",
                    "btMultiBodyDynamicsWorld::getAnalyticsData",
                    "btMultiBodyJacobianData::m_solverBodyPool",
                    "btPoint2PointConstraintDoubleData2::m_typeConstraintData",
                    "btPoint2PointConstraintDoubleData::m_typeConstraintData",
                    "btPoint2PointConstraintFloatData::m_typeConstraintData",
                    "btRaycastVehicle::m_wheelInfo",
                    "btSequentialImpulseConstraintSolverMt::internalConvertMultipleJoints",
                    "btSimulationIslandManagerMt::Island::bodyArray",
                    "btSimulationIslandManagerMt::Island::constraintArray",
                    "btSimulationIslandManagerMt::Island::manifoldArray",
                    "btSimulationIslandManagerMt::IslandDispatchFunc",
                    "btSimulationIslandManagerMt::buildAndProcessIslands",
                    "btSimulationIslandManagerMt::parallelIslandDispatch",
                    "btSimulationIslandManagerMt::serialIslandDispatch",
                    "btSingleConstraintRowSolver",
                    "btSliderConstraintData::m_typeConstraintData",
                    "btSliderConstraintDoubleData::m_typeConstraintData",
                    "btSolverInfo"
                ).skip())
            ;
    }
}
