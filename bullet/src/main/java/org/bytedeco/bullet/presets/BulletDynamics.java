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
                "BulletDynamics/ConstraintSolver/btBatchedConstraints.h",
                "BulletDynamics/ConstraintSolver/btConeTwistConstraint.h",
                "BulletDynamics/ConstraintSolver/btConstraintSolver.h",
                "BulletDynamics/ConstraintSolver/btContactConstraint.h",
                "BulletDynamics/ConstraintSolver/btContactSolverInfo.h",
                "BulletDynamics/ConstraintSolver/btFixedConstraint.h",
                "BulletDynamics/ConstraintSolver/btGearConstraint.h",
                "BulletDynamics/ConstraintSolver/btGeneric6DofConstraint.h",
                "BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h",
                "BulletDynamics/ConstraintSolver/btGeneric6DofSpringConstraint.h",
                "BulletDynamics/ConstraintSolver/btHinge2Constraint.h",
                "BulletDynamics/ConstraintSolver/btHingeConstraint.h",
                "BulletDynamics/ConstraintSolver/btJacobianEntry.h",
                "BulletDynamics/ConstraintSolver/btNNCGConstraintSolver.h",
                "BulletDynamics/ConstraintSolver/btPoint2PointConstraint.h",
                "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h",
                "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h",
                "BulletDynamics/ConstraintSolver/btSliderConstraint.h",
                "BulletDynamics/ConstraintSolver/btSolve2LinearConstraint.h",
                "BulletDynamics/ConstraintSolver/btSolverBody.h",
                "BulletDynamics/ConstraintSolver/btSolverConstraint.h",
                "BulletDynamics/Dynamics/btRigidBody.h",
                "BulletDynamics/ConstraintSolver/btTypedConstraint.h",
                "BulletDynamics/ConstraintSolver/btUniversalConstraint.h",
                "BulletDynamics/Dynamics/btActionInterface.h",
                "BulletDynamics/Dynamics/btDynamicsWorld.h",
                "BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h",
                "BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h",
                "BulletDynamics/Dynamics/btSimpleDynamicsWorld.h",
                "BulletDynamics/Dynamics/btSimulationIslandManagerMt.h",
                "BulletDynamics/Vehicle/btRaycastVehicle.h",
                "BulletDynamics/Vehicle/btVehicleRaycaster.h",
                "BulletDynamics/Vehicle/btWheelInfo.h",
                "BulletDynamics/Featherstone/btMultiBody.h",
                "BulletDynamics/Featherstone/btMultiBodyConstraint.h",
                "BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h",
                "BulletDynamics/Featherstone/btMultiBodyDynamicsWorld.h",
                "BulletDynamics/Featherstone/btMultiBodySolverConstraint.h",
                "BulletDynamics/Featherstone/btMultiBodyFixedConstraint.h",
                "BulletDynamics/Featherstone/btMultiBodyGearConstraint.h",
                "BulletDynamics/Featherstone/btMultiBodyInplaceSolverIslandCallback.h",
                "BulletDynamics/Featherstone/btMultiBodyJointFeedback.h",
                "BulletDynamics/Featherstone/btMultiBodyJointLimitConstraint.h",
                "BulletDynamics/Featherstone/btMultiBodyJointMotor.h",
                "BulletDynamics/Featherstone/btMultiBodyLink.h",
                "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h",
                "BulletDynamics/Featherstone/btMultiBodyMLCPConstraintSolver.h",
                "BulletDynamics/Featherstone/btMultiBodyPoint2Point.h",
                "BulletDynamics/Featherstone/btMultiBodySliderConstraint.h",
                "BulletDynamics/Featherstone/btMultiBodySphericalJointMotor.h",
                "BulletDynamics/MLCPSolvers/btDantzigLCP.h",
                "BulletDynamics/MLCPSolvers/btDantzigSolver.h",
                "BulletDynamics/MLCPSolvers/btLemkeAlgorithm.h",
                "BulletDynamics/MLCPSolvers/btLemkeSolver.h",
                "BulletDynamics/MLCPSolvers/btMLCPSolver.h",
                "BulletDynamics/MLCPSolvers/btMLCPSolverInterface.h",
                "BulletDynamics/MLCPSolvers/btSolveProjectedGaussSeidel.h",
                "BulletDynamics/Character/btCharacterControllerInterface.h",
                "BulletDynamics/Character/btKinematicCharacterController.h",
            },
            link = "BulletDynamics@.3.25"
        )
    },
    target = "org.bytedeco.bullet.BulletDynamics",
    global = "org.bytedeco.bullet.global.BulletDynamics"
)
public class BulletDynamics implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("btSimdScalar").cppText("#define btSimdScalar btScalar"))

            .put(new Info(
                    "btConeTwistConstraintData2",
                    "btGearConstraintData",
                    "btGeneric6DofConstraintData2",
                    "btGeneric6DofSpring2ConstraintData2",
                    "btGeneric6DofSpringConstraintData2",
                    "btHingeConstraintData",
                    "btMultiBodyData",
                    "btMultiBodyLinkColliderData",
                    "btMultiBodyLinkData",
                    "btPoint2PointConstraintData2",
                    "btRigidBodyData",
                    "btSliderConstraintData2",
                    "btTypedConstraintData2"
                ).cppTypes().translate(false))

            .put(new Info(
                    "IN_PARALLELL_SOLVER",
                    "defined(BT_CLAMP_VELOCITY_TO) && BT_CLAMP_VELOCITY_TO > 0"
                ).define(false))

            .put(new Info(
                    "BT_BACKWARDS_COMPATIBLE_SERIALIZATION"
                ).define(true))

            .put(new Info("btAlignedObjectArray<btBatchedConstraints::Range>").pointerTypes("RangeArray"))
            .put(new Info("btAlignedObjectArray<btMultiBodyConstraint*>").pointerTypes("btMultiBodyConstraintArray"))
            .put(new Info("btAlignedObjectArray<btMultiBodySolverConstraint>").pointerTypes("btMultiBodySolverConstraintArray"))
            .put(new Info("btAlignedObjectArray<btRigidBody*>").pointerTypes("btRigidBodyArray"))
            .put(new Info("btAlignedObjectArray<btSequentialImpulseConstraintSolverMt::JointParams>").pointerTypes("JointParamsArray"))
            .put(new Info("btAlignedObjectArray<btSimulationIslandManagerMt::Island*>").pointerTypes("IslandArray"))
            .put(new Info("btAlignedObjectArray<btSolverAnalyticsData>").pointerTypes("btSolverAnalyticsDataArray"))
            .put(new Info("btAlignedObjectArray<btSolverBody>").pointerTypes("btSolverBodyArray"))
            .put(new Info("btAlignedObjectArray<btSolverConstraint>").pointerTypes("btSolverConstraintArray"))
            .put(new Info("btAlignedObjectArray<btTypedConstraint*>").pointerTypes("btTypedConstraintArray"))
            .put(new Info("btAlignedObjectArray<btWheelInfo>").pointerTypes("btWheelInfoArray"))
            .put(new Info("btBatchedConstraints::Range").pointerTypes("btBatchedConstraints.Range"))
            .put(new Info("btConstraintArray").pointerTypes("btSolverConstraintArray"))
            .put(new Info("btMultiBodyConstraintArray").pointerTypes("btMultiBodySolverConstraintArray"))
            .put(new Info("btSequentialImpulseConstraintSolverMt::JointParams").pointerTypes("btSequentialImpulseConstraintSolverMt.JointParams"))
            .put(new Info("btSimulationIslandManagerMt::Island").pointerTypes("btSimulationIslandManagerMt.Island"))

            .put(new Info(
                    "btAlignedObjectArray<btBatchedConstraints::Range>::findBinarySearch",
                    "btAlignedObjectArray<btBatchedConstraints::Range>::findLinearSearch",
                    "btAlignedObjectArray<btBatchedConstraints::Range>::findLinearSearch2",
                    "btAlignedObjectArray<btBatchedConstraints::Range>::remove",
                    "btAlignedObjectArray<btMultiBodySolverConstraint>::findBinarySearch",
                    "btAlignedObjectArray<btMultiBodySolverConstraint>::findLinearSearch",
                    "btAlignedObjectArray<btMultiBodySolverConstraint>::findLinearSearch2",
                    "btAlignedObjectArray<btMultiBodySolverConstraint>::remove",
                    "btAlignedObjectArray<btSequentialImpulseConstraintSolverMt::JointParams>::findBinarySearch",
                    "btAlignedObjectArray<btSequentialImpulseConstraintSolverMt::JointParams>::findLinearSearch",
                    "btAlignedObjectArray<btSequentialImpulseConstraintSolverMt::JointParams>::findLinearSearch2",
                    "btAlignedObjectArray<btSequentialImpulseConstraintSolverMt::JointParams>::remove",
                    "btAlignedObjectArray<btSolverAnalyticsData>::findBinarySearch",
                    "btAlignedObjectArray<btSolverAnalyticsData>::findLinearSearch",
                    "btAlignedObjectArray<btSolverAnalyticsData>::findLinearSearch2",
                    "btAlignedObjectArray<btSolverAnalyticsData>::remove",
                    "btAlignedObjectArray<btSolverBody>::findBinarySearch",
                    "btAlignedObjectArray<btSolverBody>::findLinearSearch",
                    "btAlignedObjectArray<btSolverBody>::findLinearSearch2",
                    "btAlignedObjectArray<btSolverBody>::remove",
                    "btAlignedObjectArray<btSolverConstraint>::findBinarySearch",
                    "btAlignedObjectArray<btSolverConstraint>::findLinearSearch",
                    "btAlignedObjectArray<btSolverConstraint>::findLinearSearch2",
                    "btAlignedObjectArray<btSolverConstraint>::remove",
                    "btAlignedObjectArray<btWheelInfo>::findBinarySearch",
                    "btAlignedObjectArray<btWheelInfo>::findLinearSearch",
                    "btAlignedObjectArray<btWheelInfo>::findLinearSearch2",
                    "btAlignedObjectArray<btWheelInfo>::remove",
                    "btDantzigScratchMemory::Arows",
                    "btSequentialImpulseConstraintSolver::getSSE2ConstraintRowSolverGeneric",
                    "btSequentialImpulseConstraintSolver::getSSE2ConstraintRowSolverLowerLimit",
                    "btSequentialImpulseConstraintSolver::getSSE4_1ConstraintRowSolverGeneric",
                    "btSequentialImpulseConstraintSolver::getSSE4_1ConstraintRowSolverLowerLimit"
                ).skip())
            ;
    }
}
