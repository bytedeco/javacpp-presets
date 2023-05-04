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
    inherit = Bullet3Collision.class,
    value = {
        @Platform(
            include = {
                "Bullet3Common/b3AlignedObjectArray.h",
                "Bullet3Dynamics/ConstraintSolver/b3ContactSolverInfo.h",
                "Bullet3Dynamics/ConstraintSolver/b3SolverBody.h",
                "Bullet3Dynamics/ConstraintSolver/b3SolverConstraint.h",
                "Bullet3Dynamics/ConstraintSolver/b3TypedConstraint.h",
                "Bullet3Dynamics/ConstraintSolver/b3FixedConstraint.h",
                "Bullet3Dynamics/ConstraintSolver/b3Generic6DofConstraint.h",
                "Bullet3Dynamics/ConstraintSolver/b3JacobianEntry.h",
                "Bullet3Dynamics/ConstraintSolver/b3PgsJacobiSolver.h",
                "Bullet3Dynamics/ConstraintSolver/b3Point2PointConstraint.h",
                "Bullet3Dynamics/shared/b3ContactConstraint4.h",
                "Bullet3Dynamics/shared/b3Inertia.h",
                "Bullet3Dynamics/shared/b3IntegrateTransforms.h",
                "Bullet3Dynamics/b3CpuRigidBodyPipeline.h",
            },
            link = "Bullet3Dynamics@.3.25"
        )
    },
    target = "org.bytedeco.bullet.Bullet3Dynamics",
    global = "org.bytedeco.bullet.global.Bullet3Dynamics"
)
public class Bullet3Dynamics implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info(
                    "IN_PARALLELL_SOLVER",
                    "b3Point2PointConstraintData"
                ).cppTypes().translate(false))

            .put(new Info("b3AlignedObjectArray<b3TypedConstraint*>").pointerTypes("b3TypedConstraintArray"))
            .put(new Info("b3ContactConstraint4_t").pointerTypes("b3ContactConstraint4"))

            .put(new Info(
                    "b3CpuRigidBodyPipeline::addConstraint",
                    "b3CpuRigidBodyPipeline::castRays",
                    "b3CpuRigidBodyPipeline::copyConstraintsToHost",
                    "b3CpuRigidBodyPipeline::createFixedConstraint",
                    "b3CpuRigidBodyPipeline::createPoint2PointConstraint",
                    "b3CpuRigidBodyPipeline::registerConvexPolyhedron",
                    "b3CpuRigidBodyPipeline::removeConstraint",
                    "b3CpuRigidBodyPipeline::removeConstraintByUid",
                    "b3CpuRigidBodyPipeline::reset",
                    "b3CpuRigidBodyPipeline::setGravity",
                    "b3CpuRigidBodyPipeline::writeAllInstancesToGpu",
                    "b3RotationalLimitMotor::solveAngularLimits",
                    "b3TranslationalLimitMotor::solveLinearAxis"
                ).skip())
            ;
    }
}
