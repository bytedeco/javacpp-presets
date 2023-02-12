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
    inherit = BulletDynamics.class,
    value = {
        @Platform(
            include = {
                "LinearMath/btAlignedObjectArray.h",
                "BulletSoftBody/btSoftBody.h",
                "BulletSoftBody/btCGProjection.h",
                "BulletSoftBody/btConjugateGradient.h",
                "BulletSoftBody/btConjugateResidual.h",
                "BulletSoftBody/btDefaultSoftBodySolver.h",
                "BulletSoftBody/btDeformableBackwardEulerObjective.h",
                "BulletSoftBody/btDeformableBodySolver.h",
                "BulletSoftBody/btDeformableContactConstraint.h",
                "BulletSoftBody/btDeformableContactProjection.h",
                "BulletSoftBody/btDeformableLagrangianForce.h",
                "BulletSoftBody/btDeformableCorotatedForce.h",
                "BulletSoftBody/btDeformableGravityForce.h",
                "BulletSoftBody/btDeformableLinearElasticityForce.h",
                "BulletSoftBody/btDeformableMassSpringForce.h",
                "BulletSoftBody/btDeformableMousePickingForce.h",
                "BulletSoftBody/btDeformableMultiBodyConstraintSolver.h",
                "BulletSoftBody/btDeformableMultiBodyDynamicsWorld.h",
                "BulletSoftBody/btDeformableNeoHookeanForce.h",
                "BulletSoftBody/btKrylovSolver.h",
                "BulletSoftBody/btPreconditioner.h",
                "BulletSoftBody/btSoftBodyConcaveCollisionAlgorithm.h",
                "BulletSoftBody/btSoftBodyData.h",
                "BulletSoftBody/btSoftBodyHelpers.h",
                "BulletSoftBody/btSoftBodyInternals.h",
                "BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h",
                "BulletSoftBody/btSoftBodySolvers.h",
                "BulletSoftBody/btSoftBodySolverVertexBuffer.h",
                "BulletSoftBody/btSoftMultiBodyDynamicsWorld.h",
                "BulletSoftBody/btSoftRigidCollisionAlgorithm.h",
                "BulletSoftBody/btSoftRigidDynamicsWorld.h",
                "BulletSoftBody/btSoftSoftCollisionAlgorithm.h",
                "BulletSoftBody/btSparseSDF.h",
                "BulletSoftBody/DeformableBodyInplaceSolverIslandCallback.h",
            },
            link = "BulletSoftBody@.3.25"
        )
    },
    target = "org.bytedeco.bullet.BulletSoftBody",
    global = "org.bytedeco.bullet.global.BulletSoftBody"
)
public class BulletSoftBody implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("btSoftBodyData").cppTypes().translate(false))

            .put(new Info(
                    "USE_MGS"
                ).define(false))

            .put(new Info("btSoftBodySolver::SolverTypes").enumerate())
            .put(new Info("btVertexBufferDescriptor::BufferTypes").enumerate())

            .put(new Info("btDeformableBackwardEulerObjective").immutable(true))

            .put(new Info("TVStack").pointerTypes("btVector3Array"))
            .put(new Info("btAlignedObjectArray<LagrangeMultiplier>").pointerTypes("LagrangeMultiplierArray"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceNodeContactConstraint> >").pointerTypes("btDeformableFaceNodeContactConstraintArrayArray"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceRigidContactConstraint> >").pointerTypes("btDeformableFaceRigidContactConstraintArrayArray"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeAnchorConstraint> >").pointerTypes("btDeformableNodeAnchorConstraintArrayArray"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeRigidContactConstraint> >").pointerTypes("btDeformableNodeRigidContactConstraintArrayArray"))
            .put(new Info("btAlignedObjectArray<btAlignedObjectArray<btDeformableStaticConstraint> >").pointerTypes("btDeformableStaticConstraintArrayArray"))
            .put(new Info("btAlignedObjectArray<btDeformableContactConstraint*>").pointerTypes("btDeformableContactConstraintArray"))
            .put(new Info("btAlignedObjectArray<btDeformableFaceNodeContactConstraint>").pointerTypes("btDeformableFaceNodeContactConstraintArray"))
            .put(new Info("btAlignedObjectArray<btDeformableFaceRigidContactConstraint>").pointerTypes("btDeformableFaceRigidContactConstraintArray"))
            .put(new Info("btAlignedObjectArray<btDeformableLagrangianForce*>").pointerTypes("btDeformableLagrangianForceArray"))
            .put(new Info("btAlignedObjectArray<btDeformableNodeAnchorConstraint>").pointerTypes("btDeformableNodeAnchorConstraintArray"))
            .put(new Info("btAlignedObjectArray<btDeformableNodeRigidContactConstraint>").pointerTypes("btDeformableNodeRigidContactConstraintArray"))
            .put(new Info("btAlignedObjectArray<btDeformableStaticConstraint>").pointerTypes("btDeformableStaticConstraintArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody*>").pointerTypes("btSoftBodyArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Anchor>").pointerTypes("btSoftBodyAnchorArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Cluster*>").pointerTypes("btSoftBodyClusterArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::DeformableFaceNodeContact>").pointerTypes("btSoftBodyDeformableFaceNodeContactArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::DeformableFaceRigidContact>").pointerTypes("btSoftBodyDeformableFaceRigidContactArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::DeformableNodeRigidAnchor>").pointerTypes("btSoftBodyDeformableNodeRigidAnchorArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::DeformableNodeRigidContact>").pointerTypes("btSoftBodyDeformableNodeRigidContactArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Face>").pointerTypes("btSoftBodyFaceArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Joint*>").pointerTypes("btSoftBodyJointArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Link>").pointerTypes("btSoftBodyLinkArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Material*>").pointerTypes("btSoftBodyMaterialArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Node*>").pointerTypes("btSoftBodyNodePointerArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Node>").pointerTypes("btSoftBodyNodeArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Note>").pointerTypes("btSoftBodyNoteArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RContact>").pointerTypes("btSoftBodyRContactArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderFace>").pointerTypes("btSoftBodyRenderFaceArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderNode>").pointerTypes("btSoftBodyRenderNodeArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::SContact>").pointerTypes("btSoftBodySContactArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Tetra>").pointerTypes("btSoftBodyTetraArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::TetraScratch>").pointerTypes("btSoftBodyTetraSratchArray"))
            .put(new Info("btAlignedObjectArray<btSparseSdf<3>::Cell*>").pointerTypes("btSparseSdf3CellArray"))
            .put(new Info("btDeformableBodySolver::TVStack").pointerTypes("btVector3Array"))
            .put(new Info("btSoftBody::Anchor").pointerTypes("btSoftBody.Anchor"))
            .put(new Info("btSoftBody::Cluster").pointerTypes("btSoftBody.Cluster"))
            .put(new Info("btSoftBody::DeformableFaceNodeContact").pointerTypes("btSoftBody.DeformableFaceNodeContact"))
            .put(new Info("btSoftBody::DeformableFaceRigidContact").pointerTypes("btSoftBody.DeformableFaceRigidContact"))
            .put(new Info("btSoftBody::DeformableNodeRigidAnchor").pointerTypes("btSoftBody.DeformableNodeRigidAnchor"))
            .put(new Info("btSoftBody::DeformableNodeRigidContact").pointerTypes("btSoftBody.DeformableNodeRigidContact"))
            .put(new Info("btSoftBody::Face").pointerTypes("btSoftBody.Face"))
            .put(new Info("btSoftBody::Joint").pointerTypes("btSoftBody.Joint"))
            .put(new Info("btSoftBody::Link").pointerTypes("btSoftBody.Link"))
            .put(new Info("btSoftBody::Material").pointerTypes("btSoftBody.Material"))
            .put(new Info("btSoftBody::Node").pointerTypes("btSoftBody.Node"))
            .put(new Info("btSoftBody::Note").pointerTypes("btSoftBody.Note"))
            .put(new Info("btSoftBody::RContact").pointerTypes("btSoftBody.RContact"))
            .put(new Info("btSoftBody::RenderFace").pointerTypes("btSoftBody.RenderFace"))
            .put(new Info("btSoftBody::RenderNode").pointerTypes("btSoftBody.RenderNode"))
            .put(new Info("btSoftBody::SContact").pointerTypes("btSoftBody.SContact"))
            .put(new Info("btSoftBody::Tetra").pointerTypes("btSoftBody.Tetra"))
            .put(new Info("btSoftBody::TetraScratch").pointerTypes("btSoftBody.TetraScratch"))
            .put(new Info("btSparseSdf<3>").pointerTypes("btSparseSdf3"))
            .put(new Info("btSparseSdf<3>::Cell").pointerTypes("btSparseSdf3.Cell"))
            .put(new Info("btSparseSdf<3>::IntFrac").pointerTypes("btSparseSdf3.IntFrac"))

            .put(new Info(
                    "CommonFileIOInterface",
                    "DeformableContactConstraint::m_contact",
                    "SAFE_EPSILON",
                    "btAlignedObjectArray<LagrangeMultiplier>::findBinarySearch",
                    "btAlignedObjectArray<LagrangeMultiplier>::findLinearSearch",
                    "btAlignedObjectArray<LagrangeMultiplier>::findLinearSearch2",
                    "btAlignedObjectArray<LagrangeMultiplier>::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceNodeContactConstraint> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceNodeContactConstraint> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceNodeContactConstraint> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceNodeContactConstraint> >::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceRigidContactConstraint> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceRigidContactConstraint> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceRigidContactConstraint> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableFaceRigidContactConstraint> >::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeAnchorConstraint> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeAnchorConstraint> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeAnchorConstraint> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeAnchorConstraint> >::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeRigidContactConstraint> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeRigidContactConstraint> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeRigidContactConstraint> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableNodeRigidContactConstraint> >::remove",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableStaticConstraint> >::findBinarySearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableStaticConstraint> >::findLinearSearch",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableStaticConstraint> >::findLinearSearch2",
                    "btAlignedObjectArray<btAlignedObjectArray<btDeformableStaticConstraint> >::remove",
                    "btAlignedObjectArray<btDeformableFaceNodeContactConstraint>::findBinarySearch",
                    "btAlignedObjectArray<btDeformableFaceNodeContactConstraint>::findLinearSearch",
                    "btAlignedObjectArray<btDeformableFaceNodeContactConstraint>::findLinearSearch2",
                    "btAlignedObjectArray<btDeformableFaceNodeContactConstraint>::remove",
                    "btAlignedObjectArray<btDeformableFaceRigidContactConstraint>::findBinarySearch",
                    "btAlignedObjectArray<btDeformableFaceRigidContactConstraint>::findLinearSearch",
                    "btAlignedObjectArray<btDeformableFaceRigidContactConstraint>::findLinearSearch2",
                    "btAlignedObjectArray<btDeformableFaceRigidContactConstraint>::remove",
                    "btAlignedObjectArray<btDeformableNodeAnchorConstraint>::findBinarySearch",
                    "btAlignedObjectArray<btDeformableNodeAnchorConstraint>::findLinearSearch",
                    "btAlignedObjectArray<btDeformableNodeAnchorConstraint>::findLinearSearch2",
                    "btAlignedObjectArray<btDeformableNodeAnchorConstraint>::remove",
                    "btAlignedObjectArray<btDeformableNodeRigidContactConstraint>::findBinarySearch",
                    "btAlignedObjectArray<btDeformableNodeRigidContactConstraint>::findLinearSearch",
                    "btAlignedObjectArray<btDeformableNodeRigidContactConstraint>::findLinearSearch2",
                    "btAlignedObjectArray<btDeformableNodeRigidContactConstraint>::remove",
                    "btAlignedObjectArray<btDeformableStaticConstraint>::findBinarySearch",
                    "btAlignedObjectArray<btDeformableStaticConstraint>::findLinearSearch",
                    "btAlignedObjectArray<btDeformableStaticConstraint>::findLinearSearch2",
                    "btAlignedObjectArray<btDeformableStaticConstraint>::remove",
                    "btAlignedObjectArray<btSoftBody::Anchor>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::Anchor>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::Anchor>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::Anchor>::remove",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceNodeContact>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceNodeContact>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceNodeContact>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceNodeContact>::remove",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceRigidContact>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceRigidContact>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceRigidContact>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::DeformableFaceRigidContact>::remove",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidAnchor>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidAnchor>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidAnchor>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidAnchor>::remove",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidContact>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidContact>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidContact>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::DeformableNodeRigidContact>::remove",
                    "btAlignedObjectArray<btSoftBody::Face>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::Face>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::Face>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::Face>::remove",
                    "btAlignedObjectArray<btSoftBody::Link>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::Link>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::Link>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::Link>::remove",
                    "btAlignedObjectArray<btSoftBody::Node>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::Node>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::Node>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::Node>::remove",
                    "btAlignedObjectArray<btSoftBody::Note>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::Note>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::Note>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::Note>::remove",
                    "btAlignedObjectArray<btSoftBody::RContact>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::RContact>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::RContact>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::RContact>::remove",
                    "btAlignedObjectArray<btSoftBody::RenderFace>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::RenderFace>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::RenderFace>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::RenderFace>::remove",
                    "btAlignedObjectArray<btSoftBody::RenderNode>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::RenderNode>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::RenderNode>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::RenderNode>::remove",
                    "btAlignedObjectArray<btSoftBody::SContact>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::SContact>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::SContact>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::SContact>::remove",
                    "btAlignedObjectArray<btSoftBody::Tetra>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::Tetra>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::Tetra>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::Tetra>::remove",
                    "btAlignedObjectArray<btSoftBody::TetraScratch>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::TetraScratch>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::TetraScratch>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::TetraScratch>::remove",
                    "btDeformableBackwardEulerObjective::computeStep",
                    "btDeformableMultiBodyDynamicsWorld::solveMultiBodyConstraints",
                    "btDeformableMultiBodyDynamicsWorld::rayTestSingle",
                    "btSoftBody::AJoint::Type",
                    "btSoftBody::CJoint::Type",
                    "btSoftBody::LJoint::Type",
                    "btSoftBody::m_collisionDisabledObjects",
                    "btSoftBody::m_renderNodesParents"
                ).skip())
            ;
    }
}
