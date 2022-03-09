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
import org.bytedeco.javacpp.annotation.Cast;
import org.bytedeco.javacpp.annotation.Platform;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.javacpp.presets.javacpp;
import org.bytedeco.javacpp.tools.Info;
import org.bytedeco.javacpp.tools.InfoMap;
import org.bytedeco.javacpp.tools.InfoMapper;
import org.bytedeco.javacpp.FunctionPointer;
import org.bytedeco.javacpp.Pointer;

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
                "BulletSoftBody/btSparseSDF.h",
                "BulletSoftBody/btSoftBody.h",
                "BulletSoftBody/btSoftRigidDynamicsWorld.h",
                "BulletSoftBody/btSoftBodyHelpers.h",
                "BulletSoftBody/btSoftBodyRigidBodyCollisionConfiguration.h",
                "BulletSoftBody/btSoftBodySolvers.h",
                "BulletSoftBody/btSoftMultiBodyDynamicsWorld.h",
                "BulletSoftBody/btDeformableBodySolver.h",
                "BulletSoftBody/btDeformableMultiBodyConstraintSolver.h",
                "BulletSoftBody/btDeformableMultiBodyDynamicsWorld.h",
                "BulletSoftBody/btSoftBodySolverVertexBuffer.h",
                "BulletSoftBody/btDeformableBackwardEulerObjective.h",
                "BulletSoftBody/btDeformableLagrangianForce.h",
            },
            link = "BulletSoftBody@.3.20"
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

            .put(new Info("btSoftBodySolver::SolverTypes").enumerate())

            .put(new Info("btAlignedObjectArray<btSoftBody*>").pointerTypes("btSoftBodyArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Anchor>").pointerTypes("btSoftBodyAnchorArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Cluster*>").pointerTypes("btSoftBodyClusterArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Face>").pointerTypes("btSoftBodyFaceArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Joint*>").pointerTypes("btSoftBodyJointArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Link>").pointerTypes("btSoftBodyLinkArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Material*>").pointerTypes("btSoftBodyMaterialArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Node>").pointerTypes("btSoftBodyNodeArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Note>").pointerTypes("btSoftBodyNoteArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RContact>").pointerTypes("btSoftBodyRContactArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderFace>").pointerTypes("btSoftBodyRenderFaceArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderNode>").pointerTypes("btSoftBodyRenderNodeArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::SContact>").pointerTypes("btSoftBodySContactArray"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Tetra>").pointerTypes("btSoftBodyTetraArray"))
            .put(new Info("btSoftBody::Anchor").pointerTypes("btSoftBody.Anchor"))
            .put(new Info("btSoftBody::Cluster").pointerTypes("btSoftBody.Cluster"))
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
            .put(new Info("btSparseSdf<3>").pointerTypes("btSparseSdf_3"))

            .put(new Info(
                    "SAFE_EPSILON",
                    "btAlignedObjectArray<btSoftBody::Anchor>::findBinarySearch",
                    "btAlignedObjectArray<btSoftBody::Anchor>::findLinearSearch",
                    "btAlignedObjectArray<btSoftBody::Anchor>::findLinearSearch2",
                    "btAlignedObjectArray<btSoftBody::Anchor>::remove",
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
                    "btCPUVertexBufferDescriptor::getBufferType",
                    "btDeformableBackwardEulerObjective::computeStep",
                    "btDeformableBackwardEulerObjective::getIndices",
                    "btDeformableBackwardEulerObjective::m_KKTPreconditioner",
                    "btDeformableBackwardEulerObjective::m_lf",
                    "btDeformableBackwardEulerObjective::m_massPreconditioner",
                    "btDeformableBackwardEulerObjective::m_nodes",
                    "btDeformableBackwardEulerObjective::m_preconditioner",
                    "btDeformableBackwardEulerObjective::m_projection",
                    "btDeformableBodySolver::computeDescentStep",
                    "btDeformableBodySolver::computeStep",
                    "btDeformableLagrangianForce::m_nodes",
                    "btDeformableLagrangianForce::setIndices",
                    "btDeformableMultiBodyDynamicsWorld::rayTestSingle",
                    "btDeformableMultiBodyDynamicsWorld::setSolverCallback",
                    "btDeformableMultiBodyDynamicsWorld::solveMultiBodyConstraints",
                    "btSoftBody::AJoint::Type",
                    "btSoftBody::CJoint::Type",
                    "btSoftBody::Cluster::m_leaf",
                    "btSoftBody::Cluster::m_nodes",
                    "btSoftBody::Face::m_leaf",
                    "btSoftBody::Joint::eType",
                    "btSoftBody::Joint::eType::_",
                    "btSoftBody::LJoint::Type",
                    "btSoftBody::Node::m_leaf",
                    "btSoftBody::RayFromToCaster",
                    "btSoftBody::Tetra::m_leaf",
                    "btSoftBody::ePSolver",
                    "btSoftBody::eVSolver",
                    "btSoftBody::fCollision",
                    "btSoftBody::fMaterial",
                    "btSoftBody::getSolver",
                    "btSoftBody::m_collisionDisabledObjects",
                    "btSoftBody::m_deformableAnchors",
                    "btSoftBody::m_faceNodeContacts",
                    "btSoftBody::m_faceRigidContacts",
                    "btSoftBody::m_fdbvnt",
                    "btSoftBody::m_nodeRigidContacts",
                    "btSoftBody::m_renderNodesParents",
                    "btSoftBody::m_tetraScratches",
                    "btSoftBody::m_tetraScratchesTn",
                    "btSoftBody::solveClusters",
                    "btSoftBody::tPSolverArray",
                    "btSoftBody::tVSolverArray",
                    "btSoftBody::updateNode",
                    "btSoftBodyLinkData",
                    "btSoftBodyTriangleData",
                    "btSoftBodyVertexData",
                    "btSparseSdf<3>::Cell",
                    "btSparseSdf<3>::IntFrac",
                    "btSparseSdf<3>::cells"
                ).skip())
            ;
    }
}
