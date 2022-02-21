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
            link = "BulletSoftBody"
        )
    },
    target = "org.bytedeco.bullet.BulletSoftBody",
    global = "org.bytedeco.bullet.global.BulletSoftBody"
)
public class BulletSoftBody implements InfoMapper {
    static { Loader.checkVersion("org.bytedeco", "bullet"); }

    public void map(InfoMap infoMap) {
        infoMap
            .put(new Info("btSparseSdf<3>").pointerTypes("btSparseSdf_3"))
            .put(new Info("btSparseSdf<3>::IntFrac").skip())
            .put(new Info("btSparseSdf<3>::Cell").skip())
            .put(new Info("btSparseSdf<3>::cells").skip())
            .put(new Info("btSoftBody::m_collisionDisabledObjects").skip())
            .put(new Info("btSoftBody::Cluster::m_nodes").skip())
            .put(new Info("btSoftBody::Cluster::m_leaf").skip())
            .put(new Info("btSoftBody::m_tetraScratches").skip())
            .put(new Info("btSoftBody::m_tetraScratchesTn").skip())
            .put(new Info("btSoftBody::m_deformableAnchors").skip())
            .put(new Info("btSoftBody::m_nodeRigidContacts").skip())
            .put(new Info("btSoftBody::m_faceRigidContacts").skip())
            .put(new Info("btSoftBody::m_faceNodeContacts").skip())
            .put(new Info("btSoftBody::m_renderNodesParents").skip())
            .put(new Info("btSoftBody::solveClusters").skip())
            .put(new Info("btSoftBody::m_fdbvnt").skip())
            .put(new Info("btSoftBody::updateNode").skip())
            .put(new Info("btSoftBody::Joint::eType").skip())
            .put(new Info("btSoftBody::Joint::eType::_").skip())
            .put(new Info("btSoftBody::Face::m_leaf").skip())
            .put(new Info("btSoftBody::Node::m_leaf").skip())
            .put(new Info("btSoftBody::Tetra::m_leaf").skip())
            .put(new Info("btSoftBody::AJoint::Type").skip())
            .put(new Info("btSoftBody::LJoint::Type").skip())
            .put(new Info("btSoftBody::CJoint::Type").skip())
            .put(new Info("btSoftBody::RayFromToCaster").skip())
            .put(new Info("btSoftBodyData").cppText("#define btSoftBodyData btSoftBodyFloatData"))
            .put(new Info("SAFE_EPSILON").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody*>").pointerTypes("btAlignedObjectArray_btSoftBody"))
            .put(new Info("btSoftBodySolver::SolverTypes").enumerate())
            .put(new Info(
                "btDeformableBodySolver::computeStep",
                "btDeformableBodySolver::computeDescentStep"
                ).skip())
            .put(new Info("btDeformableMultiBodyDynamicsWorld::setSolverCallback").skip())
            .put(new Info("btDeformableMultiBodyDynamicsWorld::rayTestSingle").skip())
            .put(new Info("btDeformableMultiBodyDynamicsWorld::solveMultiBodyConstraints").skip())
            .put(new Info("btCPUVertexBufferDescriptor::getBufferType").skip())
            .put(new Info("btSoftBodyVertexData").skip())
            .put(new Info("btSoftBodyTriangleData").skip())
            .put(new Info("btSoftBodyLinkData").skip())
            .put(new Info("btDeformableBackwardEulerObjective::m_lf").skip())
            .put(new Info("btDeformableBackwardEulerObjective::m_nodes").skip())
            .put(new Info("btDeformableBackwardEulerObjective::getIndices").skip())
            .put(new Info("btDeformableBackwardEulerObjective::m_preconditioner").skip())
            .put(new Info("btDeformableBackwardEulerObjective::m_massPreconditioner").skip())
            .put(new Info("btDeformableBackwardEulerObjective::m_KKTPreconditioner").skip())
            .put(new Info("btDeformableBackwardEulerObjective::m_projection").skip())
            .put(new Info("btDeformableBackwardEulerObjective::computeStep").skip())
            .put(new Info("btDeformableLagrangianForce::m_nodes").skip())
            .put(new Info("btDeformableLagrangianForce::setIndices").skip())

            // typedef btAlignedObjectArray<Face> tFaceArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Face>").pointerTypes("btAlignedObjectArray_btSoftBody_Face"))
            .put(new Info("btSoftBody::Face").pointerTypes("btSoftBody.Face"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Face>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Face>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Face>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Face>::remove").skip())

            // typedef btAlignedObjectArray<Note> tNoteArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Note>").pointerTypes("btAlignedObjectArray_btSoftBody_Note"))
            .put(new Info("btSoftBody::Note").pointerTypes("btSoftBody.Note"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Note>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Note>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Note>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Note>::remove").skip())

            // typedef btAlignedObjectArray<Node> tNodeArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Node>").pointerTypes("btAlignedObjectArray_btSoftBody_Node"))
            .put(new Info("btSoftBody::Node").pointerTypes("btSoftBody.Node"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Node>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Node>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Node>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Node>::remove").skip())

            // typedef btAlignedObjectArray<RenderNode> tRenderNodeArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderNode>").pointerTypes("btAlignedObjectArray_btSoftBody_RenderNode"))
            .put(new Info("btSoftBody::RenderNode").pointerTypes("btSoftBody.RenderNode"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderNode>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderNode>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderNode>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderNode>::remove").skip())

            // typedef btAlignedObjectArray<Cluster*> tClusterArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Cluster*>").pointerTypes("btAlignedObjectArray_btSoftBody_ClusterPointer"))
            .put(new Info("btSoftBody::Cluster").pointerTypes("btSoftBody.Cluster"))

            // typedef btAlignedObjectArray<Link> tLinkArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Link>").pointerTypes("btAlignedObjectArray_btSoftBody_Link"))
            .put(new Info("btSoftBody::Link").pointerTypes("btSoftBody.Link"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Link>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Link>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Link>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Link>::remove").skip())

            // typedef btAlignedObjectArray<RenderFace> tRenderFaceArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderFace>").pointerTypes("btAlignedObjectArray_btSoftBody_RenderFace"))
            .put(new Info("btSoftBody::RenderFace").pointerTypes("btSoftBody.RenderFace"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderFace>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderFace>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderFace>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RenderFace>::remove").skip())

            // typedef btAlignedObjectArray<Tetra> tTetraArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Tetra>").pointerTypes("btAlignedObjectArray_btSoftBody_Tetra"))
            .put(new Info("btSoftBody::Tetra").pointerTypes("btSoftBody.Tetra"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Tetra>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Tetra>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Tetra>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Tetra>::remove").skip())

            // typedef btAlignedObjectArray<Anchor> tAnchorArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Anchor>").pointerTypes("btAlignedObjectArray_btSoftBody_Anchor"))
            .put(new Info("btSoftBody::Anchor").pointerTypes("btSoftBody.Anchor"))
            .put(new Info("btAlignedObjectArray<btSoftBody::Anchor>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Anchor>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Anchor>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::Anchor>::remove").skip())

            // typedef btAlignedObjectArray<RContact> tRContactArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::RContact>").pointerTypes("btAlignedObjectArray_btSoftBody_RContact"))
            .put(new Info("btSoftBody::RContact").pointerTypes("btSoftBody.RContact"))
            .put(new Info("btAlignedObjectArray<btSoftBody::RContact>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RContact>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RContact>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::RContact>::remove").skip())

            // typedef btAlignedObjectArray<SContact> tSContactArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::SContact>").pointerTypes("btAlignedObjectArray_btSoftBody_SContact"))
            .put(new Info("btSoftBody::SContact").pointerTypes("btSoftBody.SContact"))
            .put(new Info("btAlignedObjectArray<btSoftBody::SContact>::findBinarySearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::SContact>::findLinearSearch").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::SContact>::findLinearSearch2").skip())
            .put(new Info("btAlignedObjectArray<btSoftBody::SContact>::remove").skip())

            // typedef btAlignedObjectArray<Material*> tMaterialArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Material*>").pointerTypes("btAlignedObjectArray_btSoftBody_MaterialPointer"))
            .put(new Info("btSoftBody::Material").pointerTypes("btSoftBody.Material"))

            // typedef btAlignedObjectArray<Joint*> tJointArray;
            .put(new Info("btAlignedObjectArray<btSoftBody::Joint*>").pointerTypes("btAlignedObjectArray_btSoftBody_JointPointer"))
            .put(new Info("btSoftBody::Joint").pointerTypes("btSoftBody.Joint"))

            .put(new Info("btSoftBody::eVSolver").skip())
            .put(new Info("btSoftBody::ePSolver").skip())
            .put(new Info("btSoftBody::getSolver").skip())
            .put(new Info("btSoftBody::fMaterial").skip())
            .put(new Info("btSoftBody::fCollision").skip())
            ;
    }
}
