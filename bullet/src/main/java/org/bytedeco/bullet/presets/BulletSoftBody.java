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
            ;
    }
}
