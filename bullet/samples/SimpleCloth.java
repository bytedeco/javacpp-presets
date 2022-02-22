import org.bytedeco.javacpp.*;
import org.bytedeco.bullet.BulletCollision.*;
import org.bytedeco.bullet.BulletDynamics.*;
import org.bytedeco.bullet.BulletSoftBody.*;
import org.bytedeco.bullet.LinearMath.*;

public class SimpleCloth {

    private static btDefaultCollisionConfiguration m_collisionConfiguration;
    private static btCollisionDispatcher m_dispatcher;
    private static btBroadphaseInterface m_broadphase;
    private static btConstraintSolver m_solver;
    private static btSoftRigidDynamicsWorld m_dynamicsWorld;
    private static btSoftBodyWorldInfo softBodyWorldInfo;

    public static void main(String[] args) {
        createEmptyDynamicsWorld();

        final float s = 4;  //size of cloth patch
        final int NUM_X = 31;  //vertices on X axis
        final int NUM_Z = 31;  //vertices on Z axis
        btSoftBody cloth = createSoftBody(s, NUM_X, NUM_Z, 1 + 2);

        for (int i = 0; i < 50; ++ i)
        {
            m_dynamicsWorld.stepSimulation(0.1f, 10, 0.01f);
            btSoftBody.Face face = cloth.m_faces().at(1799);
            btSoftBody.Node node = face.m_n(0);
            btVector3 position = node.m_x();
            System.out.println(position.y());
        }

        System.out.println(
            "\n" +
            "This sample simulates a square piece of cloth (4 units of \n" +
            "size in each dimension), with two corners fixed rigidly \n" +
            "at the height of 5 units.\n" +
            "At the beginning of the simulation, the cloth is positioned \n" +
            "horizontally and, due to the presence of the gravity force, \n" +
            "should proceed with a damped swinging motion. \n" +
            "The numbers show the height at each simulation step of one \n" +
            "of the free corners of the cloth. It should start around 5.0 \n" +
            "and end up floating around 1.0.\n");
    }

    private static void createEmptyDynamicsWorld()
    {
        m_collisionConfiguration = new btSoftBodyRigidBodyCollisionConfiguration();
        m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);

        m_broadphase = new btDbvtBroadphase();

        m_solver = new btSequentialImpulseConstraintSolver();

        m_dynamicsWorld = new btSoftRigidDynamicsWorld(
            m_dispatcher,
            m_broadphase,
            m_solver,
            m_collisionConfiguration);
        m_dynamicsWorld.setGravity(new btVector3(0, -10, 0));

        softBodyWorldInfo = new btSoftBodyWorldInfo();
        softBodyWorldInfo.m_broadphase(m_broadphase);
        softBodyWorldInfo.m_dispatcher(m_dispatcher);
        softBodyWorldInfo.m_gravity(m_dynamicsWorld.getGravity());
        softBodyWorldInfo.m_sparsesdf().Initialize();
    }

    public static btSoftBody createSoftBody(
        float s, int numX, int numY, int fixed)
    {
        btSoftBody cloth = btSoftBodyHelpers.CreatePatch(
            softBodyWorldInfo,
            new btVector3(-s / 2, s + 1, 0),
            new btVector3(+s / 2, s + 1, 0),
            new btVector3(-s / 2, s + 1, +s),
            new btVector3(+s / 2, s + 1, +s),
            numX, numY,
            fixed, true);

        cloth.getCollisionShape().setMargin(0.001f);
        cloth.getCollisionShape().setUserPointer(cloth);
        cloth.generateBendingConstraints(2, cloth.appendMaterial());
        cloth.setTotalMass(10);
        cloth.m_cfg().piterations(5);
        cloth.m_cfg().kDP(0.005f);
        m_dynamicsWorld.addSoftBody(cloth);

        return cloth;
    }
}
