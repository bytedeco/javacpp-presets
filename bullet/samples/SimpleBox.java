import org.bytedeco.javacpp.*;
import org.bytedeco.bullet.BulletCollision.*;
import org.bytedeco.bullet.BulletDynamics.*;
import org.bytedeco.bullet.LinearMath.*;

public class SimpleBox {

    private static btDefaultCollisionConfiguration m_collisionConfiguration;
    private static btCollisionDispatcher m_dispatcher;
    private static btBroadphaseInterface m_broadphase;
    private static btConstraintSolver m_solver;
    private static btDiscreteDynamicsWorld m_dynamicsWorld;

    public static void main(String[] args)
    {
        createEmptyDynamicsWorld();

        btBoxShape groundShape = new btBoxShape(new btVector3(50, 50, 50));

        btTransform groundTransform = new btTransform();
        groundTransform.setIdentity();
        groundTransform.setOrigin(new btVector3(0, -50, 0));

        createRigidBody(0, groundTransform, groundShape);

        btBoxShape colShape = new btBoxShape(new btVector3(1, 1, 1));
        float mass = 1.0f;

        colShape.calculateLocalInertia(mass, new btVector3(0, 0, 0));

        btTransform startTransform = new btTransform();
        startTransform.setIdentity();
        startTransform.setOrigin(new btVector3(0, 3, 0));
        btRigidBody box = createRigidBody(mass, startTransform, colShape);

        for (int i = 0; i < 10; ++ i)
        {
            m_dynamicsWorld.stepSimulation(0.1f, 10, 0.01f);
            btVector3 position = box.getWorldTransform().getOrigin();
            System.out.println(position.y());
        }

        System.out.println(
            "\n" +
            "This sample simulates falling of a rigid box, followed by \n" +
            "an inelastic collision with a ground plane.\n" +
            "The numbers show height of the box at each simulation step. \n" +
            "It should start around 3.0 and end up around 1.0.\n");
    }

    private static void createEmptyDynamicsWorld()
    {
        m_collisionConfiguration = new btDefaultCollisionConfiguration();
        m_dispatcher = new btCollisionDispatcher(m_collisionConfiguration);
        m_broadphase = new btDbvtBroadphase();
        m_solver = new btSequentialImpulseConstraintSolver();
        m_dynamicsWorld = new btDiscreteDynamicsWorld(
            m_dispatcher, m_broadphase, m_solver, m_collisionConfiguration);
        m_dynamicsWorld.setGravity(new btVector3(0, -10, 0));
    }

    private static btRigidBody createRigidBody(
        float mass,
        btTransform startTransform,
        btCollisionShape shape)
    {
        boolean isDynamic = (mass != 0.f);

        btVector3 localInertia = new btVector3(0, 0, 0);
        if (isDynamic)
            shape.calculateLocalInertia(mass, localInertia);

        btDefaultMotionState motionState = new btDefaultMotionState(
            startTransform, btTransform.getIdentity());

        btRigidBody.btRigidBodyConstructionInfo cInfo =
            new btRigidBody.btRigidBodyConstructionInfo(
                mass, motionState, shape, localInertia);

        btRigidBody body = new btRigidBody(cInfo);

        body.setUserIndex(-1);
        m_dynamicsWorld.addRigidBody(body);

        return body;
    }
}
