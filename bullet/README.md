JavaCPP Presets for Bullet Physics SDK
======================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/bullet/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/bullet) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/bullet.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![bullet](https://github.com/bytedeco/javacpp-presets/workflows/bullet/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Abullet)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Bullet Physics SDK 3.25  https://github.com/bulletphysics/bullet3

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/bullet/apidocs/


### Special Mappings
Mappings of `btAlignedObjectArray`'s instances for privitime types:

| C++                                  | Java            |
|--------------------------------------|-----------------|
| `btAlignedObjectArray<bool>`         | `btBoolArray`   |
| `btAlignedObjectArray<char>`         | `btCharArray`   |
| `btAlignedObjectArray<int>`          | `btIntArray`    |
| `btAlignedObjectArray<unsigned int>` | `btUIntArray`   |
| `btAlignedObjectArray<btScalar>`     | `btScalarArray` |

Name of a Java class, corresponding to an instance of `btAlignedObjectArray`
for a composite type, is constructed by adding `Array` suffix to the name of
the composite type, e.g. `btAlignedObjectArray<btQuaternion>` maps to
`btQuaternionArray`.


Sample Usage
------------
Here is a simple example of Bullet Physics SDK ported to Java and based on code found from:

 * https://github.com/bulletphysics/bullet3/tree/3.25/examples/

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SimpleBox.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.bullet</groupId>
    <artifactId>samples</artifactId>
    <version>1.5.9-SNAPSHOT</version>
    <properties>
        <exec.mainClass>SimpleBox</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>bullet-platform</artifactId>
            <version>3.25-1.5.9-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SimpleBox.java` source file
```java
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
```

See the [samples](samples) subdirectory for more.
