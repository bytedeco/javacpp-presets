JavaCPP Presets for LiquidFun
=============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * LiquidFun  https://github.com/google/liquidfun

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/liquidfun/apidocs/


Sample Usage
------------
Here is a simple example of LiquidFun ported to Java.

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Example.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.liquidfun</groupId>
    <artifactId>example</artifactId>
    <version>1.5.3</version>
    <properties>
        <exec.mainClass>Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>liquidfun-platform</artifactId>
            <version>master-1.5.3</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Example.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.liquidfun.*;
import static org.bytedeco.liquidfun.global.liquidfun.*;

public class Example {
  public static void main(String[] args) {
    b2World w = new b2World(0.0f, -10.0f);
    b2BodyDef bd = new b2BodyDef();
    bd.type(b2_dynamicBody);
    bd.SetPosition(1.0f, 5.0f);
    b2CircleShape c = new b2CircleShape();
    c.m_radius(2.0f);
    b2FixtureDef fd = new b2FixtureDef();
    fd.shape(c).density(1.0f);
    b2Body b = w.CreateBody(bd);
    b.CreateFixture(fd);
    for (int i = 1; i <= 5; i++) {
      System.out.println(i + ": ball at " + b.GetPositionX() + "," + b.GetPositionY());
      w.Step(0.1f, 2, 8);
    }
    System.exit(0);
  }
}
```
