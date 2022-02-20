JavaCPP Presets for LiquidFun
=============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/liquidfun/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/liquidfun) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/liquidfun.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![liquidfun](https://github.com/bytedeco/javacpp-presets/workflows/liquidfun/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Aliquidfun)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


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
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>liquidfun-platform</artifactId>
            <version>master-1.5.7</version>
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
