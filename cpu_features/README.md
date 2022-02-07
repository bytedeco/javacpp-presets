JavaCPP Presets for cpu_features
================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/cpu_features/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/cpu_features) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/cpu_features.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![cpu_features](https://github.com/bytedeco/javacpp-presets/workflows/cpu_features/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Acpu_features)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * cpu_features 0.6.0  https://github.com/google/cpu_features

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/cpu_features/apidocs/


Sample Usage
------------
Here is a simple example of cpu_features ported to Java from this C source file:

 * https://github.com/google/cpu_features#checking-features-at-runtime

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SimpleExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.cpu_features</groupId>
    <artifactId>simpleexample</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>SimpleExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cpu_features-platform</artifactId>
            <version>0.6.0-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SimpleExample.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.cpu_features.*;
import static org.bytedeco.cpu_features.global.cpu_features.*;

public class SimpleExample {

    static X86Features features = GetX86Info().features();

    static void Compute() {
        if (features.aes() != 0 && features.sse4_2() != 0) {
            System.out.println("Run optimized code.");
        } else {
            System.out.println("Run standard code.");
        }
    }

    public static void main(String args[]) {
        Compute();
    }
}
```
