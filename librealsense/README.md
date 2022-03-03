JavaCPP Presets for librealsense
================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/librealsense/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/librealsense) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/librealsense.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![librealsense](https://github.com/bytedeco/javacpp-presets/workflows/librealsense/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Alibrealsense)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * librealsense 1.12.4  https://github.com/IntelRealSense/librealsense

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/librealsense/apidocs/


Sample Usage
------------
Here is a very simple example that check if you can load the library, and connect to a camera.

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `TestConnection.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.librealsense</groupId>
    <artifactId>TestConnection</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>TestConnection</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>librealsense-platform</artifactId>
            <version>1.12.4-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `TestConnection.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.librealsense.*;
import static org.bytedeco.librealsense.global.RealSense.*;

public class TestConnection {

    public static void main(String[] args) {
        context context = new context();
        System.out.println("Devices found: " + context.get_device_count());

        device device = context.get_device(0);
        System.out.println("Using device 0, an " + device.get_name());
    }
}
```
