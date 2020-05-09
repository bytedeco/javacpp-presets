JavaCPP Presets for librealsense
================================

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
    <version>1.5.3</version>
    <properties>
        <exec.mainClass>TestConnection</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>librealsense-platform</artifactId>
            <version>1.12.4-1.5.3</version>
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
