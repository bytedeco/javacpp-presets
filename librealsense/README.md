JavaCPP Presets for librealsense
================================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * librealsense 1.9.6  https://github.com/IntelRealSense/librealsense

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/librealsense/apidocs/


Example
-------

Here is a very simple example that check if you can load the library,
and connect to a camera.

### The `pom.xml`


``` xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.librealsense</groupId>
    <artifactId>TestConnection</artifactId>
    <version>1.3</version>
    <properties>
        <exec.mainClass>TestConnection</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>librealsense-platform</artifactId>
            <version>1.9.6-1.3</version>
        </dependency>
    </dependencies>
</project>
```


### The `src/main/java/TestConnection.java`

``` java
import org.bytedeco.javacpp.RealSense;
import org.bytedeco.javacpp.RealSense.context;
import org.bytedeco.javacpp.RealSense.device;

public class TestConnection {

    public static void main(String[] args) {
        context context = new context();
        System.out.println("Devices found: " + context.get_device_count());

        device device = context.get_device(0);
        System.out.println("Using device 0, an " + device.get_name());
    }
}

```
