JavaCPP Presets for libfreenect2
================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libfreenect2/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libfreenect2) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/libfreenect2.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![libfreenect2](https://github.com/bytedeco/javacpp-presets/workflows/libfreenect2/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Alibfreenect2)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libfreenect2 0.2.0  https://github.com/OpenKinect/libfreenect2

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libfreenect2/apidocs/


Sample Usage
------------
Here is the full code of the example found in the [`samples/`](samples/) folder.

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `TestConnection.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.libfreenect</groupId>
    <artifactId>freenect2Example</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>freenect2Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>libfreenect2-platform</artifactId>
            <version>0.2.0-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `freenect2Example.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.libfreenect2.*;
import static org.bytedeco.libfreenect2.global.freenect2.*;

/**
 *
 * @author Jeremy Laviole
 */
public class freenect2Example {
    public static void main(String[] args) {
        Freenect2 freenect2Context;
        try {
            Loader.load(org.bytedeco.libfreenect2.global.freenect2.class);
            // Context is shared accross cameras.
            freenect2Context = new Freenect2();
        } catch (Exception e) {
            System.out.println("Exception in the TryLoad !" + e);
            e.printStackTrace();
            return;
        }
        Freenect2Device device = null;
        PacketPipeline pipeline = null;
        String serial = "";

        // Only CPU pipeline tested.
        pipeline = new CpuPacketPipeline();
//        pipeline = new libfreenect2::OpenGLPacketPipeline();
//        pipeline = new libfreenect2::OpenCLPacketPipeline(deviceId);
//        pipeline = new libfreenect2::CudaPacketPipeline(deviceId);

        if (serial == "") {
            serial = freenect2Context.getDefaultDeviceSerialNumber().getString();
            System.out.println("Serial:" + serial);
        }

        device = freenect2Context.openDevice(serial, pipeline);
        // [listeners]
        int types = 0;
        types |= Frame.Color;
        types |= Frame.Ir | Frame.Depth;

        SyncMultiFrameListener listener = new SyncMultiFrameListener(types);

        device.setColorFrameListener(listener);
        device.setIrAndDepthFrameListener(listener);

        device.start();

        System.out.println("Serial: " + device.getSerialNumber().getString());
        System.out.println("Firmware: " + device.getFirmwareVersion().getString());
/// [start]

        FrameMap frames = new FrameMap();
        // Fetch 100 frames.
        int frameCount = 0;
        for (int i = 0; i < 100; i++) {
            System.out.println("getting frame " + frameCount);
            if (!listener.waitForNewFrame(frames, 10 * 1000)) // 10 sconds
            {
                System.out.println("timeout!");
                return;
            }

            Frame rgb = frames.get(Frame.Color);
            Frame ir = frames.get(Frame.Ir);
            Frame depth = frames.get(Frame.Depth);
/// [loop start]
            System.out.println("RGB image, w:" + rgb.width() + " " + rgb.height());
            byte[] imgData = new byte[1000];
            rgb.data().get(imgData);
            for (int pix = 0; pix < 10; pix++) {
                System.out.print(imgData[pix] + " ");
            }
            System.out.println();
            frameCount++;
            listener.release(frames);
            continue;
        }
        device.stop();
        device.close();
    }
}
```
