JavaCPP Presets for DepthAI
===========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/depthai/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/depthai) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/depthai.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![depthai](https://github.com/bytedeco/javacpp-presets/workflows/depthai/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Adepthai)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * DepthAI 2.24.0  https://luxonis.com/depthai

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/depthai/apidocs/


Sample Usage
------------
Here is a simple example of DepthAI ported to Java from this C++ source file:

 * https://github.com/luxonis/depthai-core/blob/v2.3.0/examples/src/camera_preview_example.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `CameraPreviewExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.depthai</groupId>
    <artifactId>examples</artifactId>
    <version>1.5.10-SNAPSHOT</version>
    <properties>
        <exec.mainClass>CameraPreviewExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>depthai-platform</artifactId>
            <version>2.24.0-1.5.10-SNAPSHOT</version>
        </dependency>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>opencv-platform</artifactId>
            <version>4.9.0-1.5.10-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `CameraPreviewExample.java` source file
```java
// Inludes common necessary includes for development using depthai library
import org.bytedeco.javacpp.*;
import org.bytedeco.depthai.*;
import org.bytedeco.depthai.Device;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_highgui.*;
import static org.bytedeco.depthai.global.depthai.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;

public class CameraPreviewExample {
    static Pipeline createCameraPipeline() {
        Pipeline p = new Pipeline();

        ColorCamera colorCam = p.createColorCamera();
        XLinkOut xlinkOut = p.createXLinkOut();
        xlinkOut.setStreamName("preview");

        colorCam.setPreviewSize(300, 300);
        colorCam.setResolution(ColorCameraProperties.SensorResolution.THE_1080_P);
        colorCam.setInterleaved(true);

        // Link plugins CAM -> XLINK
        colorCam.preview().link(xlinkOut.input());

        return p;
    }

    public static void main(String[] args) {
        Pipeline p = createCameraPipeline();

        // Start the pipeline
        Device d = new Device();

        System.out.print("Connected cameras: ");
        IntPointer cameras = d.getConnectedCameras();
        for (int i = 0; i < cameras.limit(); i++) {
            System.out.print(cameras.get(i) + " ");
        }
        System.out.println();

        // Start the pipeline
        d.startPipeline(p);

        Mat frame;
        DataOutputQueue preview = d.getOutputQueue("preview");

        while (true) {
            ImgFrame imgFrame = preview.getImgFrame();
            if (imgFrame != null) {
                System.out.printf("Frame - w: %d, h: %d\n", imgFrame.getWidth(), imgFrame.getHeight());
                frame = new Mat(imgFrame.getHeight(), imgFrame.getWidth(), CV_8UC3, imgFrame.getData());
                imshow("preview", frame);
                int key = waitKey(1);
                if (key == 'q') {
                    System.exit(0);
                }
            } else {
                System.out.println("Not ImgFrame");
            }
        }
    }
}
```
