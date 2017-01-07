JavaCPP Presets for libfreenect2
================================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libfreenect2 0.2.0  https://github.com/OpenKinect/libfreenect2

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libfreenect2/apidocs/


Example
-------

Here is the full code of the example found in the [`example/`](example/) folder.

### The `pom.xml` file

```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.libfreenect</groupId>
    <artifactId>freenect2Example</artifactId>
    <version>0.2.0</version>
    <properties>
        <exec.mainClass>freenect2Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>libfreenect2</artifactId>
            <version>0.2.0-1.3.2-SNAPSHOT</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/freenect2Example.java` file

```java
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.freenect2;
import org.bytedeco.javacpp.freenect2.CpuPacketPipeline;
import org.bytedeco.javacpp.freenect2.FrameMap;
import org.bytedeco.javacpp.freenect2.Freenect2;
import org.bytedeco.javacpp.freenect2.Freenect2Device;
import org.bytedeco.javacpp.freenect2.PacketPipeline;
import org.bytedeco.javacpp.freenect2.SyncMultiFrameListener;

public class freenect2Example {
    public static void main(String[] args) {
        Freenect2 freenect2Context;
        try {
            Loader.load(org.bytedeco.javacpp.freenect2.class);
            freenect2Context = new Freenect2();
        } catch (Exception e) {
            System.out.println("Exception in the TryLoad !" + e);
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
        types |= freenect2.Frame.Color;
        types |= freenect2.Frame.Ir | freenect2.Frame.Depth;

        SyncMultiFrameListener listener = new freenect2.SyncMultiFrameListener(types);

        device.setColorFrameListener(listener);
        device.setIrAndDepthFrameListener(listener);

        device.start();

        System.out.println("Serial: " + device.getSerialNumber().getString());
        System.out.println("Firmware: " + device.getFirmwareVersion().getString());

        FrameMap frames = new FrameMap();

        int frameCount = 0;
        for (int i = 0; i < 100; i++) {
            System.out.println("getting frame " + frameCount);
            if (!listener.waitForNewFrame(frames, 10 * 1000)) // 10 sconds
            {
                System.out.println("timeout!");
                return;
            }

            freenect2.Frame rgb = frames.get(freenect2.Frame.Color);
            freenect2.Frame ir = frames.get(freenect2.Frame.Ir);
            freenect2.Frame depth = frames.get(freenect2.Frame.Depth);

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
