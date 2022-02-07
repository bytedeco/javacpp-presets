JavaCPP Presets for librealsense2
=================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/librealsense2/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/librealsense2) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/librealsense2.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![librealsense2](https://github.com/bytedeco/javacpp-presets/workflows/librealsense2/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Alibrealsense2)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * librealsense2 2.50.0  https://github.com/IntelRealSense/librealsense

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/librealsense2/apidocs/

&lowast; Bindings are currently available only for the C API of librealsense2.


Sample Usage
------------
Here is a simple example of librealsense2 ported to Java from this C source file:

 * https://github.com/IntelRealSense/librealsense/blob/master/examples/C/distance/rs-distance.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `RsDistance.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.librealsense2</groupId>
    <artifactId>rsdistance</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>RsDistance</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>librealsense2-platform</artifactId>
            <version>2.50.0-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `RsDistance.java` source file
```java
// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

/* Include the librealsense C header files */
import org.bytedeco.javacpp.*;
import org.bytedeco.librealsense2.*;
import static org.bytedeco.librealsense2.global.realsense2.*;

public class RsDistance {

    /* Function calls to librealsense may raise errors of type rs_error*/
    static void check_error(rs2_error e) {
        if (!e.isNull()) {
            System.out.printf("rs_error was raised when calling %s(%s):\n", rs2_get_failed_function(e), rs2_get_failed_args(e));
            System.out.printf("    %s\n", rs2_get_error_message(e));
            System.exit(1);
        }
    }

    static void print_device_info(rs2_device dev) {
        rs2_error e = new rs2_error();
        System.out.printf("\nUsing device 0, an %s\n", rs2_get_device_info(dev, RS2_CAMERA_INFO_NAME, e));
        check_error(e);
        System.out.printf("    Serial number: %s\n", rs2_get_device_info(dev, RS2_CAMERA_INFO_SERIAL_NUMBER, e));
        check_error(e);
        System.out.printf("    Firmware version: %s\n\n", rs2_get_device_info(dev, RS2_CAMERA_INFO_FIRMWARE_VERSION, e));
        check_error(e);
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                     These parameters are reconfigurable                                           //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static int STREAM       = RS2_STREAM_DEPTH;  // rs2_stream is a types of data provided by RealSense device           //
    static int FORMAT       = RS2_FORMAT_Z16;    // rs2_format is identifies how binary data is encoded within a frame   //
    static int WIDTH        = 640;               // Defines the number of columns for each frame or zero for auto resolve//
    static int HEIGHT       = 0;                 // Defines the number of lines for each frame or zero for auto resolve  //
    static int FPS          = 30;                // Defines the rate of frames per second                                //
    static int STREAM_INDEX = 0;                 // Defines the stream index, used for multiple streams of the same type //
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    public static void main(String[] args) {
        rs2_error e = new rs2_error();

        // Create a context object. This object owns the handles to all connected realsense devices.
        // The returned object should be released with rs2_delete_context(...)
        rs2_context ctx = rs2_create_context(RS2_API_VERSION, e);
        check_error(e);

        /* Get a list of all the connected devices. */
        // The returned object should be released with rs2_delete_device_list(...)
        rs2_device_list device_list = rs2_query_devices(ctx, e);
        check_error(e);

        int dev_count = rs2_get_device_count(device_list, e);
        check_error(e);
        System.out.printf("There are %d connected RealSense devices.\n", dev_count);
        if (0 == dev_count)
            System.exit(1);

        // Get the first connected device
        // The returned object should be released with rs2_delete_device(...)
        rs2_device dev = rs2_create_device(device_list, 0, e);
        check_error(e);

        print_device_info(dev);

        // Create a pipeline to configure, start and stop camera streaming
        // The returned object should be released with rs2_delete_pipeline(...)
        rs2_pipeline pipeline =  rs2_create_pipeline(ctx, e);
        check_error(e);

        // Create a config instance, used to specify hardware configuration
        // The retunred object should be released with rs2_delete_config(...)
        rs2_config config = rs2_create_config(e);
        check_error(e);

        // Request a specific configuration
        rs2_config_enable_stream(config, STREAM, STREAM_INDEX, WIDTH, HEIGHT, FORMAT, FPS, e);
        check_error(e);

        // Start the pipeline streaming
        // The retunred object should be released with rs2_delete_pipeline_profile(...)
        rs2_pipeline_profile pipeline_profile = rs2_pipeline_start_with_config(pipeline, config, e);
        if (!e.isNull()) {
            System.out.printf("The connected device doesn't support depth streaming!\n");
            System.exit(1);
        }

        while (e.isNull()) {
            // This call waits until a new composite_frame is available
            // composite_frame holds a set of frames. It is used to prevent frame drops
            // The returned object should be released with rs2_release_frame(...)
            rs2_frame frames = rs2_pipeline_wait_for_frames(pipeline, RS2_DEFAULT_TIMEOUT, e);
            check_error(e);

            // Returns the number of frames embedded within the composite frame
            int num_of_frames = rs2_embedded_frames_count(frames, e);
            check_error(e);

            for (int i = 0; i < num_of_frames; ++i) {
                // The retunred object should be released with rs2_release_frame(...)
                rs2_frame frame = rs2_extract_frame(frames, i, e);
                check_error(e);

                // Check if the given frame can be extended to depth frame interface
                // Accept only depth frames and skip other frames
                if (0 == rs2_is_frame_extendable_to(frame, RS2_EXTENSION_DEPTH_FRAME, e))
                    continue;

                // Get the depth frame's dimensions
                int width = rs2_get_frame_width(frame, e);
                check_error(e);
                int height = rs2_get_frame_height(frame, e);
                check_error(e);

                // Query the distance from the camera to the object in the center of the image
                float dist_to_center = rs2_depth_frame_get_distance(frame, width / 2, height / 2, e);
                check_error(e);

                // Print the distance
                System.out.printf("The camera is facing an object %.3f meters away.\n", dist_to_center);

                rs2_release_frame(frame);
            }

            rs2_release_frame(frames);
        }

        // Stop the pipeline streaming
        rs2_pipeline_stop(pipeline, e);
        check_error(e);

        // Release resources
        rs2_delete_pipeline_profile(pipeline_profile);
        rs2_delete_config(config);
        rs2_delete_pipeline(pipeline);
        rs2_delete_device(dev);
        rs2_delete_device_list(device_list);
        rs2_delete_context(ctx);

        System.exit(0);
    }
}
```
