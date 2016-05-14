JavaCPP Presets for libdc1394
=============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libdc1394 2.2.4  http://damien.douxchamps.net/ieee1394/libdc1394/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libdc1394/apidocs/


Sample Usage
------------
Here is a simple example of libdc1394 ported to Java from this C source file:

 * http://sourceforge.net/p/libdc1394/code/ci/V_2_2_2/tree/libdc1394/examples/grab_color_image.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/GrabColorImage.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.libdc1394</groupId>
    <artifactId>grabcolorimage</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>GrabColorImage</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>libdc1394</artifactId>
            <version>2.2.4-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/GrabColorImage.java` source file
```java
/*
 * Get one image using libdc1394 and store it as portable pix map
 * (ppm). Based on 'grab_gray_image' from Olaf Ronneberge
 *
 * Written by Damien Douxchamps <ddouxchamps@users.sf.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */

import java.io.*;
import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.dc1394.*;

public class GrabColorImage {
    static final String IMAGE_FILE_NAME = "image.ppm";

    /*-----------------------------------------------------------------------
     *  Releases the cameras and exits
     *-----------------------------------------------------------------------*/
    static void cleanup_and_exit(dc1394camera_t camera) {
        dc1394_video_set_transmission(camera, DC1394_OFF);
        dc1394_capture_stop(camera);
        dc1394_camera_free(camera);
        System.exit(1);
    }

    public static void main(String[] args) throws IOException {
        OutputStream imagestream;
        dc1394camera_t camera;
        int[] width = new int[1], height = new int[1];
        dc1394video_frame_t frame = new dc1394video_frame_t(null);
        //dc1394featureset_t features;
        dc1394_t d;
        dc1394camera_list_t list = new dc1394camera_list_t();
        int err;

        d = dc1394_new();
        if (d == null) {
            System.exit(1);
        }
        err = dc1394_camera_enumerate(d, list);
        if (err != 0) {
            dc1394_log_error("Failed to enumerate cameras: " + err);
        }

        if (list.num() == 0) {
            dc1394_log_error("No cameras found");
            System.exit(1);
        }

        camera = dc1394_camera_new(d, list.ids().guid());
        if (camera == null) {
            dc1394_log_error("Failed to initialize camera with guid "
                    + Long.toHexString(list.ids().guid()));
            System.exit(1);
        }
        dc1394_camera_free_list(list);

        System.out.println("Using camera with GUID " + Long.toHexString(camera.guid()));

        /*-----------------------------------------------------------------------
         *  setup capture
         *-----------------------------------------------------------------------*/

        err = dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400);
        if (err != 0) {
            dc1394_log_error("Could not set iso speed: " + err);
            cleanup_and_exit(camera);
        }

        err = dc1394_video_set_mode(camera, DC1394_VIDEO_MODE_640x480_RGB8);
        if (err != 0) {
            dc1394_log_error("Could not set video mode: " + err);
            cleanup_and_exit(camera);
        }

        err = dc1394_video_set_framerate(camera, DC1394_FRAMERATE_7_5);
        if (err != 0) {
            dc1394_log_error("Could not set framerate: " + err);
            cleanup_and_exit(camera);
        }

        err = dc1394_capture_setup(camera,4, DC1394_CAPTURE_FLAGS_DEFAULT);
        if (err != 0) {
            dc1394_log_error("Could not setup camera-\n"
                           + "make sure that the video mode and framerate are\n"
                           + "supported by your camera: " + err);
            cleanup_and_exit(camera);
        }

        /*-----------------------------------------------------------------------
         *  have the camera start sending us data
         *-----------------------------------------------------------------------*/
        err = dc1394_video_set_transmission(camera, DC1394_ON);
        if (err != 0) {
            dc1394_log_error("Could not start camera iso transmission: " + err);
            cleanup_and_exit(camera);
        }

        /*-----------------------------------------------------------------------
         *  capture one frame
         *-----------------------------------------------------------------------*/
        err = dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, frame);
        if (err != 0) {
            dc1394_log_error("Could not capture a frame: " + err);
            cleanup_and_exit(camera);
        }

        /*-----------------------------------------------------------------------
         *  stop data transmission
         *-----------------------------------------------------------------------*/
        err = dc1394_video_set_transmission(camera,DC1394_OFF);
        if (err != 0) {
            dc1394_log_error("Could not stop the camera? " + err);
            cleanup_and_exit(camera);
        }

        /*-----------------------------------------------------------------------
         *  save image as 'Image.pgm'
         *-----------------------------------------------------------------------*/
        OutputStream stream = new FileOutputStream(IMAGE_FILE_NAME);

        dc1394_get_image_size_from_video_mode(camera,
                DC1394_VIDEO_MODE_640x480_RGB8, width, height);
        stream.write(("P6\n" + width[0] + " " + height[0] + "\n255\n").getBytes());
        byte[] bytes = new byte[height[0] * width[0] * 3];
        frame.image().get(bytes);
        stream.write(bytes);
        stream.close();
        System.out.println(
                "wrote: " + IMAGE_FILE_NAME + " (" + bytes.length + " image bytes)");

        /*-----------------------------------------------------------------------
         *  close camera
         *-----------------------------------------------------------------------*/
        dc1394_video_set_transmission(camera, DC1394_OFF);
        dc1394_capture_stop(camera);
        dc1394_camera_free(camera);
        dc1394_free(d);
    }
}
```
