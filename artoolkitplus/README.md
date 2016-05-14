JavaCPP Presets for ARToolKitPlus
=================================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * ARToolKitPlus 2.3.1  https://launchpad.net/artoolkitplus

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/artoolkitplus/apidocs/


Sample Usage
------------
Here is a simple example of ARToolKitPlus ported to Java from this C++ source file and for this data:

 * http://bazaar.launchpad.net/~rojtberg/artoolkitplus/trunk/view/head:/sample/multi/main.cpp
 * http://bazaar.launchpad.net/~rojtberg/artoolkitplus/trunk/files/head:/sample/data/

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/MultiMain.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.artoolkitplus</groupId>
    <artifactId>multimain</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>MultiMain</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>artoolkitplus</artifactId>
            <version>2.3.1-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/MultiMain.java` source file
```java
/**
 * Copyright (C) 2010  ARToolkitPlus Authors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * Authors:
 *  Daniel Wagner
 *  Pavel Rojtberg
 */

// Simple example to demonstrate multi-marker tracking with ARToolKitPlus
// This sample does not open any graphics window. It just
// loads test images and shows use to use the ARToolKitPlus API.

import java.io.*;
import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.ARToolKitPlus.*;

public class MultiMain {
    public static void main(String[] args) throws IOException {
        final int width = 320, height = 240, bpp = 1;
        int numPixels = width * height * bpp;
        int numBytesRead = 0;
        String fName = "data/markerboard_480-499.raw";
        byte[] cameraBuffer = new byte[numPixels];

        // try to load a test camera image.
        // these images files are expected to be simple 8-bit raw pixel
        // data without any header. the images are expetected to have a
        // size of 320x240.
        try {
            FileInputStream stream = new FileInputStream(fName);
            numBytesRead = stream.read(cameraBuffer);
            stream.close();
        } catch (IOException e) {
            System.out.println("Failed to open " + fName);
            System.exit(-1);
        }

        if (numBytesRead != numPixels) {
            System.out.println("Failed to read " + fName);
            System.exit(-1);
        }

        // create a tracker that does:
        //  - 6x6 sized marker images (required for binary markers)
        //  - samples at a maximum of 6x6
        //  - works with luminance (gray) images
        //  - can load a maximum of 0 non-binary pattern
        //  - can detect a maximum of 8 patterns in one image
        TrackerMultiMarker tracker = new TrackerMultiMarker(width, height, 8, 6, 6, 6, 0);

        tracker.setPixelFormat(PIXEL_FORMAT_LUM);

        // load a camera file.
        if (!tracker.init("data/PGR_M12x0.5_2.5mm.cal", "data/markerboard_480-499.cfg", 1.0f, 1000.0f)) {
            System.out.println("ERROR: init() failed");
            System.exit(-1);
        }

        tracker.getCamera().printSettings();

        // the marker in the BCH test image has a thiner border...
        tracker.setBorderWidth(0.125f);

        // set a threshold. we could also activate automatic thresholding
        tracker.setThreshold(160);

        // let's use lookup-table undistortion for high-speed
        // note: LUT only works with images up to 1024x1024
        tracker.setUndistortionMode(UNDIST_LUT);

        // switch to simple ID based markers
        // use the tool in tools/IdPatGen to generate markers
        tracker.setMarkerMode(MARKER_ID_SIMPLE);

        // do the OpenGL camera setup
        //glMatrixMode(GL_PROJECTION)
        //glLoadMatrixf(tracker.getProjectionMatrix());

        // here we go, just one call to find the camera pose
        int numDetected = tracker.calc(cameraBuffer);

        // use the result of calc() to setup the OpenGL transformation
        //glMatrixMode(GL_MODELVIEW)
        //glLoadMatrixf(tracker.getModelViewMatrix());

        System.out.println("\n" + numDetected + " good Markers found and used for pose estimation.\nPose-Matrix:");
        System.out.print("  ");
        for (int i = 0; i < 16; i++) {
            System.out.printf("%.2f  ", tracker.getModelViewMatrix().get(i));
            if (i % 4 == 3) {
                System.out.println();
                System.out.print("  ");
            }
        }

        boolean showConfig = false;

        if (showConfig) {
            final ARMultiMarkerInfoT artkpConfig = tracker.getMultiMarkerConfig();
            System.out.println(artkpConfig.marker_num() + " markers defined in multi marker cfg");

            System.out.println("marker matrices:");
            for (int multiMarkerCounter = 0; multiMarkerCounter < artkpConfig.marker_num(); multiMarkerCounter++) {
                ARMultiEachMarkerInfoT marker = artkpConfig.marker();
                System.out.println("marker " + multiMarkerCounter + ", id " + marker.position(multiMarkerCounter).patt_id() + ":");
                for (int row = 0; row < 3; row++) {
                    for (int column = 0; column < 4; column++) {
                        System.out.printf("%.2f  ", marker.position(multiMarkerCounter).trans(row, column));
                    }
                    System.out.println();
                }
            }
        }
    }
}
```
