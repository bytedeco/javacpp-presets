JavaCPP Presets for Chilitags
=============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Chilitags  http://chili.epfl.ch/software

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/chilitags/apidocs/


Sample Usage
------------
Here is the live detection sample of Chilitags ported to Java from this C++ source file:

 * https://github.com/chili-epfl/chilitags/blob/master/samples/detection/detect-live.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/DetectLive.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="[xRes] [yRes] [cameraIndex]"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.chilitags</groupId>
    <artifactId>detectlive</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>DetectLive</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>chilitags</artifactId>
            <version>master-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/DetectLive.java` source file
```java
/*******************************************************************************
*   Copyright 2013-2014 EPFL                                                   *
*   Copyright 2013-2014 Quentin Bonnard                                        *
*                                                                              *
*   This file is part of chilitags.                                            *
*                                                                              *
*   Chilitags is free software: you can redistribute it and/or modify          *
*   it under the terms of the Lesser GNU General Public License as             *
*   published by the Free Software Foundation, either version 3 of the         *
*   License, or (at your option) any later version.                            *
*                                                                              *
*   Chilitags is distributed in the hope that it will be useful,               *
*   but WITHOUT ANY WARRANTY; without even the implied warranty of             *
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
*   GNU Lesser General Public License for more details.                        *
*                                                                              *
*   You should have received a copy of the GNU Lesser General Public License   *
*   along with Chilitags.  If not, see <http://www.gnu.org/licenses/>.         *
*******************************************************************************/

// This file serves as an illustration of how to use Chilitags

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;

// The Chilitags header
import static org.bytedeco.javacpp.chilitags.*;

import static org.bytedeco.javacpp.opencv_imgproc.*; // getTickCount...

import static org.bytedeco.javacpp.opencv_core.*; // CV_AA

// OpenCV goodness for I/O
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_videoio.*;

public class DetectLive {
    public static void main(String[] args) {
        // Simple parsing of the parameters related to the image acquisition
        int xRes = 640;
        int yRes = 480;
        int cameraIndex = 0;
        if (args.length > 1) {
            xRes = Integer.parseInt(args[0]);
            yRes = Integer.parseInt(args[1]);
        }
        if (args.length > 2) {
            cameraIndex = Integer.parseInt(args[2]);
        }

        // The source of input images
        VideoCapture capture = new VideoCapture(cameraIndex);
        if (!capture.isOpened()) {
            System.err.println("Unable to initialise video capture.");
            System.exit(1);
        }
        capture.set(CAP_PROP_FRAME_WIDTH, xRes);
        capture.set(CAP_PROP_FRAME_HEIGHT, yRes);
        Mat inputImage = new Mat();

        // The tag detection happens in the Chilitags class.
        Chilitags chilitags = new Chilitags();

        // The detection is not perfect, so if a tag is not detected during one frame,
        // the tag will shortly disappears, which results in flickering.
        // To address this, Chilitags "cheats" by keeping tags for n frames
        // at the same position. When tags disappear for more than 5 frames,
        // Chilitags actually removes it.
        // Here, we cancel this to show the raw detection results.
        chilitags.setFilter(0, 0.0f);

        namedWindow("DisplayChilitags");
        // Main loop, exiting when 'q is pressed'
        while ('q' != (char)waitKey(1)) {

            // Capture a new image.
            capture.read(inputImage);

            // Start measuring the time needed for the detection
            long startTime = getTickCount();

            // Detect tags on the current image (and time the detection);
            // The resulting map associates tag ids (between 0 and 1023)
            // to four 2D points corresponding to the corners positions
            // in the picture.
            TagCornerMap tags = chilitags.find(inputImage);

            // Measure the processing time needed for the detection
            long endTime = getTickCount();
            float processingTime = 1000.0f * ((float)endTime - startTime) / (float)getTickFrequency();


            // Now we start using the result of the detection.

            // First, we set up some constants related to the information overlaid
            // on the captured image
            final Scalar COLOR = new Scalar(255, 0, 255, 0);
            // OpenCv can draw with sub-pixel precision with fixed point coordinates
            final int SHIFT = 16;
            final float PRECISION = 1 << SHIFT;

            // We dont want to draw directly on the input image, so we clone it
            Mat outputImage = inputImage.clone();

            for (TagCornerMap.Iterator tag = tags.begin(); !tag.equals(tags.end()); tag = tag.increment()) {

                int id = tag.first();
                // We wrap the corner matrix into a datastructure that allows an
                // easy access to the coordinates
                FloatIndexer corners = FloatIndexer.create(tag.second().capacity(8), new int[] { 4 }, new int[] { 2 });

                // We start by drawing the borders of the tag
                for (int i = 0; i < 4; i++) {
                    line(outputImage,
                         new Point(Math.round(PRECISION*corners.get(i, 0)), Math.round(PRECISION*corners.get(i, 1))),
                         new Point(Math.round(PRECISION*corners.get((i+1)%4, 0)), Math.round(PRECISION*corners.get((i+1)%4, 1))),
                         COLOR, 1, LINE_AA, SHIFT);
                }

                // Other points can be computed from the four corners of the Quad.
                // Chilitags are oriented. It means that the points 0,1,2,3 of
                // the Quad coordinates are consistently the top-left, top-right,
                // bottom-right and bottom-left corners.
                // (i.e. clockwise, starting from top-left)
                // Using this, we can compute (an approximation of) the center of
                // tag.
                Point center = new Point(Math.round(0.5f*(corners.get(0, 0) + corners.get(2, 0))),
                                         Math.round(0.5f*(corners.get(0, 1) + corners.get(2, 1))));
                putText(outputImage, String.format("%d", id), center,
                        FONT_HERSHEY_SIMPLEX, 0.5f, COLOR);
            }

            // Some stats on the current frame (resolution and processing time)
            putText(outputImage,
                    String.format("%dx%d %4.0f ms (press q to quit)",
                                  outputImage.cols(), outputImage.rows(),
                                  processingTime),
                    new Point(32,32),
                    FONT_HERSHEY_SIMPLEX, 0.5f, COLOR);

            // Finally...
            imshow("DisplayChilitags", outputImage);
        }

        destroyWindow("DisplayChilitags");
        capture.release();
    }
}
```
