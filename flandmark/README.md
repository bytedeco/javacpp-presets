JavaCPP Presets for flandmark
=============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * flandmark 1.07  http://cmp.felk.cvut.cz/~uricamic/flandmark/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/flandmark/apidocs/


Sample Usage
------------
Here is a simple example of flandmark ported to Java from this C++ source file and for this data:

 * https://github.com/uricamic/flandmark/blob/master/examples/simple_example.cpp
 * https://github.com/uricamic/flandmark/tree/master/data

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/SimpleExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="<path_to_input_image> <face_bbox - 4int> [<path_to_output_image>]"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.flandmark</groupId>
    <artifactId>simpleexample</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>SimpleExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>flandmark</artifactId>
            <version>1.07-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/SimpleExample.java` source file
```java
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Michal Uricar
 * Copyright (C) 2012 Michal Uricar
 */

import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.flandmark.*;
import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class SimpleExample {
    public static void main(String[] args) {
        String flandmark_window = "flandmark_simple_example";
        long ms;
        int[] bbox = new int[4];

        if (args.length < 5) {
            System.err.println("Usage: flandmark_1 <path_to_input_image> <face_bbox - 4int> [<path_to_output_image>]");
            System.exit(1);
        }

        //cvNamedWindow(flandmark_window, 0);

        ms = System.currentTimeMillis();
        FLANDMARK_Model model = flandmark_init("flandmark_model.dat");
        if (model == null) {
            System.out.println("Structure model wasn't created. Corrupted file flandmark_model.dat?");
            System.exit(1);
        }
        ms = System.currentTimeMillis() - ms;
        System.out.println("Structure model loaded in " + ms + " ms.");


        // input image
        IplImage img = cvLoadImage(args[0]);
        if (img == null) {
            //System.err.println("Wrong path to image. Exiting...");
            System.err.println("Cannot open image " + args[0] + ". Exiting...");
            System.exit(1);
        }

        // convert image to grayscale
        IplImage img_grayscale = cvCreateImage(cvSize(img.width(), img.height()), IPL_DEPTH_8U, 1);
        cvCvtColor(img, img_grayscale, CV_BGR2GRAY);

        // face bbox
        bbox[0] = Integer.parseInt(args[1]);
        bbox[1] = Integer.parseInt(args[2]);
        bbox[2] = Integer.parseInt(args[3]);
        bbox[3] = Integer.parseInt(args[4]);


        // call flandmark_detect
        ms = System.currentTimeMillis();
        double[] landmarks = new double[2 * model.data().options().M()];
        if (flandmark_detect(img_grayscale, bbox, model, landmarks) != 0) {
            System.out.println("Error during detection.");
        }
        ms = System.currentTimeMillis() - ms;
        System.out.println("Landmarks detected in " + ms + " ms.");

        double[] bb = new double[4];
        model.bb().get(bb);
        cvRectangle(img, cvPoint(bbox[0], bbox[1]), cvPoint(bbox[2], bbox[3]), CV_RGB(255, 0, 0));
        cvRectangle(img, cvPoint((int)bb[0], (int)bb[1]), cvPoint((int)bb[2], (int)bb[3]), CV_RGB(0, 0, 255));
        cvCircle(img, cvPoint((int)landmarks[0], (int)landmarks[1]), 3, CV_RGB(0, 0, 255), CV_FILLED, 8, 0);
        for (int i = 2; i < landmarks.length; i += 2) {
            cvCircle(img, cvPoint((int)landmarks[i], (int)landmarks[i+1]), 3, CV_RGB(255, 0, 0), CV_FILLED, 8, 0);
        }
        System.out.print("detection = \t[");
        for (int ii = 0; ii < landmarks.length; ii += 2) {
            System.out.printf("%.2f ", landmarks[ii]);
        }
        System.out.println("]");
        System.out.print("\t\t[");
        for (int ii = 1; ii < landmarks.length; ii += 2) {
            System.out.printf("%.2f ", landmarks[ii]);
        }
        System.out.println("]");

        cvShowImage(flandmark_window, img);
        cvWaitKey(0);


        if (args.length > 5) {
            System.out.println("Saving image to file " + args[5] + "...");
            cvSaveImage(args[5], img);
        }

        // cleanup
        cvDestroyWindow(flandmark_window);
        cvReleaseImage(img);
        cvReleaseImage(img_grayscale);
        flandmark_free(model);
    }
}
```
