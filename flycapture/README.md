JavaCPP Presets for FlyCapture
==============================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * FlyCapture 2.9.3.43  http://www.ptgrey.com/flycapture-sdk

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/flycapture/apidocs/


Sample Usage
------------
Here is a simple example of FlyCapture ported to Java from this C++ source file:

 * http://www.ptgrey.com/products/flycapture2/examples/FlyCapture2Test.zip

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/FlyCapture2Test.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.flycapture</groupId>
    <artifactId>flycapture2test</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>FlyCapture2Test</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>flycapture</artifactId>
            <version>2.9.3.43-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/FlyCapture2Test.java` source file
```java
//=============================================================================
// Copyright © 2008 Point Grey Research, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of Point
// Grey Research, Inc. ("Confidential Information").  You shall not
// disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with PGR.
//
// PGR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. PGR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================
//=============================================================================
// $Id: FlyCapture2Test.cpp,v 1.18 2009/09/08 22:10:50 soowei Exp $
//=============================================================================

import java.io.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.FlyCapture2.Error;
import static org.bytedeco.javacpp.FlyCapture2.*;

public class FlyCapture2Test {
    static void PrintBuildInfo() {
        FC2Version fc2Version = new FC2Version();
        Utilities.GetLibraryVersion(fc2Version);
        System.out.println("FlyCapture2 library version: "
                + fc2Version.major() + "." + fc2Version.minor() + "."
                + fc2Version.type() + "." + fc2Version.build());

        System.out.println("JavaCPP Presets version: "
                + FlyCapture2.class.getPackage().getImplementationVersion());
    }

    static void PrintCameraInfo(CameraInfo pCamInfo) {
        System.out.println(
            "\n*** CAMERA INFORMATION ***\n"
          + "Serial number - " + pCamInfo.serialNumber() + "\n"
          + "Camera model - " + pCamInfo.modelName().getString() + "\n"
          + "Camera vendor - " + pCamInfo.vendorName().getString() + "\n"
          + "Sensor - " + pCamInfo.sensorInfo().getString() + "\n"
          + "Resolution - " + pCamInfo.sensorResolution().getString() + "\n"
          + "Firmware version - " + pCamInfo.firmwareVersion().getString() + "\n"
          + "Firmware build time - " + pCamInfo.firmwareBuildTime().getString() + "\n");
    }

    static void PrintError(Error error) {
        error.PrintErrorTrace();
    }

    static int RunSingleCamera(PGRGuid guid) {
        final int k_numImages = 10;

        Error error;
        Camera cam = new Camera();

        // Connect to a camera
        error = cam.Connect(guid);
        if (error.notEquals(PGRERROR_OK)) {
            PrintError(error);
            return -1;
        }

        // Get the camera information
        CameraInfo camInfo = new CameraInfo();
        error = cam.GetCameraInfo(camInfo);
        if (error.notEquals(PGRERROR_OK)) {
            PrintError(error);
            return -1;
        }

        PrintCameraInfo(camInfo);

        // Start capturing images
        error = cam.StartCapture();
        if (error.notEquals(PGRERROR_OK)) {
            PrintError(error);
            return -1;
        }

        Image rawImage = new Image();
        for (int imageCnt = 0; imageCnt < k_numImages; imageCnt++) {
            // Retrieve an image
            error = cam.RetrieveBuffer(rawImage);
            if (error.notEquals(PGRERROR_OK)) {
                PrintError(error);
                continue;
            }

            System.out.println("Grabbed image " + imageCnt);

            // Create a converted image
            Image convertedImage = new Image();

            // Convert the raw image
            error = rawImage.Convert(PIXEL_FORMAT_MONO8, convertedImage);
            if (error.notEquals(PGRERROR_OK)) {
                PrintError(error);
                return -1;
            }

            // Create a unique filename
            String filename = camInfo.serialNumber() + "-" + imageCnt + ".pgm";

            // Save the image. If a file format is not passed in, then the file
            // extension is parsed to attempt to determine the file format.
            error = convertedImage.Save(filename);
            if (error.notEquals(PGRERROR_OK)) {
                PrintError(error);
                return -1;
            }
        }

        // Stop capturing images
        error = cam.StopCapture();
        if (error.notEquals(PGRERROR_OK)) {
            PrintError(error);
            return -1;
        }

        // Disconnect the camera
        error = cam.Disconnect();
        if (error.notEquals(PGRERROR_OK)) {
            PrintError(error);
            return -1;
        }

        return 0;
    }

    public static void main(String[] args) throws IOException {
        PrintBuildInfo();

        Error error;

        // Since this application saves images in the current folder
        // we must ensure that we have permission to write to this folder.
        // If we do not have permission, fail right away.
        File tempFile = new File("test.txt");
        try {
            new FileOutputStream(tempFile).close();
        } catch (IOException e) {
            System.out.println("Failed to create file in current folder.  "
                             + "Please check permissions.");
            System.exit(-1);
        }
        tempFile.delete();

        BusManager busMgr = new BusManager();
        int[] numCameras = new int[1];
        error = busMgr.GetNumOfCameras(numCameras);
        if (error.notEquals(PGRERROR_OK)) {
            PrintError(error);
            System.exit(-1);
        }

        System.out.println("Number of cameras detected: " + numCameras[0]);

        for (int i = 0; i < numCameras[0]; i++) {
            PGRGuid guid = new PGRGuid();
            error = busMgr.GetCameraFromIndex(i, guid);
            if (error.notEquals(PGRERROR_OK)) {
                PrintError(error);
                System.exit(-1);
            }

            RunSingleCamera(guid);
        }

        System.out.println("Done! Press Enter to exit...");
        System.in.read();
    }
}
```
