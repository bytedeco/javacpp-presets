JavaCPP Presets for videoInput
==============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/videoinput/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/videoinput) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/videoinput.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![videoinput](https://github.com/bytedeco/javacpp-presets/workflows/videoinput/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Avideoinput)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * videoInput 0.200  https://github.com/ofTheo/videoInput

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/videoinput/apidocs/


Sample Usage
------------
Here is a simple example of videoInput ported to Java from the "Example Usage" in this C++ source file:

 * https://github.com/ofTheo/videoInput/blob/master/videoInputSrcAndDemos/libs/videoInput/videoInput.h

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `ExampleUsage.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.videoinput</groupId>
    <artifactId>exampleusage</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>ExampleUsage</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>videoinput-platform</artifactId>
            <version>0.200-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `ExampleUsage.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.videoinput.*;
import static org.bytedeco.videoinput.global.videoInputLib.*;

public class ExampleUsage {
    public static void main(String[] args) {
        //create a videoInput object
        videoInput VI = new videoInput();

        //Prints out a list of available devices and returns num of devices found
        int numDevices = VI.listDevices();

        int device1 = 0;  //this could be any deviceID that shows up in listDevices
        int device2 = 1;  //this could be any deviceID that shows up in listDevices

        //if you want to capture at a different frame rate (default is 30)
        //specify it here, you are not guaranteed to get this fps though.
        //VI.setIdealFramerate(dev, 60);

        //setup the first device - there are a number of options:

        VI.setupDevice(device1);                            //setup the first device with the default settings
        //VI.setupDevice(device1, VI_COMPOSITE);            //or setup device with specific connection type
        //VI.setupDevice(device1, 320, 240);                //or setup device with specified video size
        //VI.setupDevice(device1, 320, 240, VI_COMPOSITE);  //or setup device with video size and connection type

        //VI.setFormat(device1, VI_NTSC_M);                 //if your card doesn't remember what format it should be
                                                            //call this with the appropriate format listed above
                                                            //NOTE: must be called after setupDevice!

        //optionally setup a second (or third, fourth ...) device - same options as above
        VI.setupDevice(device2);

        //As requested width and height can not always be accomodated
        //make sure to check the size once the device is setup

        int width   = VI.getWidth(device1);
        int height  = VI.getHeight(device1);
        int size    = VI.getSize(device1);

        BytePointer yourBuffer1 = new BytePointer(size);
        BytePointer yourBuffer2 = new BytePointer(size);

        //to get the data from the device first check if the data is new
        if (VI.isFrameNew(device1)){
            VI.getPixels(device1, yourBuffer1, false, false);   //fills pixels as a BGR (for openCV) unsigned char array - no flipping
            VI.getPixels(device1, yourBuffer2, true, true);     //fills pixels as a RGB (for openGL) unsigned char array - flipping!
        }

        //same applies to device2 etc

        //to get a settings dialog for the device
        VI.showSettingsWindow(device1);


        //Shut down devices properly
        VI.stopDevice(device1);
        VI.stopDevice(device2);
    }
}
```
