JavaCPP Presets for videoInput
==============================

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

 * https://github.com/ofTheo/videoInput/blob/update2013/videoInputSrcAndDemos/libs/videoInput/videoInput.h

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/ExampleUsage.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.videoinput</groupId>
    <artifactId>exampleusage</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>ExampleUsage</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>videoinput</artifactId>
            <version>0.200-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/ExampleUsage.java` source file
```java
import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.videoInputLib.*;

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
