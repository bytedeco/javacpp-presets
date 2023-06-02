JavaCPP Presets for LibRaw
==========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libraw/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libraw) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/libraw.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![libraw](https://github.com/bytedeco/javacpp-presets/workflows/libraw/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Alibraw)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * LibRaw 0.21.1  https://www.libraw.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libraw/apidocs/


Sample Usage
------------
Here is an example of implementing `dcraw` functionality using LibRaw ported to Java from this C++ source file:

 * https://github.com/LibRaw/LibRaw/blob/master/samples/dcraw_emu.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `LibRawDemo.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.libraw</groupId>
    <artifactId>librawdemo</artifactId>
    <version>1.5.9</version>
    <properties>
        <exec.mainClass>LibRawDemo</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>libraw-platform</artifactId>
            <version>0.21.1-1.5.9</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `LibRawDemo.java` source file
```java
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.libraw.LibRaw;
import org.bytedeco.libraw.libraw_output_params_t;

import static org.bytedeco.libraw.global.LibRaw.*;

public class LibRawDemo {
    public static String libRawVersion() {
        try (BytePointer version = LibRaw.version()) {
            return version.getString();
        }
    }

    public static void handleError(int err, String message) {
        if (err != LibRaw_errors.LIBRAW_SUCCESS.value) {
            final String msg;
            try (BytePointer e = libraw_strerror(err)) {
                msg = e.getString();
            }
            System.err.println(message + " : " + msg);
            System.exit(err);
        }
    }

    public static void main(String[] args) {
        System.out.println("");
        System.out.println("LibRaw.version(): " + libRawVersion());

        try (LibRaw rawProcessor = new LibRaw()) {
            // Set processing parameters
            libraw_output_params_t params = rawProcessor.imgdata().params();
            params.half_size(1); // Create half size image
            params.output_tiff(1); // Save as TIFF

            String srcFile = "my_sample_image.dng";
            System.out.println("Reading: " + srcFile);
            int ret = rawProcessor.open_file(srcFile);
            handleError(ret, "Cannot open " + srcFile);

            System.out.println("Unpacking: " + srcFile);
            ret = rawProcessor.unpack();
            handleError(ret, "Cannot unpack " + srcFile);

            System.out.println("Processing");
            ret = rawProcessor.dcraw_process();
            handleError(ret, "Cannot process" + srcFile);

            String dstFile = "my_sample_image.tif";
            System.out.println("Writing file: " + dstFile);
            ret = rawProcessor.dcraw_ppm_tiff_writer(dstFile);
            handleError(ret, "Cannot write " + dstFile);

            System.out.println("Cleaning up");
            rawProcessor.recycle();
        }

        System.out.println("Done");
        System.exit(LibRaw_errors.LIBRAW_SUCCESS.value);
    }
}
```
