JavaCPP Presets for LibRaw
==========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libraw/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libraw) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/libraw.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![libraw](https://github.com/bytedeco/javacpp-presets/workflows/libraw/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Alibraw)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * LibRaw https://www.libraw.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.

Documentation
-------------
Java API documentation is available here:

* http://bytedeco.org/javacpp-presets/libraw/apidocs/

Sample Usage
------------
Here is an example of implementing `dcraw` functionality using LibRaw ported to Java from the `dcraw_emu.cpp` C++ source file:

```java
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.libraw.LibRaw;
import org.bytedeco.libraw.libraw_output_params_t;

import static org.bytedeco.libraw.global.LibRaw.*;

public class LibRawDemo {
    public static void main(String[] args) {
        
        BytePointer version = LibRaw.version();
        System.out.println("LibRaw.version(): " + version.getString());

        LibRaw rawProcessor = new LibRaw();
        libraw_output_params_t params = rawProcessor.imgdata().params();

        String srcFile = "my_sample_image.dng";
        System.out.println("Reading: " + srcFile);
        int ret = rawProcessor.open_file(srcFile);
        if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
            BytePointer msg = libraw_strerror(ret);
            System.out.println("Cannot unpack " + srcFile + " : " + msg.getString());
            System.exit(ret);
        }

        System.out.println("Unpacking: " + srcFile);
        ret = rawProcessor.unpack();
        if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
            BytePointer msg = libraw_strerror(ret);
            System.out.println("Cannot unpack " + srcFile + " : " + msg.getString());
            System.exit(ret);
        }

        System.out.println("Processing");
        ret = rawProcessor.dcraw_process();
        if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
            BytePointer msg = libraw_strerror(ret);
            System.out.println("Cannot process : " + msg.getString());
            System.exit(ret);
        }

        String dstFile = "my_sample_image.ppm";
        System.out.println("Writing file: " + dstFile);
        ret = rawProcessor.dcraw_ppm_tiff_writer(dstFile);
        if (ret != LibRaw_errors.LIBRAW_SUCCESS.value) {
            BytePointer msg = libraw_strerror(ret);
            System.out.println("Cannot write file : " + msg.getString());
            System.exit(ret);
        }

        System.out.println("Cleaning up");
        rawProcessor.recycle();

        System.out.println("Done");
        System.exit(LibRaw_errors.LIBRAW_SUCCESS.value);
    }
}
```