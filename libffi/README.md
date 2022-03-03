JavaCPP Presets for libffi
==========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libffi/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/libffi) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/libffi.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![libffi](https://github.com/bytedeco/javacpp-presets/workflows/libffi/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Alibffi)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * libffi 3.4.2  https://sourceware.org/libffi/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/libffi/apidocs/


Sample Usage
------------
Here is a simple example of libffi ported to Java from the "Simple Example" in this file:

 * https://github.com/libffi/libffi/blob/master/doc/libffi.texi

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SimpleExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.libffi</groupId>
    <artifactId>simpleexample</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>SimpleExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>libffi-platform</artifactId>
            <version>3.4.2-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SimpleExample.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.libffi.*;
import static org.bytedeco.libffi.global.ffi.*;

public class SimpleExample {
     static Pointer puts = Loader.addressof("puts");

     public static void main(String[] a) {
       ffi_cif cif = new ffi_cif();
       PointerPointer<ffi_type> args = new PointerPointer<>(1);
       PointerPointer<PointerPointer> values = new PointerPointer<>(1);
       PointerPointer<BytePointer> s = new PointerPointer<>(1);
       LongPointer rc = new LongPointer(1);

       /* Initialize the argument info vectors */
       args.put(0, ffi_type_pointer());
       values.put(0, s);

       /* Initialize the cif */
       if (ffi_prep_cif(cif, FFI_DEFAULT_ABI(), 1,
                        ffi_type_sint(), args) == FFI_OK)
         {
           s.putString("Hello World!");
           ffi_call(cif, puts, rc, values);
           /* rc now holds the result of the call to puts */

           /* values holds a pointer to the function's arg, so to
              call puts() again all we need to do is change the
              value of s */
           s.putString("This is cool!");
           ffi_call(cif, puts, rc, values);
         }

       System.exit(0);
     }
}
```
