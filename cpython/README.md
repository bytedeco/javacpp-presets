JavaCPP Presets for CPython
===========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/cpython/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/cpython) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/cpython.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![cpython](https://github.com/bytedeco/javacpp-presets/workflows/cpython/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Acpython)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


License Agreements
------------------
By downloading these archives, you agree to the [terms and conditions for accessing or otherwise using Python](https://docs.python.org/3/license.html).


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * CPython 3.10.2  https://www.python.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/cpython/apidocs/

&lowast; Call `Py_Initialize(cachePackages())` instead of just `Py_Initialize()`.  
&lowast; To satisfy OpenSSL, we might need to set the `SSL_CERT_FILE` environment variable to the full path of `cacert.pem` extracted by default under `~/.javacpp/cache/`.


Sample Usage
------------
Here is a simple example of CPython ported to Java from this C source file:

 * https://docs.python.org/3/extending/embedding.html#very-high-level-embedding

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Simple.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.cpython</groupId>
    <artifactId>simple</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>Simple</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cpython-platform</artifactId>
            <version>3.10.2-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Simple.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import static org.bytedeco.cpython.global.python.*;

public class Simple {
    public static void main(String[] args) throws Exception {
        Pointer program = Py_DecodeLocale(Simple.class.getSimpleName(), null);
        if (program == null) {
            System.err.println("Fatal error: cannot decode class name");
            System.exit(1);
        }
        Py_SetProgramName(program);  /* optional but recommended */
        Py_Initialize(cachePackages());
        PyRun_SimpleString("from time import time,ctime\n"
                         + "print('Today is', ctime(time()))\n");
        if (Py_FinalizeEx() < 0) {
            System.exit(120);
        }
        PyMem_RawFree(program);
        System.exit(0);
    }
}
```
