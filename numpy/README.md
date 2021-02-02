JavaCPP Presets for NumPy
=========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/numpy/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/numpy) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/numpy.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![numpy](https://github.com/bytedeco/javacpp-presets/workflows/numpy/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Anumpy)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * NumPy 1.20.0  http://www.numpy.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/numpy/apidocs/

&lowast; Call `Py_AddPath(cachePackages())` before calling `Py_Initialize()`.


Sample Usage
------------
Here is a simple example of NumPy based on the information available here:

 * https://docs.scipy.org/doc/numpy/reference/c-api.html

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `MatMul.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.numpy</groupId>
    <artifactId>matmul</artifactId>
    <version>1.5.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>MatMul</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>numpy-platform</artifactId>
            <version>1.20.0-1.5.5-SNAPSHOT</version>
        </dependency>

        <!-- Additional dependencies to use bundled full version of MKL -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>mkl-platform-redist</artifactId>
            <version>2021.1-1.5.5-SNAPSHOT</version>
        </dependency>

    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `MatMul.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

public class MatMul {
    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        Py_AddPath(org.bytedeco.numpy.global.numpy.cachePackages());
        Py_Initialize();
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        long[] dimsx = {2, 2};
        DoublePointer datax = new DoublePointer(1, 2, 3, 4);
        PyObject x = PyArray_New(PyArray_Type(), dimsx.length, new SizeTPointer(dimsx),
                                 NPY_DOUBLE, null, datax, 0, NPY_ARRAY_CARRAY, null);
        PyDict_SetItemString(globals, "x", x);
        System.out.println("x = " + DoubleIndexer.create(datax, dimsx));

        PyRun_StringFlags("import numpy; y = numpy.matmul(x, x)", Py_single_input, globals, globals, null);

        PyArrayObject y = new PyArrayObject(PyDict_GetItemString(globals, "y"));
        DoublePointer datay = new DoublePointer(PyArray_BYTES(y)).capacity(PyArray_Size(y));
        long[] dimsy = new long[PyArray_NDIM(y)];
        PyArray_DIMS(y).get(dimsy);
        System.out.println("y = " + DoubleIndexer.create(datay, dimsy));
    }
}
```
