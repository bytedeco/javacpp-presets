JavaCPP Presets for SciPy
=========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/scipy/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/scipy) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/scipy.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![scipy](https://github.com/bytedeco/javacpp-presets/workflows/scipy/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Ascipy)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * SciPy 1.8.0  https://www.scipy.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/scipy/apidocs/

&lowast; Call `Py_Initialize(cachePackages())` instead of just `Py_Initialize()`.


Sample Usage
------------
Here is a simple example of SciPy based on the information available here:

 * https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SparseLinalg.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.scipy</groupId>
    <artifactId>sparselinalg</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>SparseLinalg</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>scipy-platform</artifactId>
            <version>1.8.0-1.5.7</version>
        </dependency>

        <!-- Additional dependencies to use bundled full version of MKL -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>mkl-platform-redist</artifactId>
            <version>2022.0-1.5.7</version>
        </dependency>

    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SparseLinalg.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

public class SparseLinalg {
    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        Py_Initialize(org.bytedeco.scipy.presets.scipy.cachePackages());
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        PyRun_StringFlags("import numpy as np\n"
                + "from scipy.linalg import eig, eigh\n"
                + "from scipy.sparse.linalg import eigs, eigsh\n"
                + "np.set_printoptions(suppress=True)\n"

                + "np.random.seed(0)\n"
                + "X = np.random.random((100,100)) - 0.5\n"
                + "X = np.dot(X, X.T) #create a symmetric matrix\n"

                + "evals_all, evecs_all = eigh(X)\n"
                + "evals_large, evecs_large = eigsh(X, 3, which='LM')\n"
                + "print(evals_all[-3:])\n"
                + "print(evals_large)\n"
                + "print(np.dot(evecs_large.T, evecs_all[:,-3:]))\n"

                + "evals_small, evecs_small = eigsh(X, 3, sigma=0, which='LM')\n"
                + "print(evals_all[:3])\n"
                + "print(evals_small)\n"
                + "print(np.dot(evecs_small.T, evecs_all[:,:3]))\n", Py_file_input, globals, globals, null);

        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }
    }
}
```
