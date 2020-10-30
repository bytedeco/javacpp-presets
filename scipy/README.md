JavaCPP Presets for SciPy
=========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * SciPy 1.5.3  https://www.scipy.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/scipy/apidocs/

&lowast; Call `Py_AddPath(cachePackages())` before calling `Py_Initialize()`.


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
    <version>1.5.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>SparseLinalg</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>scipy-platform</artifactId>
            <version>1.5.3-1.5.5-SNAPSHOT</version>
        </dependency>

        <!-- Additional dependencies to use bundled full version of MKL -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>mkl-platform-redist</artifactId>
            <version>2020.4-1.5.5-SNAPSHOT</version>
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

        Py_AddPath(org.bytedeco.scipy.presets.scipy.cachePackages());
        Py_Initialize();
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
