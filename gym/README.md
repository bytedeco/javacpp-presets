JavaCPP Presets for Gym
=======================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Gym 0.17.3  https://gym.openai.com/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/gym/apidocs/

&lowast; Call `Py_AddPath(cachePackages())` before calling `Py_Initialize()`.


Sample Usage
------------
Here is a simple example of Gym based on the Python script available here:

 * https://github.com/openai/gym/blob/master/examples/scripts/list_envs

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `ListEnvs.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.gym</groupId>
    <artifactId>listenvs</artifactId>
    <version>1.5.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>ListEnvs</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>gym-platform</artifactId>
            <version>0.17.3-1.5.5-SNAPSHOT</version>
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

### The `ListEnvs.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

public class ListEnvs {
    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        Py_AddPath(org.bytedeco.gym.presets.gym.cachePackages());
        Py_Initialize();
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        PyRun_StringFlags("from gym import envs\n"
                + "envids = [spec.id for spec in envs.registry.all()]\n"
                + "for envid in sorted(envids):\n"
                + "    print(envid)\n", Py_file_input, globals, globals, null);

        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }
    }
}
```
