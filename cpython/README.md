JavaCPP Presets for CPython
===========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * CPython 3.6  https://www.python.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/cpython/apidocs/


Sample Usage
------------
Here is a simple example of CPython ported to Java from this C source file:

 * https://docs.python.org/3/extending/embedding.html#very-high-level-embedding

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/Simple.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.cpython</groupId>
    <artifactId>simple</artifactId>
    <version>1.4.4</version>
    <properties>
        <exec.mainClass>Simple</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>cpython-platform</artifactId>
            <version>3.6-1.4.4</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/Simple.java` source file
```java
import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.python.*;

public class Simple {
    public static void main(String[] args) {
        Pointer program = Py_DecodeLocale(Simple.class.getSimpleName(), null);
        if (program == null) {
            System.err.println("Fatal error: cannot get class name");
            System.exit(1);
        }
        Py_SetProgramName(program);  /* optional but recommended */
        Py_Initialize();
        PyRun_SimpleStringFlags("from time import time,ctime\n"
                              + "print('Today is', ctime(time()))\n", null);
        if (Py_FinalizeEx() < 0) {
            System.exit(120);
        }
        PyMem_RawFree(program);
        System.exit(0);
    }
}
```
