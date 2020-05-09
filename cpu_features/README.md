JavaCPP Presets for cpu_features
================================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * cpu_features 0.4.1  https://github.com/google/cpu_features

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/cpu_features/apidocs/


Sample Usage
------------
Here is a simple example of cpu_features ported to Java from this C source file:

 * https://github.com/google/cpu_features#checking-features-at-runtime

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SimpleExample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.cpu_features</groupId>
    <artifactId>simpleexample</artifactId>
    <version>1.5.3</version>
    <properties>
        <exec.mainClass>SimpleExample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cpu_features-platform</artifactId>
            <version>0.4.1-1.5.3</version>
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
import org.bytedeco.cpu_features.*;
import static org.bytedeco.cpu_features.global.cpu_features.*;

public class SimpleExample {

    static X86Features features = GetX86Info().features();

    static void Compute() {
        if (features.aes() != 0 && features.sse4_2() != 0) {
            System.out.println("Run optimized code.");
        } else {
            System.out.println("Run standard code.");
        }
    }

    public static void main(String args[]) {
        Compute();
    }
}
```
