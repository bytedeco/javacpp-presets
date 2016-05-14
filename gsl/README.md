JavaCPP Presets for GSL
=======================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * GSL 2.1  http://www.gnu.org/software/gsl/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/gsl/apidocs/


Sample Usage
------------
Here is a simple example of GSL ported to Java from this C source file:

 * https://www.gnu.org/software/gsl/manual/html_node/An-Example-Program.html

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/Example.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.gsl</groupId>
    <artifactId>example</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>gsl</artifactId>
            <version>2.1-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/Example.java` source file
```java
import org.bytedeco.javacpp.*;
import static org.bytedeco.javacpp.gsl.*;

public class Example {
    public static void main(String[] args) {
        double x = 5.0;
        double y = gsl_sf_bessel_J0(x);
        System.out.printf("J0(%g) = %.18e\n", x, y);
    }
}
```
