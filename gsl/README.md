JavaCPP Presets for GSL
=======================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/gsl/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/gsl) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/gsl.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![gsl](https://github.com/bytedeco/javacpp-presets/workflows/gsl/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Agsl)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * GSL 2.7  http://www.gnu.org/software/gsl/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/gsl/apidocs/


Sample Usage
------------
Here is a simple example of GSL ported to Java from this demo.c source file:

 * https://www.gnu.org/software/gsl/doc/html/randist.html#examples

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Demo.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.gsl</groupId>
    <artifactId>demo</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>Demo</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>gsl-platform</artifactId>
            <version>2.7-1.5.7</version>
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

### The `Demo.java` source file
```java
import org.bytedeco.javacpp.*;
import org.bytedeco.gsl.*;
import static org.bytedeco.gsl.global.gsl.*;

public class Demo {
    public static void main(String[] args) {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        gsl_rng_type T;
        gsl_rng r;

        int n = 10;
        double mu = 3.0;

        /* create a generator chosen by the 
           environment variable GSL_RNG_TYPE */

        gsl_rng_env_setup();

        T = gsl_rng_default();
        r = gsl_rng_alloc(T);

        /* print n random variates chosen from 
           the poisson distribution with mean 
           parameter mu */

        for (int i = 0; i < n; i++) {
            int k = gsl_ran_poisson(r, mu);
            System.out.printf(" %d", k);
        }

        System.out.println();
        gsl_rng_free(r);
        System.exit(0);
    }
}
```
