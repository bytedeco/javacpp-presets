JavaCPP Presets for GSL
=======================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * GSL 2.6  http://www.gnu.org/software/gsl/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/gsl/apidocs/


Sample Usage
------------
Here is a simple example of GSL ported to Java from this demo.c source file:

 * https://www.gnu.org/software/gsl/manual/html_node/Random-Number-Distribution-Examples.html

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
    <version>1.5.4</version>
    <properties>
        <exec.mainClass>Demo</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>gsl-platform</artifactId>
            <version>2.6-1.5.4</version>
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
