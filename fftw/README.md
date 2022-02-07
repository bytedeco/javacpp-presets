JavaCPP Presets for FFTW
========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/fftw/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/fftw) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/fftw.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![fftw](https://github.com/bytedeco/javacpp-presets/workflows/fftw/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Afftw)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * FFTW 3.3.10  http://www.fftw.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/fftw/apidocs/


Sample Usage
------------
Here is a simple example of FFTW ported to Java from this C source file:

 * https://github.com/undees/fftw-example/blob/master/fftw_example.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Example.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.fftw</groupId>
    <artifactId>example</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>fftw-platform</artifactId>
            <version>3.3.10-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Example.java` source file
```java
/* Start reading here */

import org.bytedeco.javacpp.*;
import static java.lang.Math.*;
import static org.bytedeco.fftw.global.fftw3.*;

public class Example {

    static final int NUM_POINTS = 64;


    /* Never mind this bit */

    static final int REAL = 0;
    static final int IMAG = 1;

    static void acquire_from_somewhere(DoublePointer signal) {
        /* Generate two sine waves of different frequencies and amplitudes. */

        double[] s = new double[(int)signal.capacity()];
        for (int i = 0; i < NUM_POINTS; i++) {
            double theta = (double)i / (double)NUM_POINTS * PI;

            s[2 * i + REAL] = 1.0 * cos(10.0 * theta) +
                              0.5 * cos(25.0 * theta);

            s[2 * i + IMAG] = 1.0 * sin(10.0 * theta) +
                              0.5 * sin(25.0 * theta);
        }
        signal.put(s);
    }

    static void do_something_with(DoublePointer result) {
        double[] r = new double[(int)result.capacity()];
        result.get(r);
        for (int i = 0; i < NUM_POINTS; i++) {
            double mag = sqrt(r[2 * i + REAL] * r[2 * i + REAL] +
                              r[2 * i + IMAG] * r[2 * i + IMAG]);

            System.out.println(mag);
        }
    }


    /* Resume reading here */

    public static void main(String args[]) {
        Loader.load(org.bytedeco.fftw.global.fftw3.class);

        DoublePointer signal = new DoublePointer(2 * NUM_POINTS);
        DoublePointer result = new DoublePointer(2 * NUM_POINTS);

        fftw_plan plan = fftw_plan_dft_1d(NUM_POINTS, signal, result,
                                          FFTW_FORWARD, (int)FFTW_ESTIMATE);

        acquire_from_somewhere(signal);
        fftw_execute(plan);
        do_something_with(result);

        fftw_destroy_plan(plan);
    }
}
```
