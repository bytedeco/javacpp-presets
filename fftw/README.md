JavaCPP Presets for FFTW
========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * FFTW 3.3.7  http://www.fftw.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/fftw/apidocs/


Sample Usage
------------
Here is a simple example of FFTW ported to Java from this C source file:

 * https://github.com/undees/fftw-example/blob/master/fftw_example.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/Example.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.fftw</groupId>
    <artifactId>example</artifactId>
    <version>1.3.4-SNAPSHOT</version>
    <properties>
        <exec.mainClass>Example</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>fftw-platform</artifactId>
            <version>3.3.7-1.3.4-SNAPSHOT</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/Example.java` source file
```java
/* Start reading here */

import org.bytedeco.javacpp.*;
import static java.lang.Math.*;
import static org.bytedeco.javacpp.fftw3.*;

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
        Loader.load(fftw3.class);

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
