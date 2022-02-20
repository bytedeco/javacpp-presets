JavaCPP Presets for CMINPACK
============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/cminpack/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/cminpack) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/cminpack.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![cminpack](https://github.com/bytedeco/javacpp-presets/workflows/cminpack/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Acminpack)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * CMINPACK 1.3.8  http://devernay.free.fr/hacks/cminpack/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/cminpack/apidocs/


Sample Usage
------------
Here is a simple example of CMINPACK ported to Java from this C source file:

 * https://github.com/devernay/cminpack/blob/master/examples/tlmdifc.c

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Tlmdif1c.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.cminpack</groupId>
    <artifactId>tlmdif1c</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>Tlmdif1c</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cminpack-platform</artifactId>
            <version>1.3.8-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Tlmdif1c.java` source file
```java
/*     driver for lmdif1 example. */

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;

import static java.lang.Math.*;
import static org.bytedeco.cminpack.global.cminpack.*;

public class Tlmdif1c {
  public static void main(String[] args)
  {
    Loader.load(org.bytedeco.cminpack.global.cminpack.class);

    int info, lwa, iwa[] = new int[3];
    double tol, fnorm, x[] = new double[3], fvec[] = new double[15], wa[] = new double[75];
    int m = 15;
    int n = 3;
    /* auxiliary data (e.g. measurements) */
    double[] y = {1.4e-1, 1.8e-1, 2.2e-1, 2.5e-1, 2.9e-1, 3.2e-1, 3.5e-1,
                    3.9e-1, 3.7e-1, 5.8e-1, 7.3e-1, 9.6e-1, 1.34, 2.1, 4.39};
    /* the following struct defines the data points */
    DoublePointer data = new DoublePointer(y);

    /* the following starting values provide a rough fit. */

    x[0] = 1.;
    x[1] = 1.;
    x[2] = 1.;

    lwa = 75;

    /* set tol to the square root of the machine precision.  unless high
       precision solutions are required, this is the recommended
       setting. */

    tol = sqrt(dpmpar(1));

    info = lmdif1(new Fcn(), data, m, n, x, fvec, tol, iwa, wa, lwa);

    fnorm = enorm(m, fvec);

    System.out.printf("      final l2 norm of the residuals%15.7g\n\n",(double)fnorm);
    System.out.printf("      exit parameter                %10d\n\n", info);
    System.out.printf("      final approximate solution\n\n %15.7g%15.7g%15.7g\n",
          (double)x[0], (double)x[1], (double)x[2]);
    System.exit(0);
  }

  public static class Fcn extends cminpack_func_mn {
    @Override public int call(Pointer p, int m, int n, DoublePointer x, DoublePointer fvec, int iflag)
    {
      /* function fcn for lmdif1 example */

      int i;
      double tmp1,tmp2,tmp3;
      DoublePointer y = new DoublePointer(p);
      assert m == 15 && n == 3;

      DoubleIndexer xIdx = DoubleIndexer.create(x.capacity(n));
      DoubleIndexer yIdx = DoubleIndexer.create(y.capacity(m));
      DoubleIndexer fvecIdx = DoubleIndexer.create(fvec.capacity(m));
      for (i = 0; i < 15; ++i)
        {
          tmp1 = i + 1;
          tmp2 = 15 - i;
          tmp3 = (i > 7) ? tmp2 : tmp1;
          fvecIdx.put(i, y.get(i) - (x.get(0) + tmp1/(x.get(1)*tmp2 + x.get(2)*tmp3)));
        }
      return 0;
    }
  }
}
```
