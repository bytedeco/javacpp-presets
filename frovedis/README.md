JavaCPP Presets for Frovedis
============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/frovedis/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/frovedis) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/frovedis.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![frovedis](https://github.com/bytedeco/javacpp-presets/workflows/frovedis/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Afrovedis)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * Frovedis 1.0.0  https://github.com/frovedis/frovedis

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/frovedis/apidocs/

&lowast; Call `frovedis_server.initialize()` instead of `FrovedisServer.initialize()`.  


Sample Usage
------------
Here is a simple example of Frovedis ported to Java from this Scala source file:

 * https://github.com/frovedis/frovedis/blob/master/src/foreign_if/spark/examples/scala/PCADemo.scala

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `PCADemo.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.frovedis</groupId>
    <artifactId>examples</artifactId>
    <version>1.5.7-SNAPSHOT</version>
    <properties>
        <exec.mainClass>PCADemo</exec.mainClass>
        <maven.compiler.source>1.8</maven.compiler.source>
        <maven.compiler.target>1.8</maven.compiler.target>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-core_2.12</artifactId>
            <version>3.1.2</version>
        </dependency>
        <dependency>
            <groupId>org.apache.spark</groupId>
            <artifactId>spark-mllib_2.12</artifactId>
            <version>3.1.2</version>
        </dependency>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>frovedis-platform</artifactId>
            <version>1.0.0-1.5.7-SNAPSHOT</version>
        </dependency>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>frovedis-platform-ve</artifactId>
            <version>1.0.0-1.5.7-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `PCADemo.java` source file
```java
import java.util.Arrays;
import java.util.List;
import scala.Tuple2;

import com.nec.frovedis.matrix.FrovedisPCAModel;
import com.nec.frovedis.matrix.RowMatrixUtils;
import org.bytedeco.frovedis.frovedis_server;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

public class PCADemo {
  public static void main(String[] args) throws Exception {
    Logger.getLogger("org").setLevel(Level.ERROR);

    // -------- configurations --------
    SparkConf conf = new SparkConf().setAppName("PCADemo").setMaster("local[2]");
    SparkContext sc = new SparkContext(conf);
    JavaSparkContext jsc = JavaSparkContext.fromSparkContext(sc);

    // initializing Frovedis server with "personalized command", if provided in command line
    frovedis_server.initialize(args.length != 0 ? args[0] : "-np 1");

    List<Vector> data = Arrays.asList(
            Vectors.dense(1.0, 0.0, 7.0, 0.0, 0.0),
            Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
            Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    );
    JavaRDD<Vector> rows = jsc.parallelize(data);
    RowMatrix mat = new RowMatrix(rows.rdd());

    // (SPARK WAY) Compute the top 2 principal components.
    System.out.println("\nComputation using Spark native APIs:");
    Matrix s_pc1 = mat.computePrincipalComponents(2);
    System.out.println("Principal Components: ");
    System.out.println(s_pc1);

    // with variance
    System.out.println("\nWith variance: ");
    Tuple2<Matrix, Vector> s_pc2var = mat.computePrincipalComponentsAndExplainedVariance(2);
    Matrix s_pc2 = s_pc2var._1;
    Vector s_var = s_pc2var._2;
    System.out.println("Principal Components: ");
    System.out.println(s_pc2);
    System.out.println("Variance: ");
    System.out.println(s_var);

    // (FROVEDIS WAY) Compute the top 2 principal components.
    System.out.println("\n\nComputation using Frovedis APIs getting called from Spark client:");
    FrovedisPCAModel res1 = RowMatrixUtils.computePrincipalComponents(mat,2); // res: Frovedis side result pointer
    Matrix f_pc1 = res1.to_spark_result()._1;
    System.out.println("Principal Components: ");
    System.out.println(f_pc1);

    // with variance
    System.out.println("\nWith variance: ");
    FrovedisPCAModel res2 = RowMatrixUtils.computePrincipalComponentsAndExplainedVariance(mat,2);
    Tuple2<Matrix, Vector> f_pc2var = res2.to_spark_result();
    Matrix f_pc2 = f_pc2var._1;
    Vector f_var = f_pc2var._2;
    System.out.println("Principal Components: ");
    System.out.println(f_pc2);
    System.out.println("Variance: ");
    System.out.println(f_var);

    frovedis_server.shut_down();
    jsc.stop();
  }
}
```
