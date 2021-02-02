JavaCPP Presets for ONNX
========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/onnx/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/onnx) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/onnx.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![onnx](https://github.com/bytedeco/javacpp-presets/workflows/onnx/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Aonnx)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * ONNX 1.8.1  https://onnx.ai/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/onnx/apidocs/


Sample Usage
------------
Here is a simple example of ONNX ported to Java from this C++ source file and for this data:

 * https://github.com/onnx/onnx/issues/418#issuecomment-357596638
 * https://github.com/onnx/onnx/blob/master/onnx/examples/resources/single_relu.onnx

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `LoadModel.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.onnx</groupId>
    <artifactId>loadmodel</artifactId>
    <version>1.5.5-SNAPSHOT</version>
    <properties>
        <exec.mainClass>LoadModel</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>onnx-platform</artifactId>
            <version>1.8.1-1.5.5-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `LoadModel.java` source file
```java
import java.nio.file.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.onnx.*;
import static org.bytedeco.onnx.global.onnx.*;

public class LoadModel {
    public static void main(String[] args) throws Exception {
        OpSchemaVector allSchemas = OpSchemaRegistry.get_all_schemas();
        System.out.println(allSchemas.size());

        byte[] bytes = Files.readAllBytes(Paths.get("examples/resources/single_relu.onnx"));

        ModelProto model = new ModelProto();
        ParseProtoFromBytes(model, new BytePointer(bytes), bytes.length);

        check_model(model);

        InferShapes(model);

        StringVector passes = new StringVector("eliminate_nop_transpose", "eliminate_nop_pad", "fuse_consecutive_transposes", "fuse_transpose_into_gemm");
        Optimize(model, passes);

        check_model(model);

        ConvertVersion(model, 8);

        System.out.println(model.graph().input_size());
    }
}
```
