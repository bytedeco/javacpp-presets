JavaCPP Presets for nGraph
==========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/ngraph/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/ngraph) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/ngraph.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![ngraph](https://github.com/bytedeco/javacpp-presets/workflows/ngraph/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Angraph)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * nGraph 0.26.0  https://ai.intel.com/intel-ngraph/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/ngraph/apidocs/


Sample Usage
------------
Here is a simple example of nGraph ported to Java from the C++ source file at:

 * https://ngraph.nervanasys.com/index.html/core/constructing-graphs/execute.html

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `ABC.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.ngraph</groupId>
    <artifactId>abc</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>ABC</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>ngraph-platform</artifactId>
            <version>0.26.0-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `ABC.java` source file
```java
//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

import org.bytedeco.javacpp.*;

import org.bytedeco.ngraph.*;
import static org.bytedeco.ngraph.global.ngraph.*;

public class ABC {
    public static void main(String[] args) {
        // Build the graph
        Shape s = new Shape(new SizeTVector(2, 3));
        Parameter a = new Parameter(f32(), new PartialShape(s), false);
        Parameter b = new Parameter(f32(), new PartialShape(s), false);
        Parameter c = new Parameter(f32(), new PartialShape(s), false);

        Op t0 = new Add(new NodeOutput(a), new NodeOutput(b));
        Op t1 = new Multiply(new NodeOutput(t0), new NodeOutput(c));

        // Make the function
        Function f = new Function(new NodeVector(t1),
                                  new ParameterVector(a, b, c));

        // Create the backend
        Backend backend = Backend.create("CPU");

        // Allocate tensors for arguments a, b, c
        Tensor t_a = backend.create_tensor(f32(), s);
        Tensor t_b = backend.create_tensor(f32(), s);
        Tensor t_c = backend.create_tensor(f32(), s);
        // Allocate tensor for the result
        Tensor t_result = backend.create_tensor(f32(), s);

        // Initialize tensors
        float[] v_a = {1, 2, 3, 4, 5, 6};
        float[] v_b = {7, 8, 9, 10, 11, 12};
        float[] v_c = {1, 0, -1, -1, 1, 2};

        t_a.write(new FloatPointer(v_a), v_a.length * 4);
        t_b.write(new FloatPointer(v_b), v_b.length * 4);
        t_c.write(new FloatPointer(v_c), v_c.length * 4);

        // Invoke the function
        Executable exec = backend.compile(f);
        exec.call(new TensorVector(t_result), new TensorVector(t_a, t_b, t_c));

        // Get the result
        float[] r = new float[2 * 3];
        FloatPointer p = new FloatPointer(r);
        t_result.read(p, r.length * 4);
        p.get(r);

        System.out.println("[");
        for (int i = 0; i < s.get(0); i++) {
            System.out.print(" [");
            for (int j = 0; j < s.get(1); j++) {
                System.out.print(r[i * (int)s.get(1) + j] + " ");
            }
            System.out.println("]");
        }
        System.out.println("]");

        System.exit(0);
    }
}
```
