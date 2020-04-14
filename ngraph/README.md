JavaCPP Presets for nGraph
==========================

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
    <version>1.5.3</version>
    <properties>
        <exec.mainClass>ABC</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>ngraph-platform</artifactId>
            <version>0.26.0-1.5.3</version>
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
