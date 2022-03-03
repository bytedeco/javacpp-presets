JavaCPP Presets for TensorFlow Lite
===================================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/tensorflow-lite/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/tensorflow-lite) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/tensorflow-lite.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![tensorflow-lite](https://github.com/bytedeco/javacpp-presets/workflows/tensorflow-lite/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Atensorflow-lite)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * TensorFlow Lite 2.8.0  https://www.tensorflow.org/lite

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/tensorflow-lite/apidocs/


Sample Usage
------------
Here is a simple example of TensorFlow ported to Java from this C++ source file:

 * https://github.com/tensorflow/tensorflow/blob/v2.8.0/tensorflow/lite/examples/minimal/minimal.cc

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `Minimal.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.tensorflow-lite</groupId>
    <artifactId>examples</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>Minimal</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorflow-lite-platform</artifactId>
            <version>2.8.0-1.5.7</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `Minimal.java` source file
```java
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import org.bytedeco.javacpp.*;
import org.bytedeco.tensorflowlite.*;
import static org.bytedeco.tensorflowlite.global.tensorflowlite.*;

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>
public class Minimal {
    static void TFLITE_MINIMAL_CHECK(boolean x) {
      if (!x) {
        System.err.print("Error at ");
        Thread.dumpStack();
        System.exit(1);
      }
    }

    public static void main(String[] args) {
      if (args.length != 1) {
        System.err.println("minimal <tflite model>");
        System.exit(1);
      }
      String filename = args[0];

      // Load model
      FlatBufferModel model = FlatBufferModel.BuildFromFile(filename);
      TFLITE_MINIMAL_CHECK(model != null && !model.isNull());

      // Build the interpreter with the InterpreterBuilder.
      // Note: all Interpreters should be built with the InterpreterBuilder,
      // which allocates memory for the Interpreter and does various set up
      // tasks so that the Interpreter can read the provided model.
      BuiltinOpResolver resolver = new BuiltinOpResolver();
      InterpreterBuilder builder = new InterpreterBuilder(model, resolver);
      Interpreter interpreter = new Interpreter((Pointer)null);
      builder.apply(interpreter);
      TFLITE_MINIMAL_CHECK(interpreter != null && !interpreter.isNull());

      // Allocate tensor buffers.
      TFLITE_MINIMAL_CHECK(interpreter.AllocateTensors() == kTfLiteOk);
      System.out.println("=== Pre-invoke Interpreter State ===");
      PrintInterpreterState(interpreter);

      // Fill input buffers
      // TODO(user): Insert code to fill input tensors.
      // Note: The buffer of the input tensor with index `i` of type T can
      // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

      // Run inference
      TFLITE_MINIMAL_CHECK(interpreter.Invoke() == kTfLiteOk);
      System.out.println("\n\n=== Post-invoke Interpreter State ===");
      PrintInterpreterState(interpreter);

      // Read output buffers
      // TODO(user): Insert getting data out code.
      // Note: The buffer of the output tensor with index `i` of type T can
      // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

      System.exit(0);
    }
}
```
