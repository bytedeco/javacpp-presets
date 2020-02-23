JavaCPP Presets for ONNX Runtime
================================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * ONNX Runtime 1.1.1  https://github.com/microsoft/onnxruntime

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/onnxruntime/apidocs/


Sample Usage
------------
Here is a simple example of ONNX Runtime ported to Java from this C++ source file:

 * https://github.com/microsoft/onnxruntime/blob/v1.0.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `CXXApiSample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.onnxruntime</groupId>
    <artifactId>cxxapisample</artifactId>
    <version>1.5.3-SNAPSHOT</version>
    <properties>
        <exec.mainClass>CXXApiSample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>onnxruntime-platform</artifactId>
            <version>1.1.1-1.5.3-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `CXXApiSample.java` source file
```java
// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.onnxruntime.*;
import static org.bytedeco.onnxruntime.global.onnxruntime.*;

public class CXXApiSample {

    public static void main(String[] args) throws Exception {
      //*************************************************************************
      // initialize  enviroment...one enviroment per process
      // enviroment maintains thread pools and other state info
      Env env = new Env(ORT_LOGGING_LEVEL_WARNING, "test");

      // initialize session options if needed
      SessionOptions session_options = new SessionOptions();
      session_options.SetIntraOpNumThreads(1);

      // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
      // session (we also need to include cuda_provider_factory.h above which defines it)
      // #include "cuda_provider_factory.h"
      // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);
      OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options.asOrtSessionOptions(), 1);

      // Sets graph optimization level
      // Available levels are
      // ORT_DISABLE_ALL -> To disable all optimizations
      // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
      // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
      // ORT_ENABLE_ALL -> To Enable All possible opitmizations
      session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

      //*************************************************************************
      // create session and load model into memory
      // using squeezenet version 1.3
      // URL = https://github.com/onnx/models/tree/master/squeezenet
      String s = args.length > 0 ? args[0] : "squeezenet.onnx";
      Pointer model_path = Loader.getPlatform().startsWith("windows") ? new CharPointer(s) : new BytePointer(s);

      System.out.println("Using Onnxruntime C++ API");
      Session session = new Session(env, model_path, session_options);

      //*************************************************************************
      // print model input layer (node names, types, shape etc.)
      AllocatorWithDefaultOptions allocator = new AllocatorWithDefaultOptions();

      // print number of model input nodes
      long num_input_nodes = session.GetInputCount();
      PointerPointer input_node_names = new PointerPointer(num_input_nodes);
      LongPointer input_node_dims = null;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

      System.out.println("Number of inputs = " + num_input_nodes);

      // iterate over all input nodes
      for (long i = 0; i < num_input_nodes; i++) {
        // print input node names
        BytePointer input_name = session.GetInputName(i, allocator.asOrtAllocator());
        System.out.println("Input " + i + " : name=" + input_name.getString());
        input_node_names.put(i, input_name);

        // print input node types
        TypeInfo type_info = session.GetInputTypeInfo(i);
        TensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();

        int type = tensor_info.GetElementType();
        System.out.println("Input " + i + " : type=" + type);

        // print input shapes/dims
        input_node_dims = tensor_info.GetShape();
        System.out.println("Input " + i + " : num_dims=" + input_node_dims.capacity());
        for (long j = 0; j < input_node_dims.capacity(); j++)
          System.out.println("Input " + i + " : dim " + j + "=" + input_node_dims.get(j));
      }

      // Results should be...
      // Number of inputs = 1
      // Input 0 : name = data_0
      // Input 0 : type = 1
      // Input 0 : num_dims = 4
      // Input 0 : dim 0 = 1
      // Input 0 : dim 1 = 3
      // Input 0 : dim 2 = 224
      // Input 0 : dim 3 = 224

      //*************************************************************************
      // Similar operations to get output node information.
      // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
      // OrtSessionGetOutputTypeInfo() as shown above.

      //*************************************************************************
      // Score the model using sample data, and inspect values


      long input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                               // use OrtGetTensorShapeElementCount() to get official size!

      FloatPointer input_tensor_values = new FloatPointer(input_tensor_size);
      PointerPointer output_node_names = new PointerPointer("softmaxout_1");

      // initialize input data with values in [0.0, 1.0]
      FloatIndexer idx = FloatIndexer.create(input_tensor_values);
      for (long i = 0; i < input_tensor_size; i++)
        idx.put(i, (float)i / (input_tensor_size + 1));

      // create input tensor object from data values
      MemoryInfo memory_info = MemoryInfo.CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Value input_tensor = Value.CreateTensorFloat(memory_info.asOrtMemoryInfo(), input_tensor_values, input_tensor_size, input_node_dims, 4);
      assert input_tensor.IsTensor();

      // score model & input tensor, get back output tensor
      ValueVector output_tensor = session.Run(new RunOptions(), input_node_names, input_tensor, 1, output_node_names, 1);
      assert output_tensor.size() == 1 && output_tensor.get(0).IsTensor();

      // Get pointer to output tensor float values
      FloatPointer floatarr = output_tensor.get(0).GetTensorMutableDataFloat();
      assert Math.abs(floatarr.get(0) - 0.000045) < 1e-6;

      // score the model, and print scores for first 5 classes
      for (int i = 0; i < 5; i++)
        System.out.println("Score for class [" + i + "] =  " + floatarr.get(i));

      // Results should be as below...
      // Score for class[0] = 0.000045
      // Score for class[1] = 0.003846
      // Score for class[2] = 0.000125
      // Score for class[3] = 0.001180
      // Score for class[4] = 0.001317
      System.out.println("Done!");
      System.exit(0);
    }
}
```
