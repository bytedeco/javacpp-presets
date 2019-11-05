JavaCPP Presets for ONNX Runtime
================================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * ONNX Runtime 0.5.0  https://github.com/microsoft/onnxruntime

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/onnxruntime/apidocs/

&lowast; Bindings are currently available only for the C API of ONNX Runtime.


Sample Usage
------------
Here is a simple example of ONNX ported to Java from this C source file:

 * https://github.com/microsoft/onnxruntime/blob/v0.5.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `CApiSample.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.onnxruntime</groupId>
    <artifactId>capisample</artifactId>
    <version>1.5.2</version>
    <properties>
        <exec.mainClass>CApiSample</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>onnxruntime-platform</artifactId>
            <version>0.5.0-1.5.2</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `CApiSample.java` source file
```java
// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

import java.nio.file.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.bytedeco.onnxruntime.*;
import static org.bytedeco.onnxruntime.global.onnxruntime.*;

public class CApiSample {

    //*****************************************************************************
    // helper function to check for status
    static void CheckStatus(OrtStatus status) {
        if (status != null && !status.isNull()) {
          String msg = OrtGetErrorMessage(status).getString();
          System.err.println(msg);
          OrtReleaseStatus(status);
          System.exit(1);
        }
    }

    public static void main(String[] args) throws Exception {
      //*************************************************************************
      // initialize  enviroment...one enviroment per process
      // enviroment maintains thread pools and other state info
      OrtEnv env = new OrtEnv();
      CheckStatus(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", env));

      // initialize session options if needed
      OrtSessionOptions session_options = new OrtSessionOptions();
      CheckStatus(OrtCreateSessionOptions(session_options));
      OrtSetSessionThreadPoolSize(session_options, 1);

      // Available levels are
      // 0 -> To disable all optimizations
      // 1 -> To enable basic optimizations (Such as redundant node removals)
      // 2 -> To enable all optimizations (Includes level 1 + more complex optimizations like node fusions)
      OrtSetSessionGraphOptimizationLevel(session_options, 1);

      // Optionally add more execution providers via session_options
      // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
      // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

      //*************************************************************************
      // create session and load model into memory
      // using squeezenet version 1.3
      // URL = https://github.com/onnx/models/tree/master/squeezenet
      OrtSession session = new OrtSession();
      String model_path = args.length > 0 ? args[0] : "squeezenet.onnx";

      System.out.println("Using Onnxruntime C API");
      CheckStatus(OrtCreateSession(env, model_path, session_options, session));

      //*************************************************************************
      // print model input layer (node names, types, shape etc.)
      SizeTPointer num_input_nodes = new SizeTPointer(1);
      OrtStatus status;
      OrtAllocator allocator = new OrtAllocator();
      CheckStatus(OrtCreateDefaultAllocator(allocator));

      // print number of model input nodes
      status = OrtSessionGetInputCount(session, num_input_nodes);
      PointerPointer input_node_names = new PointerPointer(num_input_nodes.get());
      LongPointer input_node_dims = null;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                           // Otherwise need vector<vector<>>

      System.out.println("Number of inputs = " + num_input_nodes.get());

      // iterate over all input nodes
      for (long i = 0; i < num_input_nodes.get(); i++) {
        // print input node names
        BytePointer input_name = new BytePointer();
        status = OrtSessionGetInputName(session, i, allocator, input_name);
        System.out.println("Input " + i + " : name=" + input_name.getString());
        input_node_names.put(i, input_name);

        // print input node types
        OrtTypeInfo typeinfo = new OrtTypeInfo();
        status = OrtSessionGetInputTypeInfo(session, i, typeinfo);
        OrtTensorTypeAndShapeInfo tensor_info = new OrtTensorTypeAndShapeInfo();
        CheckStatus(OrtCastTypeInfoToTensorInfo(typeinfo, tensor_info));
        int[] type = {0};
        CheckStatus(OrtGetTensorElementType(tensor_info, type));
        System.out.println("Input " + i + " : type=" + type[0]);

        // print input shapes/dims
        SizeTPointer num_dims = new SizeTPointer(1);
        CheckStatus(OrtGetDimensionsCount(tensor_info, num_dims));
        System.out.println("Input " + i + " : num_dims=" + num_dims.get());
        input_node_dims = new LongPointer(num_dims.get());
        OrtGetDimensions(tensor_info, input_node_dims, num_dims.get());
        for (long j = 0; j < num_dims.get(); j++)
          System.out.println("Input " + i + " : dim " + j + "=" + input_node_dims.get(j));

        OrtReleaseTypeInfo(typeinfo);
      }
      OrtReleaseAllocator(allocator);

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
      OrtAllocatorInfo allocator_info = new OrtAllocatorInfo();
      CheckStatus(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, allocator_info));
      OrtValue input_tensor = new OrtValue();
      CheckStatus(OrtCreateTensorWithDataAsOrtValue(allocator_info, input_tensor_values, input_tensor_size * Float.SIZE / 8, input_node_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, input_tensor));
      int[] is_tensor = {0};
      CheckStatus(OrtIsTensor(input_tensor, is_tensor));
      assert is_tensor[0] != 0;
      OrtReleaseAllocatorInfo(allocator_info);

      // score model & input tensor, get back output tensor
      PointerPointer<OrtValue> input_tensors = new PointerPointer<OrtValue>(1).put(input_tensor);
      PointerPointer<OrtValue> output_tensors = new PointerPointer<OrtValue>(1);
      CheckStatus(OrtRun(session, null, input_node_names, input_tensors, 1, output_node_names, 1, output_tensors));
      OrtValue output_tensor = output_tensors.get(OrtValue.class);
      CheckStatus(OrtIsTensor(output_tensor, is_tensor));
      assert is_tensor[0] != 0;

      // Get pointer to output tensor float values
      FloatPointer floatarr = new FloatPointer();
      CheckStatus(OrtGetTensorMutableData(output_tensor, floatarr));
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

      OrtReleaseValue(output_tensor);
      OrtReleaseValue(input_tensor);
      OrtReleaseSession(session);
      OrtReleaseSessionOptions(session_options);
      OrtReleaseEnv(env);
      System.out.println("Done!");
      System.exit(0);
    }
}
```
