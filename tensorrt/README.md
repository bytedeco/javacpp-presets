JavaCPP Presets for TensorRT
============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/tensorrt/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/tensorrt) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/tensorrt.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![tensorrt](https://github.com/bytedeco/javacpp-presets/workflows/tensorrt/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Atensorrt)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


License Agreements
------------------
By downloading these archives, you agree to the terms of the license agreements for NVIDIA software included in the archives.

### TensorRT
To view the license for TensorRT included in these archives, click [here](https://docs.nvidia.com/deeplearning/tensorrt/sla/)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * TensorRT 10.15.1.29  https://developer.nvidia.com/tensorrt

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/tensorrt/apidocs/


Sample Usage
------------
Here is a simple example for TensorRT inference ported to Java from the `sampleOnnxMNIST.cpp` sample file available at:

 * https://github.com/NVIDIA/TensorRT/tree/main/samples/sampleOnnxMNIST

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SampleOnnxMNIST.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.tensorrt</groupId>
    <artifactId>sampleonnxmnist</artifactId>
    <version>1.5.13</version>
    <properties>
        <exec.mainClass>SampleOnnxMNIST</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorrt-platform</artifactId>
            <version>10.15-1.5.13</version>
        </dependency>

        <!-- Additional dependencies to use bundled CUDA, cuDNN, NCCL, and TensorRT -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cuda-platform-redist</artifactId>
            <version>13.1-9.19-1.5.13</version>
        </dependency>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorrt-platform-redist</artifactId>
            <version>10.15-1.5.13</version>
        </dependency>

    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SampleOnnxMNIST.java` source file
```java
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.tensorrt.global.nvinfer;
import org.bytedeco.tensorrt.nvinfer.*;
import org.bytedeco.tensorrt.nvonnxparser.IParser;
import org.bytedeco.tensorrt.nvonnxparser.IParserError;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.tensorrt.global.nvinfer.*;
import static org.bytedeco.tensorrt.global.nvonnxparser.createParser;

public class SampleOnnxMNIST {
    static class Logger extends ILogger {
        @Override
        public void log(Severity severity, String message) {
            severity = severity.intern();

            if (severity == Severity.kINFO || severity == Severity.kVERBOSE) {
                return;
            }

            System.err.println("[TensorRT][" + severity + "] " + message);
        }
    }

    private static void checkCudaError(int cudaStatus, String operationName) {
        if (cudaStatus != 0) {
            try (BytePointer errorStringPointer = cudaGetErrorString(cudaStatus)) {
                String errorMessage = errorStringPointer != null ? errorStringPointer.getString() : "unknown";
                throw new IllegalStateException(operationName + " failed: " + errorMessage + " (code=" + cudaStatus + ")");
            }
        }
    }

    private static float[] loadPgmImage(String imageFilePath) throws IOException {
        try (InputStream inputStream = Files.newInputStream(Path.of(imageFilePath))) {
            DataInputStream dataInputStream = new DataInputStream(inputStream);

            {
                int value1 = dataInputStream.readUnsignedByte();
                int value2 = dataInputStream.readUnsignedByte();

                if (value1 != 0x50 || value2 != 0x35) {
                    throw new IllegalArgumentException("Only support P5.");
                }
            }

            if (dataInputStream.readUnsignedByte() != 0x0A) {
                throw new IllegalArgumentException("Unknown file format.");
            }

            StringBuilder sizeMessageBuilder = new StringBuilder();

            do {
                int value = dataInputStream.readUnsignedByte();

                if (value == 0x0A) {
                    break;
                } else {
                    sizeMessageBuilder.append((char) value);
                }
            } while (true);


            StringBuilder maxMessageBuilder = new StringBuilder();

            do {
                int value = dataInputStream.readUnsignedByte();

                if (value == 0x0A) {
                    break;
                } else {
                    maxMessageBuilder.append((char) value);
                }
            } while (true);

            String[] sizes = sizeMessageBuilder.toString().split(" ");
            int width = Integer.parseInt(sizes[0]);
            int height = Integer.parseInt(sizes[1]);
            int max = Integer.parseInt(maxMessageBuilder.toString());

            float[] data = new float[height * width];
            for (int row = 0; row < height; row++) {
                for (int column = 0; column < width; column++) {
                    data[row * width + column] = 1.0f - (float) dataInputStream.readUnsignedByte() / ((float) max);
                }
            }

            return data;
        }
    }

    public static void main(String[] arguments) throws IOException {
        // You can download the pre-trained model and input files for inference from the URL above
        // https://github.com/bytedeco/binaries/releases/download/1.5.14/javacpp-tensorrt-sample-mnist.zip
        String modelFilePath = "model.onnx";
        String[] inputImagePaths = new String[]{
                "images/0.pgm",
                "images/1.pgm",
                "images/2.pgm",
                "images/3.pgm",
                "images/4.pgm",
                "images/5.pgm",
                "images/6.pgm",
                "images/7.pgm",
                "images/8.pgm",
                "images/9.pgm"
        };
        int batchSize = inputImagePaths.length;

        if (!Files.exists(Path.of(modelFilePath))) {
            throw new IllegalStateException("Model file not found: " + modelFilePath);
        }

        for (String inputImagePath : inputImagePaths) {
            if (!Files.exists(Path.of(inputImagePath))) {
                throw new IllegalStateException("Input image file not found: " + inputImagePath);
            }
        }

        Logger sampleLogger = new Logger();

        IBuilder inferenceBuilder = null;
        INetworkDefinition networkDefinition = null;
        IParser onnxParser = null;
        IBuilderConfig builderConfiguration = null;
        IHostMemory serializedEngineMemory = null;
        IRuntime inferenceRuntime = null;
        ICudaEngine cudaEngine = null;
        IExecutionContext executionContext = null;

        PointerPointer<Pointer> deviceBufferPointers = null;

        try {
            inferenceBuilder = Objects.requireNonNull(
                    createInferBuilder(sampleLogger),
                    "createInferBuilder failed"
            );

            networkDefinition = Objects.requireNonNull(
                    inferenceBuilder.createNetworkV2(1 << NetworkDefinitionCreationFlag.kSTRONGLY_TYPED.value),
                    "createNetworkV2 failed"
            );

            onnxParser = Objects.requireNonNull(
                    createParser(networkDefinition, sampleLogger),
                    "createParser failed"
            );

            // Model parse from a file
            if (!onnxParser.parseFromFile(modelFilePath, ILogger.Severity.kWARNING.value)) {
                System.err.println("ONNX parse failed:");

                for (int parserErrorIndex = 0; parserErrorIndex < onnxParser.getNbErrors(); parserErrorIndex++) {
                    IParserError parserError = onnxParser.getError(parserErrorIndex);
                    System.err.printf("  - [%s] %s (%s:%d)%n", parserError.code(), parserError.desc(), parserError.file(), parserError.line());
                }

                return;
            }

            builderConfiguration = Objects.requireNonNull(inferenceBuilder.createBuilderConfig(), "createBuilderConfig failed");
            builderConfiguration.setBuilderOptimizationLevel(3);
            builderConfiguration.setProfilingVerbosity(ProfilingVerbosity.kLAYER_NAMES_ONLY);
            builderConfiguration.setMemoryPoolLimit(MemoryPoolType.kWORKSPACE, 1L << 28);
            builderConfiguration.setFlag(BuilderFlag.kTF32);

            serializedEngineMemory = Objects.requireNonNull(
                    inferenceBuilder.buildSerializedNetwork(networkDefinition, builderConfiguration),
                    "buildSerializedNetwork failed"
            );

            inferenceRuntime = Objects.requireNonNull(
                    createInferRuntime(sampleLogger),
                    "createInferRuntime failed"
            );

            cudaEngine = Objects.requireNonNull(
                    inferenceRuntime.deserializeCudaEngine(serializedEngineMemory.data(), serializedEngineMemory.size()),
                    "deserializeCudaEngine failed"
            );

            executionContext = Objects.requireNonNull(
                    cudaEngine.createExecutionContext(),
                    "createExecutionContext failed"
            );

            // Create input and output buffers
            int inputOutputTensorCount = cudaEngine.getNbIOTensors();
            int inputTensorIndex = -1;
            int outputTensorIndex = -1;
            long[] tensorElementCounts = new long[inputOutputTensorCount];
            long[] tensorElementSizes = new long[inputOutputTensorCount];
            deviceBufferPointers = new PointerPointer<>(inputOutputTensorCount);

            for (int tensorIndex = 0; tensorIndex < inputOutputTensorCount; tensorIndex++) {
                String tensorName = cudaEngine.getIOTensorName(tensorIndex);
                Dims64 tensorShape = executionContext.getTensorShape(tensorName);
                nvinfer.DataType tensorDataType = cudaEngine.getTensorDataType(tensorName).intern();
                TensorIOMode tensorIOMode = cudaEngine.getTensorIOMode(tensorName);

                long tensorElementCount = 1;
                long[] tensorShapeValues = new long[tensorShape.nbDims()];

                for (int dimensionIndex = 0; dimensionIndex < tensorShape.nbDims(); dimensionIndex++) {
                    long dimensionValue = tensorShape.d(dimensionIndex);

                    if (dimensionIndex == 0 && dimensionValue != batchSize) {
                        throw new IllegalStateException("Input tensor " + tensorName + " has incorrect batch size: " + dimensionValue);
                    }

                    if (dimensionValue < 0) {
                        throw new IllegalStateException("Unresolved dynamic dimension in tensor " + tensorName + ": " + dimensionValue);
                    }

                    tensorElementCount *= dimensionValue;
                    tensorShapeValues[dimensionIndex] = tensorShape.d(dimensionIndex);
                }

                if (tensorIOMode.value == TensorIOMode.kINPUT.value) {
                    if (inputTensorIndex != -1) {
                        throw new IllegalStateException("Unexpected multiple input tensors");
                    }
                    inputTensorIndex = tensorIndex;
                } else if (tensorIOMode.value == TensorIOMode.kOUTPUT.value) {
                    if (outputTensorIndex != -1) {
                        throw new IllegalStateException("Unexpected multiple output tensors");
                    }
                    outputTensorIndex = tensorIndex;
                }

                if (tensorDataType.value != nvinfer.DataType.kFLOAT.value) {
                    throw new IllegalStateException("Unsupported data type in tensor " + tensorName + ": " + tensorDataType);
                }

                tensorElementCounts[tensorIndex] = tensorElementCount;
                tensorElementSizes[tensorIndex]  = tensorElementCount * 4;
                checkCudaError(cudaMalloc(deviceBufferPointers.position(tensorIndex), tensorElementSizes[tensorIndex]), "cudaMalloc(" + tensorName + ")");

                if (!executionContext.setTensorAddress(tensorName, deviceBufferPointers.position(tensorIndex).get())) {
                    throw new IllegalStateException("setTensorAddress failed: " + tensorName);
                }

                System.out.printf("Tensor[%d] %s, mode=%s, shape=%s, type=%s, bytes=%d%n",
                                  tensorIndex, tensorName, tensorIOMode, Arrays.toString(tensorShapeValues), tensorDataType, tensorElementSizes[tensorIndex]);
            }

            // Read PGM images
            float[] inputValues = new float[Math.toIntExact(tensorElementCounts[inputTensorIndex])];
            int inputElementCountPerBatch = inputValues.length / batchSize;

            for (int imageIndex = 0; imageIndex < batchSize; imageIndex++) {
                float[] imageValues = loadPgmImage(inputImagePaths[imageIndex]);

                if (imageValues.length != inputElementCountPerBatch) {
                    throw new IllegalStateException("Image " + imageIndex + " has " + imageValues.length + " values, expected " + inputElementCountPerBatch);
                }

                System.arraycopy(imageValues, 0, inputValues, imageIndex * inputElementCountPerBatch, inputElementCountPerBatch);
            }

            // Copy image values to the device buffer
            try (FloatPointer hostInputPointer = new FloatPointer(inputValues)) {
                checkCudaError(cudaMemcpy(
                        deviceBufferPointers.position(inputTensorIndex).get(),
                        hostInputPointer,
                        tensorElementSizes[inputTensorIndex],
                        cudaMemcpyHostToDevice
                ), "cudaMemcpy HostToDevice");
            }

            // Execute inference
            executionContext.executeV2(deviceBufferPointers);

            // Copy output values from the device buffer
            float[] outputValues = new float[Math.toIntExact(tensorElementCounts[outputTensorIndex])];

            try (FloatPointer hostOutputPointer = new FloatPointer(tensorElementCounts[outputTensorIndex])) {
                checkCudaError(cudaMemcpy(
                        hostOutputPointer,
                        deviceBufferPointers.position(outputTensorIndex).get(),
                        tensorElementSizes[outputTensorIndex],
                        cudaMemcpyDeviceToHost
                ), "cudaMemcpy DeviceToHost");

                hostOutputPointer.get(outputValues);
            }

            System.out.println();
            System.out.println("Inference finished.");
            System.out.println();

            // Post-process output values
            int outputElementCountPerBatch = outputValues.length / batchSize;

            for (int batchIndex = 0; batchIndex < batchSize; batchIndex++) {
                int segmentStartIndex = batchIndex * outputElementCountPerBatch;
                float[] oneOutputValues = Arrays.copyOfRange(
                        outputValues,
                        segmentStartIndex,
                        segmentStartIndex + outputElementCountPerBatch
                );

                float bestScore = Float.NEGATIVE_INFINITY;
                int bestClassIndex = -1;

                for (int classIndex = 0; classIndex < oneOutputValues.length; classIndex++) {
                    float score = outputValues[segmentStartIndex + classIndex];

                    if (score > bestScore) {
                        bestScore      = score;
                        bestClassIndex = classIndex;
                    }
                }

                System.out.println("Image: " + inputImagePaths[batchIndex]);
                System.out.println("Predicted digit: " + bestClassIndex);
                System.out.println("Probabilities: " + Arrays.toString(oneOutputValues));
                System.out.println();
            }
        } finally {
            if (executionContext != null) {
                executionContext.deallocate();
            }

            if (cudaEngine != null) {
                cudaEngine.deallocate();
            }

            if (inferenceRuntime != null) {
                inferenceRuntime.deallocate();
            }

            if (serializedEngineMemory != null) {
                serializedEngineMemory.deallocate();
            }

            if (builderConfiguration != null) {
                builderConfiguration.deallocate();
            }

            if (onnxParser != null) {
                onnxParser.deallocate();
            }

            if (networkDefinition != null) {
                networkDefinition.deallocate();
            }

            if (inferenceBuilder != null) {
                inferenceBuilder.deallocate();
            }

            sampleLogger.deallocate();
        }
    }
}
```
