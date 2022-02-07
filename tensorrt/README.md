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

 * TensorRT 8.2.3.0  https://developer.nvidia.com/tensorrt

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/tensorrt/apidocs/


Sample Usage
------------
Here is a simple example of TensorRT ported to Java from the `sampleGoogleNet.cpp` sample file included in `TensorRT-4.0.0.3.Ubuntu-16.04.4.x86_64-gnu.cuda-9.0.cudnn7.0.tar.gz` available at:

 * https://developer.nvidia.com/nvidia-tensorrt-download

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SampleGoogleNet.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.tensorrt</groupId>
    <artifactId>samplegooglenet</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>SampleGoogleNet</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorrt-platform</artifactId>
            <version>8.2-1.5.7</version>
        </dependency>

        <!-- Additional dependencies to use bundled CUDA, cuDNN, NCCL, and TensorRT -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cuda-platform-redist</artifactId>
            <version>11.6-8.3-1.5.7</version>
        </dependency>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorrt-platform-redist</artifactId>
            <version>8.2-1.5.7</version>
        </dependency>

    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SampleGoogleNet.java` source file
```java
import java.io.*;
import java.util.*;
import org.bytedeco.javacpp.*;

import org.bytedeco.cuda.cudart.*;
import org.bytedeco.tensorrt.nvinfer.*;
import org.bytedeco.tensorrt.nvparsers.*;
import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.tensorrt.global.nvinfer.*;
import static org.bytedeco.tensorrt.global.nvparsers.*;

public class SampleGoogleNet {
    static void CHECK(int status)
    {
        if (status != 0)
        {
            System.out.println("Cuda failure: " + status);
            System.exit(6);
        }
    }

    // Logger for GIE info/warning/errors
    static class Logger extends ILogger
    {
        @Override public void log(Severity severity, String msg)
        {
            severity = severity.intern();

            // suppress info-level messages
            if (severity == Severity.kINFO) return;

            switch (severity)
            {
                case kINTERNAL_ERROR: System.err.print("INTERNAL_ERROR: "); break;
                case kERROR: System.err.print("ERROR: "); break;
                case kWARNING: System.err.print("WARNING: "); break;
                case kINFO: System.err.print("INFO: "); break;
                default: System.err.print("UNKNOWN: "); break;
            }
            System.err.println(msg);
        }
    }
    static Logger gLogger = new Logger();

    static String locateFile(String input, String[] directories)
    {
        String file = "";
        int MAX_DEPTH = 10;
        boolean found = false;
        for (String dir : directories)
        {
            file = dir + input;
            for (int i = 0; i < MAX_DEPTH && !found; i++)
            {
                File checkFile = new File(file);
                found = checkFile.exists();
                if (found) break;
                file = "../" + file;
            }
            if (found) break;
            file = "";
        }

        if (file.isEmpty())
            System.err.println("Could not find a file due to it not existing in the data directory.");
        return file;
    }

    // stuff we know about the network and the caffe input/output blobs

    static int BATCH_SIZE = 4;
    static int TIMING_ITERATIONS = 1000;

    static String INPUT_BLOB_NAME = "data";
    static String OUTPUT_BLOB_NAME = "prob";


    static String locateFile(String input)
    {
        String[] dirs = {"data/samples/googlenet/", "data/googlenet/"};
        return locateFile(input, dirs);
    }

    static class Profiler extends IProfiler
    {
        LinkedHashMap<String, Float> mProfile = new LinkedHashMap<String, Float>();

        @Override public void reportLayerTime(String layerName, float ms)
        {
            Float time = mProfile.get(layerName);
            mProfile.put(layerName, (time != null ? time : 0) + ms);
        }

        public void printLayerTimes()
        {
            float totalTime = 0;
            for (Map.Entry<String,Float> e : mProfile.entrySet())
            {
                System.out.printf("%-40.40s %4.3fms\n", e.getKey(), e.getValue() / TIMING_ITERATIONS);
                totalTime += e.getValue();
            }
            System.out.printf("Time over all layers: %4.3f\n", totalTime / TIMING_ITERATIONS);
        }

    }
    static Profiler gProfiler = new Profiler();

    static void caffeToGIEModel(String deployFile,     // name for caffe prototxt
                         String modelFile,             // name for model 
                         String[] outputs,             // network outputs
                         int maxBatchSize,             // batch size - NB must be at least as large as the batch we want to run with)
                         IHostMemory[] gieModelStream)
    {
        // create API root class - must span the lifetime of the engine usage
        IBuilder builder = createInferBuilder(gLogger);
        INetworkDefinition network = builder.createNetwork();

        // parse the caffe model to populate the network, then set the outputs
        ICaffeParser parser = createCaffeParser();

        boolean useFp16 = builder.platformHasFastFp16();

        DataType modelDataType = useFp16 ? DataType.kHALF : DataType.kFLOAT; // create a 16-bit model if it's natively supported
        IBlobNameToTensor blobNameToTensor =
            parser.parse(locateFile(deployFile),                // caffe deploy file
                                     locateFile(modelFile),     // caffe model file
                                     network,                   // network definition that the parser will populate
                                     modelDataType);

        assert blobNameToTensor != null;
        // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate    
        for (String s : outputs)
            network.markOutput(blobNameToTensor.find(s));

        // Build the engine
        builder.setMaxBatchSize(maxBatchSize);
        builder.setMaxWorkspaceSize(16 << 20);

        // set up the network for paired-fp16 format if available
        if(useFp16)
            builder.setHalf2Mode(true);

        ICudaEngine engine = builder.buildCudaEngine(network);
        assert engine != null;

        // we don't need the network any more, and we can destroy the parser
        network.destroy();
        parser.destroy();

        // serialize the engine, then close everything down
        gieModelStream[0] = engine.serialize();
        engine.destroy();
        builder.destroy();
        shutdownProtobufLibrary();
    }

    static void timeInference(ICudaEngine engine, int batchSize)
    {
        // input and output buffer pointers that we pass to the engine - the engine requires exactly ICudaEngine::getNbBindings(),
        // of these, but in this case we know that there is exactly one input and one output.
        assert engine.getNbBindings() == 2;
        PointerPointer buffers = new PointerPointer(2);

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than ICudaEngine::getNbBindings()
        int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME), outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

        // allocate GPU buffers
        DimsCHW inputDims = new DimsCHW(engine.getBindingDimensions(inputIndex)), outputDims = new DimsCHW(engine.getBindingDimensions(outputIndex));
        long inputSize = batchSize * inputDims.c().get() * inputDims.h().get() * inputDims.w().get() * Float.SIZE / 8;
        long outputSize = batchSize * outputDims.c().get() * outputDims.h().get() * outputDims.w().get() * Float.SIZE / 8;

        CHECK(cudaMalloc(buffers.position(inputIndex), inputSize));
        CHECK(cudaMalloc(buffers.position(outputIndex), outputSize));

        IExecutionContext context = engine.createExecutionContext();
        context.setProfiler(gProfiler);

        // zero the input buffer
        CHECK(cudaMemset(buffers.position(inputIndex).get(), 0, inputSize));

        for (int i = 0; i < TIMING_ITERATIONS;i++)
            context.execute(batchSize, buffers.position(0));

        // release the context and buffers
        context.destroy();
        CHECK(cudaFree(buffers.position(inputIndex).get()));
        CHECK(cudaFree(buffers.position(outputIndex).get()));
    }


    public static void main(String[] args)
    {
        System.out.println("Building and running a GPU inference engine for GoogleNet, N=4...");

        // parse the caffe model and the mean file
        IHostMemory[] gieModelStream = { null };
        caffeToGIEModel("googlenet.prototxt", "googlenet.caffemodel", new String[] { OUTPUT_BLOB_NAME }, BATCH_SIZE, gieModelStream);

        // create an engine
        IRuntime infer = createInferRuntime(gLogger);
        ICudaEngine engine = infer.deserializeCudaEngine(gieModelStream[0].data(), gieModelStream[0].size(), null);

        System.out.println("Bindings after deserializing:"); 
        for (int bi = 0; bi < engine.getNbBindings(); bi++) { 
            if (engine.bindingIsInput(bi)) { 
                System.out.printf("Binding %d (%s): Input.\n",  bi, engine.getBindingName(bi));
            } else { 
                System.out.printf("Binding %d (%s): Output.\n", bi, engine.getBindingName(bi));
            } 
        }

        // run inference with null data to time network performance
        timeInference(engine, BATCH_SIZE);

        engine.destroy();
        infer.destroy();

        gProfiler.printLayerTimes();

        System.out.println("Done.");

        System.exit(0);
    }
}
```
