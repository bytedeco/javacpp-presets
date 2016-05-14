JavaCPP Presets for CUDA
========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * CUDA 7.5  https://developer.nvidia.com/cuda-zone

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/cuda/apidocs/

&lowast; We can also [use Thrust with JavaCPP](https://github.com/bytedeco/javacpp/wiki/Interface-Thrust-and-CUDA).

Sample Usage
------------
Here is a simple example of cuDNN ported to Java from the `mnistCUDNN.cpp` sample file included in `cudnn-sample-v2.tgz` available at:

 * https://developer.nvidia.com/cudnn

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/MNISTCUDNN.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.cuda</groupId>
    <artifactId>mnistcudnn</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>MNISTCUDNN</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>cuda</artifactId>
            <version>7.5-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/MNISTCUDNN.java` source file
```java
/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

/*
 * This example demonstrates how to use CUDNN library to implement forward
 * pass. The sample loads weights and biases from trained network,
 * takes a few images of digits and recognizes them. The network was trained on 
 * the MNIST dataset using Caffe. The network consists of two 
 * convolution layers, two pooling layers, one relu and two 
 * fully connected layers. Final layer gets processed by Softmax. 
 * cublasSgemv is used to implement fully connected layers.
 */

import java.io.*;
import org.bytedeco.javacpp.*;

import static org.bytedeco.javacpp.cublas.*;
import static org.bytedeco.javacpp.cuda.*;
import static org.bytedeco.javacpp.cudnn.*;

public class MNISTCUDNN {
    static final int IMAGE_H = 28;
    static final int IMAGE_W = 28;

    static final String first_image = "one_28x28.pgm";
    static final String second_image = "three_28x28.pgm";
    static final String third_image = "five_28x28.pgm";

    static final String conv1_bin = "conv1.bin";
    static final String conv1_bias_bin = "conv1.bias.bin";
    static final String conv2_bin = "conv2.bin";
    static final String conv2_bias_bin = "conv2.bias.bin";
    static final String ip1_bin = "ip1.bin";
    static final String ip1_bias_bin = "ip1.bias.bin";
    static final String ip2_bin = "ip2.bin";
    static final String ip2_bias_bin = "ip2.bias.bin";

    /********************************************************
     * Prints the error message, and exits
     * ******************************************************/

    static final int EXIT_FAILURE = 1;
    static final int EXIT_SUCCESS = 0;
    static final int EXIT_WAIVED = 0;

    static void FatalError(String s) {
        System.err.println(s);
        Thread.dumpStack();
        System.err.println("Aborting...");
        cudaDeviceReset();
        System.exit(EXIT_FAILURE);
    }

    static void checkCUDNN(int status) {
        if (status != CUDNN_STATUS_SUCCESS) {
            FatalError("CUDNN failure: " + status);
        }
    }

    static void checkCudaErrors(int status) {
        if (status != 0) {
            FatalError("Cuda failure: " + status);
        }
    }

    static String get_path(String fname, String pname) {
        return "data/" + fname;
    }

    static class Layer_t {
        int inputs = 0;
        int outputs = 0;
        // linear dimension (i.e. size is kernel_dim * kernel_dim)
        int kernel_dim = 0;
        FloatPointer[] data_h = new FloatPointer[1], data_d = new FloatPointer[1];
        FloatPointer[] bias_h = new FloatPointer[1], bias_d = new FloatPointer[1];
        Layer_t(int _inputs, int _outputs, int _kernel_dim, String fname_weights,
                String fname_bias, String pname) {
            inputs = _inputs; outputs = _outputs; kernel_dim = _kernel_dim;
            String weights_path, bias_path;
            if (pname != null) {
                weights_path = get_path(fname_weights, pname);
                bias_path = get_path(fname_bias, pname);
            } else {
                weights_path = fname_weights; bias_path = fname_bias;
            }
            readBinaryFile(weights_path, inputs * outputs * kernel_dim * kernel_dim,
                            data_h, data_d);
            readBinaryFile(bias_path, outputs, bias_h, bias_d);
        }
        public void release() {
            checkCudaErrors( cudaFree(data_d[0]) );
        }

        private void readBinaryFile(String fname, int size, FloatPointer[] data_h, FloatPointer[] data_d) {
            try {
                FileInputStream stream = new FileInputStream(fname);
                int size_b = size*Float.BYTES;
                byte[] data = new byte[size_b];
                if (stream.read(data) < size_b) {
                    FatalError("Error reading file " + fname);
                }
                stream.close();
                data_h[0] = new FloatPointer(new BytePointer(data));
                data_d[0] = new FloatPointer();

                checkCudaErrors( cudaMalloc(data_d[0], size_b) );
                checkCudaErrors( cudaMemcpy(data_d[0], data_h[0],
                                            size_b,
                                            cudaMemcpyHostToDevice) );
            } catch (IOException e) {
                FatalError("Error opening file " + fname);
            }
        }
    }

    static void printDeviceVector(int size, FloatPointer vec_d) {
        FloatPointer vec = new FloatPointer(size);
        cudaDeviceSynchronize();
        cudaMemcpy(vec, vec_d, size*Float.BYTES, cudaMemcpyDeviceToHost);
        for (int i = 0; i < size; i++) {
            System.out.print(vec.get(i) + " ");
        }
        System.out.println();
    }

    static class network_t {
        int dataType = CUDNN_DATA_FLOAT;
        int tensorFormat = CUDNN_TENSOR_NCHW;
        cudnnContext cudnnHandle = new cudnnContext();
        cudnnTensorStruct srcTensorDesc = new cudnnTensorStruct(),
                          dstTensorDesc = new cudnnTensorStruct(),
                          biasTensorDesc = new cudnnTensorStruct();
        cudnnFilterStruct filterDesc = new cudnnFilterStruct();
        cudnnConvolutionStruct convDesc = new cudnnConvolutionStruct();
        cudnnPoolingStruct poolingDesc = new cudnnPoolingStruct();
        cublasContext cublasHandle = new cublasContext();
        void createHandles() {
            checkCUDNN( cudnnCreate(cudnnHandle) );
            checkCUDNN( cudnnCreateTensorDescriptor(srcTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(dstTensorDesc) );
            checkCUDNN( cudnnCreateTensorDescriptor(biasTensorDesc) );
            checkCUDNN( cudnnCreateFilterDescriptor(filterDesc) );
            checkCUDNN( cudnnCreateConvolutionDescriptor(convDesc) );
            checkCUDNN( cudnnCreatePoolingDescriptor(poolingDesc) );

            checkCudaErrors( cublasCreate_v2(cublasHandle) );
        }
        void destroyHandles() {
            checkCUDNN( cudnnDestroyPoolingDescriptor(poolingDesc) );
            checkCUDNN( cudnnDestroyConvolutionDescriptor(convDesc) );
            checkCUDNN( cudnnDestroyFilterDescriptor(filterDesc) );
            checkCUDNN( cudnnDestroyTensorDescriptor(srcTensorDesc) );
            checkCUDNN( cudnnDestroyTensorDescriptor(dstTensorDesc) );
            checkCUDNN( cudnnDestroyTensorDescriptor(biasTensorDesc) );
            checkCUDNN( cudnnDestroy(cudnnHandle) );

            checkCudaErrors( cublasDestroy_v2(cublasHandle) );
        }

        public network_t() {
            createHandles();
        }
        public void release() {
            destroyHandles();
        }

        public void resize(int size, FloatPointer data) {
            if (!data.isNull()) {
                checkCudaErrors( cudaFree(data) );
            }
            checkCudaErrors( cudaMalloc(data, size*Float.BYTES) );
        }
        void addBias(cudnnTensorStruct dstTensorDesc, Layer_t layer, int c, FloatPointer data) {
            checkCUDNN( cudnnSetTensor4dDescriptor(biasTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    1, c,
                                                    1,
                                                    1) );
            FloatPointer alpha = new FloatPointer(1.0f);
            FloatPointer beta  = new FloatPointer(1.0f);
            checkCUDNN( cudnnAddTensor(cudnnHandle, CUDNN_ADD_SAME_C,
                                          alpha, biasTensorDesc,
                                          layer.bias_d[0],
                                          beta,
                                          dstTensorDesc,
                                          data) );
        }
        void fullyConnectedForward(Layer_t ip,
                              int[] n, int[] c, int[] h, int[] w,
                              FloatPointer srcData, FloatPointer dstData) {
            if (n[0] != 1) {
                FatalError("Not Implemented");
            }
            int dim_x = c[0]*h[0]*w[0];
            int dim_y = ip.outputs;
            resize(dim_y, dstData);

            FloatPointer alpha = new FloatPointer(1.0f), beta = new FloatPointer(1.0f);
            // place bias into dstData
            checkCudaErrors( cudaMemcpy(dstData, ip.bias_d[0], dim_y*Float.BYTES, cudaMemcpyDeviceToDevice) );

            checkCudaErrors( cublasSgemv_v2(cublasHandle, CUBLAS_OP_T,
                                          dim_x, dim_y,
                                          alpha,
                                          ip.data_d[0], dim_x,
                                          srcData, 1,
                                          beta,
                                          dstData, 1) );

            h[0] = 1; w[0] = 1; c[0] = dim_y;
        }
        void convoluteForward(Layer_t conv,
                              int[] n, int[] c, int[] h, int[] w,
                              FloatPointer srcData, FloatPointer dstData) {
            int[] algo = new int[1];

            checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n[0], c[0],
                                                    h[0], w[0]) );

            checkCUDNN( cudnnSetFilter4dDescriptor(filterDesc,
                                                  dataType,
                                                  conv.outputs,
                                                  conv.inputs,
                                                  conv.kernel_dim,
                                                  conv.kernel_dim) );

            checkCUDNN( cudnnSetConvolution2dDescriptor(convDesc,
                                                       // srcTensorDesc,
                                                        //filterDesc,
                                                        0,0, // padding
                                                        1,1, // stride
                                                        1,1, // upscale
                                                        CUDNN_CROSS_CORRELATION) );
            // find dimension of convolution output
            checkCUDNN( cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    n, c, h, w) );

            checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n[0], c[0],
                                                    h[0],
                                                    w[0]) );
            checkCUDNN( cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    dstTensorDesc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    algo
                                                    ) );
            resize(n[0]*c[0]*h[0]*w[0], dstData);
            SizeTPointer sizeInBytes=new SizeTPointer(1);
            Pointer workSpace=new Pointer();
            checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                    srcTensorDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    dstTensorDesc,
                                                    algo[0],
                                                    sizeInBytes) );
            if (sizeInBytes.get(0)!=0) {
                checkCudaErrors( cudaMalloc(workSpace,sizeInBytes.get(0)) );
            }
            FloatPointer alpha = new FloatPointer(1.0f);
            FloatPointer beta  = new FloatPointer(0.0f);
            checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                                  alpha,
                                                  srcTensorDesc,
                                                  srcData,
                                                  filterDesc,
                                                  conv.data_d[0],
                                                  convDesc,
                                                  algo[0],
                                                  workSpace,
                                                  sizeInBytes.get(0),
                                                  beta,
                                                  dstTensorDesc,
                                                  dstData) );
            addBias(dstTensorDesc, conv, c[0], dstData);
            if (sizeInBytes.get(0)!=0) {
                checkCudaErrors( cudaFree(workSpace) );
            }
        }

        void poolForward(int[] n, int[] c, int[] h, int[] w,
                         FloatPointer srcData, FloatPointer dstData) {
            checkCUDNN( cudnnSetPooling2dDescriptor(poolingDesc,
                                                    CUDNN_POOLING_MAX,
                                                    2, 2, // window
                                                    0, 0, // padding
                                                    2, 2  // stride
                                                    ) );
            checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n[0], c[0],
                                                    h[0],
                                                    w[0] ) );
            h[0] = h[0] / 2; w[0] = w[0] / 2;
            checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n[0], c[0],
                                                    h[0],
                                                    w[0]) );
            resize(n[0]*c[0]*h[0]*w[0], dstData);
            FloatPointer alpha = new FloatPointer(1.0f);
            FloatPointer beta = new FloatPointer(0.0f);
            checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                              poolingDesc,
                                              alpha,
                                              srcTensorDesc,
                                              srcData,
                                              beta,
                                              dstTensorDesc,
                                              dstData) );
        }
        void softmaxForward(int n, int c, int h, int w, FloatPointer srcData, FloatPointer dstData) {
            resize(n*c*h*w, dstData);

            checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n, c,
                                                    h,
                                                    w) );
            checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n, c,
                                                    h,
                                                    w) );
            FloatPointer alpha = new FloatPointer(1.0f);
            FloatPointer beta  = new FloatPointer(0.0f);
            checkCUDNN( cudnnSoftmaxForward(cudnnHandle,
                                              CUDNN_SOFTMAX_ACCURATE ,
                                              CUDNN_SOFTMAX_MODE_CHANNEL,
                                              alpha,
                                              srcTensorDesc,
                                              srcData,
                                              beta,
                                              dstTensorDesc,
                                              dstData) );
        }
        void activationForward(int n, int c, int h, int w, FloatPointer srcData, FloatPointer dstData) {
            resize(n*c*h*w, dstData);
            checkCUDNN( cudnnSetTensor4dDescriptor(srcTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n, c,
                                                    h,
                                                    w) );
            checkCUDNN( cudnnSetTensor4dDescriptor(dstTensorDesc,
                                                    tensorFormat,
                                                    dataType,
                                                    n, c,
                                                    h,
                                                    w) );
            FloatPointer alpha = new FloatPointer(1.0f);
            FloatPointer beta  = new FloatPointer(0.0f);
            checkCUDNN( cudnnActivationForward(cudnnHandle,
                                                CUDNN_ACTIVATION_RELU,
                                                alpha,
                                                srcTensorDesc,
                                                srcData,
                                                beta,
                                                dstTensorDesc,
                                                dstData) );
        }

        int classify_example(String fname, Layer_t conv1,
                             Layer_t conv2,
                             Layer_t ip1,
                             Layer_t ip2) {
            int[] n = new int[1], c = new int[1], h = new int[1], w = new int[1];
            FloatPointer srcData = new FloatPointer(), dstData = new FloatPointer();
            FloatPointer imgData_h = new FloatPointer(IMAGE_H*IMAGE_W);

            // load gray-scale image from disk
            System.out.println("Loading image " + fname);
            try {
                // declare a host image object for an 8-bit grayscale image
                FileInputStream oHostSrc = new FileInputStream(fname);
                int lines = 0;
                while (lines < 4) {
                    // skip header, comment, width, height, and max value
                    if (oHostSrc.read() == '\n') {
                        lines++;
                    }
                }

                // Plot to console and normalize image to be in range [0,1]
                for (int i = 0; i < IMAGE_H; i++) {
                    for (int j = 0; j < IMAGE_W; j++) {
                        int idx = IMAGE_W*i + j;
                        imgData_h.put(idx, oHostSrc.read() / 255.0f);
                    }
                }
                oHostSrc.close();
            } catch (IOException rException) {
                FatalError(rException.toString());
            }

            System.out.println("Performing forward propagation ...");

            checkCudaErrors( cudaMalloc(srcData, IMAGE_H*IMAGE_W*Float.BYTES) );
            checkCudaErrors( cudaMemcpy(srcData, imgData_h,
                                        IMAGE_H*IMAGE_W*Float.BYTES,
                                        cudaMemcpyHostToDevice) );

            n[0] = c[0] = 1; h[0] = IMAGE_H; w[0] = IMAGE_W;
            convoluteForward(conv1, n, c, h, w, srcData, dstData);
            poolForward(n, c, h, w, dstData, srcData);

            convoluteForward(conv2, n, c, h, w, srcData, dstData);
            poolForward(n, c, h, w, dstData, srcData);

            fullyConnectedForward(ip1, n, c, h, w, srcData, dstData);
            activationForward(n[0], c[0], h[0], w[0], dstData, srcData);

            fullyConnectedForward(ip2, n, c, h, w, srcData, dstData);
            softmaxForward(n[0], c[0], h[0], w[0], dstData, srcData);

            final int max_digits = 10;
            FloatPointer result = new FloatPointer(max_digits);
            checkCudaErrors( cudaMemcpy(result, srcData, max_digits*Float.BYTES, cudaMemcpyDeviceToHost) );
            int id = 0;
            for (int i = 1; i < max_digits; i++) {
                if (result.get(id) < result.get(i)) id = i;
            }

            System.out.println("Resulting weights from Softmax:");
            printDeviceVector(n[0]*c[0]*h[0]*w[0], srcData);

            checkCudaErrors( cudaFree(srcData) );
            checkCudaErrors( cudaFree(dstData) );
            return id;
        }
    }

    public static void main(String[] args) {
        if (args.length > 1) {
            System.out.println("Test usage:\njava MNISTCUDNN [image]\nExiting...");
            System.exit(EXIT_FAILURE);
        }

        String image_path;
        network_t mnist = new network_t();
        String name = MNISTCUDNN.class.getName();

        if(Loader.sizeof(Pointer.class) != 8 && !Loader.getPlatform().contains("arm")) {
            System.out.println("With the exception of ARM, " + name + " is only supported on 64-bit OS and the application must be built as a 64-bit target. Test is being waived.");
            System.exit(EXIT_WAIVED);
        }

        Layer_t conv1 = new Layer_t(1,20,5,conv1_bin,conv1_bias_bin,name);
        Layer_t conv2 = new Layer_t(20,50,5,conv2_bin,conv2_bias_bin,name);
        Layer_t   ip1 = new Layer_t(800,500,1,ip1_bin,ip1_bias_bin,name);
        Layer_t   ip2 = new Layer_t(500,10,1,ip2_bin,ip2_bias_bin,name);

        if (args.length == 0) {
            int i1,i2,i3;
            image_path = get_path(first_image, name);
            i1 = mnist.classify_example(image_path, conv1, conv2, ip1, ip2);

            image_path = get_path(second_image, name);
            i2 = mnist.classify_example(image_path, conv1, conv2, ip1, ip2);

            image_path = get_path(third_image, name);
            i3 = mnist.classify_example(image_path, conv1, conv2, ip1, ip2);

            System.out.println("\nResult of classification: " + i1 + " " + i2 + " " + i3);
            if (i1 != 1 || i2 != 3 || i3 != 5) {
                System.out.println("\nTest failed!");
                FatalError("Prediction mismatch");
            } else {
                System.out.println("\nTest passed!");
            }
        } else {
            int i1 = mnist.classify_example(args[0], conv1, conv2, ip1, ip2);
            System.out.println("\nResult of classification: " + i1);
        }
        cudaDeviceReset();
        System.exit(EXIT_SUCCESS);
    }
}
```
