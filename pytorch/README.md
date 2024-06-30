JavaCPP Presets for PyTorch
===========================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/pytorch/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/pytorch) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/pytorch.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![pytorch](https://github.com/bytedeco/javacpp-presets/workflows/pytorch/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Apytorch)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * PyTorch 2.3.1  https://pytorch.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/pytorch/apidocs/

&lowast; The JNI bindings can instead link with [LibTorch](https://pytorch.org/cppdocs/installing.html), as long as its libraries are from the same version of PyTorch and can be found on the system library path, after setting the "org.bytedeco.javacpp.pathsFirst" system property to "true".


Sample Usage
------------
Here is a simple example of PyTorch ported to Java from this C++ source file:

 * https://pytorch.org/cppdocs/frontend.html#end-to-end-example

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `SimpleMNIST.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.pytorch</groupId>
    <artifactId>simplemnist</artifactId>
    <version>1.5.11-SNAPSHOT</version>
    <properties>
        <exec.mainClass>SimpleMNIST</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>pytorch-platform</artifactId>
            <version>2.3.1-1.5.11-SNAPSHOT</version>
        </dependency>

        <!-- Additional dependencies required to use CUDA, cuDNN, and NCCL -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>pytorch-platform-gpu</artifactId>
            <version>2.3.1-1.5.11-SNAPSHOT</version>
        </dependency>

        <!-- Additional dependencies to use bundled CUDA, cuDNN, and NCCL -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cuda-platform-redist</artifactId>
            <version>12.3-8.9-1.5.11-SNAPSHOT</version>
        </dependency>

        <!-- Additional dependencies to use bundled full version of MKL -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>mkl-platform-redist</artifactId>
            <version>2024.0-1.5.11-SNAPSHOT</version>
        </dependency>
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `SimpleMNIST.java` source file
```java
// © Copyright 2019, Torch Contributors.

import org.bytedeco.javacpp.*;
import org.bytedeco.pytorch.*;
import org.bytedeco.pytorch.Module;
import static org.bytedeco.pytorch.global.torch.*;

public class SimpleMNIST {

    // Define a new Module.
    static class Net extends Module {
        Net() {
            // Construct and register two Linear submodules.
            fc1 = register_module("fc1", new LinearImpl(784, 64));
            fc2 = register_module("fc2", new LinearImpl(64, 32));
            fc3 = register_module("fc3", new LinearImpl(32, 10));
        }

        // Implement the Net's algorithm.
        Tensor forward(Tensor x) {
            // Use one of many tensor manipulation functions.
            x = relu(fc1.forward(x.reshape(x.size(0), 784)));
            x = dropout(x, /*p=*/0.5, /*train=*/is_training());
            x = relu(fc2.forward(x));
            x = log_softmax(fc3.forward(x), /*dim=*/1);
            return x;
        }

        // Use one of many "standard library" modules.
        final LinearImpl fc1, fc2, fc3;
    }

    public static void main(String[] args) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        // Create a new Net.
        Net net = new Net();

        // Create a multi-threaded data loader for the MNIST dataset.
        MNISTMapDataset data_set = new MNIST("./data").map(new ExampleStack());
        MNISTRandomDataLoader data_loader = new MNISTRandomDataLoader(
                data_set, new RandomSampler(data_set.size().get()),
                new DataLoaderOptions(/*batch_size=*/64));

        // Instantiate an SGD optimization algorithm to update our Net's parameters.
        SGD optimizer = new SGD(net.parameters(), new SGDOptions(/*lr=*/0.01));

        for (int epoch = 1; epoch <= 10; ++epoch) {
            int batch_index = 0;
            // Iterate the data loader to yield batches from the dataset.
            for (ExampleIterator it = data_loader.begin(); !it.equals(data_loader.end()); it = it.increment()) {
                Example batch = it.access();
                // Reset gradients.
                optimizer.zero_grad();
                // Execute the model on the input data.
                Tensor prediction = net.forward(batch.data());
                // Compute a loss value to judge the prediction of our model.
                Tensor loss = nll_loss(prediction, batch.target());
                // Compute gradients of the loss w.r.t. the parameters of our model.
                loss.backward();
                // Update the parameters based on the calculated gradients.
                optimizer.step();
                // Output the loss and checkpoint every 100 batches.
                if (++batch_index % 100 == 0) {
                    System.out.println("Epoch: " + epoch + " | Batch: " + batch_index
                                     + " | Loss: " + loss.item_float());
                    // Serialize your model periodically as a checkpoint.
                    OutputArchive archive = new OutputArchive();
                    net.save(archive);
                    archive.save_to("net.pt");
                }
            }
        }
    }
}
```
