JavaCPP Presets for TensorFlow
==============================

[![Gitter](https://badges.gitter.im/bytedeco/javacpp.svg)](https://gitter.im/bytedeco/javacpp) [![Maven Central](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/tensorflow/badge.svg)](https://maven-badges.herokuapp.com/maven-central/org.bytedeco/tensorflow) [![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/https/oss.sonatype.org/org.bytedeco/tensorflow.svg)](http://bytedeco.org/builds/)  
<sup>Build status for all platforms:</sup> [![tensorflow](https://github.com/bytedeco/javacpp-presets/workflows/tensorflow/badge.svg)](https://github.com/bytedeco/javacpp-presets/actions?query=workflow%3Atensorflow)  <sup>Commercial support:</sup> [![xscode](https://img.shields.io/badge/Available%20on-xs%3Acode-blue?style=?style=plastic&logo=appveyor&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAMAAACdt4HsAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////////VXz1bAAAAAJ0Uk5T/wDltzBKAAAAlUlEQVR42uzXSwqAMAwE0Mn9L+3Ggtgkk35QwcnSJo9S+yGwM9DCooCbgn4YrJ4CIPUcQF7/XSBbx2TEz4sAZ2q1RAECBAiYBlCtvwN+KiYAlG7UDGj59MViT9hOwEqAhYCtAsUZvL6I6W8c2wcbd+LIWSCHSTeSAAECngN4xxIDSK9f4B9t377Wd7H5Nt7/Xz8eAgwAvesLRjYYPuUAAAAASUVORK5CYII=)](https://xscode.com/bytedeco/javacpp-presets)


Introduction
------------
This directory contains the JavaCPP Presets module for:

 * TensorFlow 1.15.5  http://www.tensorflow.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/tensorflow/apidocs/

&lowast; Call `Loader.load(org.bytedeco.tensorflow.presets.tensorflow.class)` before using the API in the `org.tensorflow` package.  
&lowast; Call `Py_Initialize(cachePackages())` instead of just `Py_Initialize()`.


Sample Usage
------------
Here is a simple example of TensorFlow ported to Java from this C++ source file:

 * https://github.com/tensorflow/tensorflow/blob/r1.0/tensorflow/cc/tutorials/example_trainer.cc

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `ExampleTrainer.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.tensorflow</groupId>
    <artifactId>exampletrainer</artifactId>
    <version>1.5.7</version>
    <properties>
        <exec.mainClass>ExampleTrainer</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorflow-platform</artifactId>
            <version>1.15.5-1.5.7</version>
        </dependency>

        <!-- Additional dependencies required to use CUDA, cuDNN, NCCL, and TensorRT -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorflow-platform-gpu</artifactId>
            <version>1.15.5-1.5.7</version>
        </dependency>

        <!-- Additional dependencies to use bundled CUDA, cuDNN, NCCL, and TensorRT -->
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>cuda-platform-redist</artifactId>
            <version>11.6-8.3-1.5.7</version>
        </dependency>

        <!-- Optional dependencies to load Python-enabled builds -->
<!--
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorflow-platform-python</artifactId>
            <version>1.15.5-1.5.7</version>
        </dependency>
        <dependency>
            <groupId>org.bytedeco</groupId>
            <artifactId>tensorflow-platform-python-gpu</artifactId>
            <version>1.15.5-1.5.7</version>
        </dependency>
-->
    </dependencies>
    <build>
        <sourceDirectory>.</sourceDirectory>
    </build>
</project>
```

### The `ExampleTrainer.java` source file
```java
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

import java.nio.FloatBuffer;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.bytedeco.tensorflow.*;
import static org.bytedeco.tensorflow.global.tensorflow.*;

public class ExampleTrainer {

    static class Options {
        int num_concurrent_sessions = 10; // The number of concurrent sessions
        int num_concurrent_steps = 10;    // The number of concurrent steps
        int num_iterations = 100;         // Each step repeats this many times
        boolean use_gpu = false;          // Whether to use gpu in the training
    }

    static Options ParseCommandLineFlags(String[] args) throws Exception {
        Options options = new Options();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i], value;
            if (arg.startsWith("--")) {
                arg = arg.substring(2);
            } else if (arg.startsWith("-")) {
                arg = arg.substring(1);
            } else {
                continue;
            }
            int j = arg.indexOf('=');
            if (j < 0) {
                value = args[++i];
            } else {
                value = arg.substring(j + 1);
                arg = arg.substring(0, j);
            }
            try {
                java.lang.reflect.Field field = Options.class.getField(arg);
                Class cls = field.getDeclaringClass();
                if (cls == String.class) {
                    field.set(options, value);
                } else if (cls == int.class) {
                    field.setInt(options, Integer.parseInt(value));
                } else if (cls == boolean.class) {
                    field.setBoolean(options, Boolean.parseBoolean(value));
                } else {
                    throw new Exception("Unsupported option type: " + cls);
                }
            } catch (NoSuchFieldException e) {
                throw new Exception("Unknown command line flag: " + arg);
            } catch (NumberFormatException e) {
                throw new Exception("Error parsing command line flag: " + value);
            }
        }
        return options;
    }

    // A = [3 2; -1 0]; x = rand(2, 1);
    // We want to compute the largest eigenvalue for A.
    // repeat x = y / y.norm(); y = A * x; end
    static GraphDef CreateGraphDef() throws Exception {
        // TODO(jeff,opensource): This should really be a more interesting
        // computation.  Maybe turn this into an mnist model instead?
        Scope root = Scope.NewRootScope();

        // a = [3 2; -1 0]
        Output a = Const(root, Tensor.create(new float[] {3.f, 2.f, -1.f, 0.f}, new TensorShape(2, 2)));

        // x = [1.0; 1.0]
        Output x = Const(root.WithOpName("x"), Tensor.create(new float[] {1.f, 1.f}, new TensorShape(2, 1)));

        // y = a * x
        MatMul y = new MatMul(root.WithOpName("y"), new Input(a), new Input(x));

        // y2 = y.^2
        Square y2 = new Square(root, y.asInput());

        // y2_sum = sum(y2)
        Sum y2_sum = new Sum(root, y2.asInput(), new Input(0));

        // y_norm = sqrt(y2_sum)
        Sqrt y_norm = new Sqrt(root, y2_sum.asInput());

        // y_normalized = y ./ y_norm
        new Div(root.WithOpName("y_normalized"), y.asInput(), y_norm.asInput());

        GraphDef def = new GraphDef();
        Status s = root.ToGraphDef(def);
        if (!s.ok()) {
            throw new Exception(s.error_message().getString());
        }
        return def;
    }

    static String DebugString(Tensor x, Tensor y) {
        assert x.NumElements() == 2;
        assert y.NumElements() == 2;
        FloatBuffer x_flat = x.createBuffer();
        FloatBuffer y_flat = y.createBuffer();
        // Compute an estimate of the eigenvalue via
        //      (x' A x) / (x' x) = (x' y) / (x' x)
        // and exploit the fact that x' x = 1 by assumption
        float lambda = x_flat.get(0) * y_flat.get(0) + x_flat.get(1) * y_flat.get(1);
        return String.format("lambda = %8.6f x = [%8.6f %8.6f] y = [%8.6f %8.6f]",
                             lambda, x_flat.get(0), x_flat.get(1), y_flat.get(0), y_flat.get(1));
    }

    static void ConcurrentSteps(final Options opts, final int session_index) throws Exception {
        // Creates a session.
        SessionOptions options = new SessionOptions();
        final Session session = new Session(options);
        GraphDef def = CreateGraphDef();
        if (options.target() == null) {
            SetDefaultDevice(opts.use_gpu ? "/gpu:0" : "/cpu:0", def);
        }

        Status s = session.Create(def);
        if (!s.ok()) {
            throw new Exception(s.error_message().getString());
        }

        // Spawn M threads for M concurrent steps.
        int M = opts.num_concurrent_steps;
        ExecutorService step_threads = Executors.newFixedThreadPool(M);

        for (int step = 0; step < M; step++) {
            final int m = step;
            step_threads.submit(new Callable<Void>() { public Void call() throws Exception {
                // Randomly initialize the input.
                Tensor x = new Tensor(DT_FLOAT, new TensorShape(2, 1));
                FloatBuffer x_flat = x.createBuffer();
                x_flat.put(0, (float)Math.random());
                x_flat.put(1, (float)Math.random());
                float inv_norm = 1 / (float)Math.sqrt(x_flat.get(0) *  x_flat.get(0) + x_flat.get(1) *  x_flat.get(1));
                x_flat.put(0, x_flat.get(0) * inv_norm);
                x_flat.put(1, x_flat.get(1) * inv_norm);

                // Iterations.
                TensorVector outputs = new TensorVector();
                for (int iter = 0; iter < opts.num_iterations; iter++) {
                    outputs.resize(0);
                    Status s = session.Run(new StringTensorPairVector(new String[] {"x"}, new Tensor[] {x}),
                                           new StringVector("y:0", "y_normalized:0"), new StringVector(), outputs);
                    if (!s.ok()) {
                        throw new Exception(s.error_message().getString());
                    }
                    assert outputs.size() == 2;

                    Tensor y = outputs.get(0);
                    Tensor y_norm = outputs.get(1);
                    // Print out lambda, x, and y.
                    System.out.printf("%06d/%06d %s\n", session_index, m, DebugString(x, y));
                    // Copies y_normalized to x.
                    x.put(y_norm);
                }
                return null;
            }});
        }

        step_threads.shutdown();
        step_threads.awaitTermination(1, TimeUnit.MINUTES);
        s = session.Close();
        if (!s.ok()) {
            throw new Exception(s.error_message().getString());
        }
    }

    static void ConcurrentSessions(final Options opts) throws Exception {
        // Spawn N threads for N concurrent sessions.
        int N = opts.num_concurrent_sessions;
        ExecutorService session_threads = Executors.newFixedThreadPool(N);
        for (int i = 0; i < N; i++) {
            final int n = i;
            session_threads.submit(new Callable<Void>() { public Void call() throws Exception {
                ConcurrentSteps(opts, n);
                return null;
            }});
        }
        session_threads.shutdown();
    }

    public static void main(String args[]) throws Exception {
        Options opts = ParseCommandLineFlags(args);
        InitMain("trainer", (int[])null, null);
        ConcurrentSessions(opts);
    }
}
```
