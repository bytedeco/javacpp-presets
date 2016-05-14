JavaCPP Presets for MXNet
=========================

Introduction
------------
This directory contains the JavaCPP Presets module for:

 * MXNet  http://mxnet.readthedocs.org/

Please refer to the parent README.md file for more detailed information about the JavaCPP Presets.


Documentation
-------------
Java API documentation is available here:

 * http://bytedeco.org/javacpp-presets/mxnet/apidocs/

&lowast; Bindings are currently available only for the C API of MXNet.

Sample Usage
------------
Here is a simple example of the predict API of MXNet ported to Java from this C++ source file and for this data:

 * https://github.com/dmlc/mxnet/blob/master/example/cpp/image-classification/image-classification-predict.cc
 * http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz

We can use [Maven 3](http://maven.apache.org/) to download and install automatically all the class files as well as the native binaries. To run this sample code, after creating the `pom.xml` and `src/main/java/ImageClassificationPredict.java` source files below, simply execute on the command line:
```bash
 $ mvn compile exec:java -Dexec.args="apple.jpg"
```

### The `pom.xml` build file
```xml
<project>
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.bytedeco.javacpp-presets.mxnet</groupId>
    <artifactId>ImageClassificationPredict</artifactId>
    <version>1.2</version>
    <properties>
        <exec.mainClass>ImageClassificationPredict</exec.mainClass>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.bytedeco.javacpp-presets</groupId>
            <artifactId>mxnet</artifactId>
            <version>master-1.2</version>
        </dependency>
    </dependencies>
</project>
```

### The `src/main/java/ImageClassificationPredict.java` source file
```java
/*!
 *  Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 */

//
//  File: image-classification-predict.cpp
//  This is a simple predictor which shows
//  how to use c api for image classfication
//  It uses opencv for image reading
//  Created by liuxiao on 12/9/15.
//  Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
//  Home Page: www.liuxiao.org
//  E-mail: liuxiao@foxmail.com
//

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.ListIterator;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerPointer;

// Path for c_predict_api
import static org.bytedeco.javacpp.mxnet.*;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

public class ImageClassificationPredict {

    // Read file to buffer
    static class BufferFile implements Closeable {
        public String file_path_;
        public int length_;
        public BytePointer buffer_;

        public BufferFile(String file_path) {
            file_path_ = file_path;
            try {
                byte[] bytes = Files.readAllBytes(Paths.get(file_path));
                length_ = bytes.length;
                System.out.println(file_path + " ... " + length_ + " bytes");
                buffer_ = new BytePointer(bytes);
            } catch (IOException e) {
                System.err.println("Can't open the file: " + e + ". Please check " + file_path + ".");
                assert false;
            }
        }

        public int GetLength() {
            return length_;
        }
        public BytePointer GetBuffer() {
            return buffer_;
        }

        public void close() throws IOException {
            buffer_.deallocate();
            buffer_ = null;
        }
    }

    static void GetMeanFile(String image_file, FloatPointer image_data,
                            int channels, Size resize_size) {
        // Read all kinds of file into a BGR color 3 channels image
        Mat im_ori = imread(image_file, 1);

        if (im_ori.empty()) {
            System.err.println("Can't open the image. Please check " + image_file + ".");
            assert false;
        }

        Mat im = new Mat();

        resize(im_ori, im, resize_size);

        // Better to be read from a mean.nb file
        float mean = 117.0f;

        int rows = im.rows();
        int cols = im.cols();
        int size = rows * cols * 3;

        FloatBuffer ptr_image_r = image_data.position(0).asBuffer();
        FloatBuffer ptr_image_g = image_data.position(size / 3).asBuffer();
        FloatBuffer ptr_image_b = image_data.position(size / 3 * 2).asBuffer();

        for (int i = 0; i < rows; i++) {
            ByteBuffer data = im.ptr(i).capacity(3 * cols).asBuffer();

            for (int j = 0; j < cols; j++) {
                float b = (float)(data.get() & 0xFF) - mean;
                float g = (float)(data.get() & 0xFF) - mean;
                float r = (float)(data.get() & 0xFF) - mean;

                ptr_image_r.put(r);
                ptr_image_g.put(g);
                ptr_image_b.put(b);
            }
        }
    }

    // LoadSynsets
    // Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
    static List<String> LoadSynset(String filename) {
        try {
            List<String> output = Files.readAllLines(Paths.get(filename));
            ListIterator<String> it = output.listIterator();
            while (it.hasNext()) {
                String synsetLemma = it.next();
                it.set(synsetLemma.substring(synsetLemma.indexOf(" ") + 1));
            }
            return output;
        } catch (IOException e) {
            System.err.println("Error opening file " + filename + ": " + e);
            assert false;
        }
        return null;
    }

    static void PrintOutputResult(FloatPointer data, List<String> synset) {
        if (data.limit() != synset.size()) {
            System.err.println("Result data and synset size does not match!");
        }

        float best_accuracy = 0.0f;
        int best_idx = 0;

        for (int i = 0; i < data.limit(); i++) {
            System.out.printf("Accuracy[%d] = %.8f\n", i, data.get(i));

            if (data.get(i) > best_accuracy) {
                best_accuracy = data.get(i);
                best_idx = i;
            }
        }

        System.out.printf("Best Result: [%s] id = %d, accuracy = %.8f\n",
                synset.get(best_idx), best_idx, best_accuracy);
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("No test image here.");
            System.out.println("Usage: java ImageClassificationPredict apple.jpg");
            return;
        }

        String test_file = args[0];

        // Preload required by JavaCPP
        Loader.load(org.bytedeco.javacpp.mxnet.class);

        // Models path for your model, you have to modify it
        BufferFile json_data = new BufferFile("model/Inception_BN-symbol.json");
        BufferFile param_data = new BufferFile("model/Inception_BN-0039.params");

        // Parameters
        int dev_type = 1;  // 1: cpu, 2: gpu
        int dev_id = 0;  // arbitrary.
        int num_input_nodes = 1;  // 1 for feedforward
        String[] input_keys = {"data"};

        // Image size and channels
        int width = 224;
        int height = 224;
        int channels = 3;

        int[] input_shape_indptr = { 0, 4 };
        // ( trained_width, trained_height, channel, num)
        int[] input_shape_data = { 1, channels, width, height };
        PredictorHandle out = new PredictorHandle();  // alias for void *

        //-- Create Predictor
        MXPredCreate(json_data.GetBuffer(),
                     param_data.GetBuffer(),
                     param_data.GetLength(),
                     dev_type,
                     dev_id,
                     num_input_nodes,
                     new PointerPointer(input_keys),
                     new IntPointer(input_shape_indptr),
                     new IntPointer(input_shape_data),
                     out);

        // Just a big enough memory 1000x1000x3
        int image_size = width * height * channels;
        FloatPointer image_data = new FloatPointer(image_size);

        //-- Read Mean Data
        GetMeanFile(test_file, image_data.position(0), channels, new Size(width, height));

        //-- Set Input Image
        MXPredSetInput(out, "data", image_data.position(0), image_size);

        //-- Do Predict Forward
        MXPredForward(out);

        int output_index = 0;

        IntPointer shape = new IntPointer((IntPointer)null);
        IntPointer shape_len = new IntPointer(1);

        //-- Get Output Result
        MXPredGetOutputShape(out, output_index, shape, shape_len);

        int size = 1;
        for (int i = 0; i < shape_len.get(0); i++) size *= shape.get(i);

        FloatPointer data = new FloatPointer(size);

        MXPredGetOutput(out, output_index, data.position(0), size);

        // Release Predictor
        MXPredFree(out);

        // Synset path for your model, you have to modify it
        List<String> synset = LoadSynset("model/synset.txt");

        //-- Print Output Data
        PrintOutputResult(data.position(0), synset);
    }
}
```
