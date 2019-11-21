/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Xiao Liu, pertusa, caprice-j
 * \file image_classification-predict.cpp
 * \brief C++ predict example of mxnet
 *
 * This is a simple predictor which shows how to use c api for image classification. It uses
 * opencv for image reading.
 *
 * Created by liuxiao on 12/9/15.
 * Thanks to : pertusa, caprice-j, sofiawu, tqchen, piiswrong
 * Home Page: www.liuxiao.org
 * E-mail: liuxiao@foxmail.com
*/

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

import org.apache.mxnet.javaapi.*;
import org.bytedeco.javacpp.*;

// Path for c_predict_api
import org.bytedeco.mxnet.*;
import static org.bytedeco.mxnet.global.mxnet.*;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ImageClassificationPredict {

    static final float DEFAULT_MEAN = 117.0f;

    // Read file to buffer
    static class BufferFile implements Closeable {
        public String file_path_;
        public int length_ = 0;
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

    static void GetImageFile(String image_file, FloatPointer image_data,
                             int channels, Size resize_size, FloatPointer mean_data) {
        // Read all kinds of file into a BGR color 3 channels image
        Mat im_ori = imread(image_file, IMREAD_COLOR);

        if (im_ori.empty()) {
            System.err.println("Can't open the image. Please check " + image_file + ".");
            assert false;
        }

        Mat im = new Mat();

        resize(im_ori, im, resize_size);

        int rows = im.rows();
        int cols = im.cols();
        int size = rows * cols * channels;

        FloatBuffer ptr_image_r = image_data.position(0).asBuffer();
        FloatBuffer ptr_image_g = image_data.position(size / 3).asBuffer();
        FloatBuffer ptr_image_b = image_data.position(size / 3 * 2).asBuffer();

        FloatBuffer ptr_mean_r, ptr_mean_g, ptr_mean_b;
        ptr_mean_r = ptr_mean_g = ptr_mean_b = null;
        if (mean_data != null && !mean_data.isNull()) {
            ptr_mean_r = mean_data.position(0).asBuffer();
            ptr_mean_g = mean_data.position(size / 3).asBuffer();
            ptr_mean_b = mean_data.position(size / 3 * 2).asBuffer();
        }

        float mean_b, mean_g, mean_r;
        mean_b = mean_g = mean_r = DEFAULT_MEAN;

        for (int i = 0; i < rows; i++) {
            ByteBuffer data = im.ptr(i).capacity(3 * cols).asBuffer();

            for (int j = 0; j < cols; j++) {
                if (mean_data != null && !mean_data.isNull()) {
                    mean_r = ptr_mean_r.get();
                    if (channels > 1) {
                        mean_g = ptr_mean_g.get();
                        mean_b = ptr_mean_b.get();
                    }
                }
                if (channels > 1) {
                    ptr_image_b.put((float)(data.get() & 0xFF) - mean_b);
                    ptr_image_g.put((float)(data.get() & 0xFF) - mean_g);
                }

                ptr_image_r.put((float)(data.get() & 0xFF) - mean_r);
            }
        }
    }

    // LoadSynsets
    // Code from : https://github.com/pertusa/mxnet_predict_cc/blob/master/mxnet_predict.cc
    static List<String> LoadSynset(String synset_file) {
        try {
            List<String> output = Files.readAllLines(Paths.get(synset_file));
            ListIterator<String> it = output.listIterator();
            while (it.hasNext()) {
                String synsetLemma = it.next();
                it.set(synsetLemma.substring(synsetLemma.indexOf(" ") + 1));
            }
            return output;
        } catch (IOException e) {
            System.err.println("Error opening synset file " + synset_file + ": " + e);
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

        System.out.printf("Best Result: %s (id=%d, accuracy=%.8f)\n",
                synset.get(best_idx).trim(), best_idx, best_accuracy);
    }

    static void predict(PredictorHandle pred_hnd, FloatPointer image_data,
                        NDListHandle nd_hnd, String synset_file, int n) {
        int image_size = (int)image_data.limit();
        // Set Input Image
        MXPredSetInput(pred_hnd, "data", image_data.position(0), image_size);

        // Do Predict Forward
        MXPredForward(pred_hnd);

        int output_index = 0;

        IntPointer shape = new IntPointer((IntPointer)null);
        IntPointer shape_len = new IntPointer(1);

        // Get Output Result
        MXPredGetOutputShape(pred_hnd, output_index, shape, shape_len);

        int size = 1;
        for (int i = 0; i < shape_len.get(0); i++) { size *= shape.get(i); }

        FloatPointer data = new FloatPointer(size);

        MXPredGetOutput(pred_hnd, output_index, data.position(0), size);

        // Release NDList
        if (nd_hnd != null) {
            MXNDListFree(nd_hnd);
        }

        // Release Predictor
        MXPredFree(pred_hnd);

        // Synset path for your model, you have to modify it
        List<String> synset = LoadSynset(synset_file);

        // Print Output Data
        PrintOutputResult(data.position(0), synset);
    }

    public static void main(String[] args) throws Exception {
        // Preload required by JavaCPP
        Loader.load(org.bytedeco.mxnet.global.mxnet.class);

        if (args.length < 1) {
            System.out.println("No test image here.");
            System.out.println("Usage: java ImageClassificationPredict apple.jpg [num_threads]");
            return;
        }

        final String test_file = args[0];
        int num_threads = 1;
        if (args.length == 2) {
            num_threads = Integer.parseInt(args[1]);
        }

        // Models path for your model, you have to modify it
        final String json_file = "data/model/Inception/Inception-BN-symbol.json";
        final String param_file = "data/model/Inception/Inception-BN-0126.params";
        final String synset_file = "data/model/Inception/synset.txt";
        final String nd_file = "data/model/Inception/mean_224.nd";

        BufferFile json_data = new BufferFile(json_file);
        BufferFile param_data = new BufferFile(param_file);

        // Parameters
        int dev_type = 1;  // 1: cpu, 2: gpu
        int dev_id = 0;  // arbitrary.
        int num_input_nodes = 1;  // 1 for feedforward
        String[] input_keys = { "data" };

        // Image size and channels
        int width = 224;
        int height = 224;
        int channels = 3;

        int[] input_shape_indptr = { 0, 4 };
        int[] input_shape_data = { 1, channels, height, width };


        if (json_data.GetLength() == 0 || param_data.GetLength() == 0) {
            System.exit(1 /* EXIT_FAILURE */);
        }


        final int image_size = width * height * channels;

        // Read Mean Data
        final FloatPointer nd_data = new FloatPointer((Pointer)null);
        final NDListHandle nd_hnd = new NDListHandle((Pointer)null);
        final BufferFile nd_buf = new BufferFile(nd_file);

        if (nd_buf.GetLength() > 0) {
            int nd_index = 0;
            IntPointer nd_len = new IntPointer(1);
            IntPointer nd_shape = new IntPointer((Pointer)null);
            BytePointer nd_key = new BytePointer((Pointer)null);
            IntPointer nd_ndim = new IntPointer(new int[]{0});

            MXNDListCreate(nd_buf.GetBuffer(),
                           nd_buf.GetLength(),
                           nd_hnd, nd_len);

            MXNDListGet(nd_hnd, nd_index, nd_key, nd_data, nd_shape, nd_ndim);
        }

        // Read Image Data
        final FloatPointer image_data = new FloatPointer(image_size);

        GetImageFile(test_file, image_data, channels, new Size(width, height), nd_data);


        if (num_threads == 1) {
            // Create Predictor
            final PointerPointer<PredictorHandle> pred_hnd = new PointerPointer<PredictorHandle>(1);
            MXPredCreate(json_data.GetBuffer(),
                         param_data.GetBuffer(),
                         param_data.GetLength(),
                         dev_type,
                         dev_id,
                         num_input_nodes,
                         new PointerPointer(input_keys),
                         new IntPointer(input_shape_indptr),
                         new IntPointer(input_shape_data),
                         pred_hnd);

            assert !pred_hnd.get().isNull();

            predict(pred_hnd.get(PredictorHandle.class), image_data, nd_hnd, synset_file, 0);
        } else {
            // Create Predictor
            final PointerPointer<PredictorHandle> pred_hnds = new PointerPointer<PredictorHandle>(num_threads);
            MXPredCreateMultiThread(json_data.GetBuffer(),
                                    param_data.GetBuffer(),
                                    param_data.GetLength(),
                                    dev_type,
                                    dev_id,
                                    num_input_nodes,
                                    new PointerPointer(input_keys),
                                    new IntPointer(input_shape_indptr),
                                    new IntPointer(input_shape_data),
                                    num_threads,
                                    pred_hnds);
            for (int i = 0; i < num_threads; i++) {
                assert !pred_hnds.get(i).isNull();
            }

            List<Thread> threads = new ArrayList<Thread>();
            for (int i = 0; i < num_threads; i++) {
                final int n = i;
                threads.add(new Thread() { public void run() {
                    predict(pred_hnds.get(PredictorHandle.class, n), image_data, nd_hnd, synset_file, n);
                }});
                threads.get(i).start();
            }
            for (int i = 0; i < num_threads; i++) {
                threads.get(i).join();
            }
        }
        System.out.println("run successfully");

        System.exit(0 /* EXIT_SUCCESS */);
    }
}
