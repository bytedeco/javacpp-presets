/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.opencv;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;

/**
 * OpenCV {@code cv::Mat} ↔ PyTorch {@code Tensor} conversion utilities.
 *
 * <p>Supports:
 * <ul>
 *   <li>Mat (HxW) → Tensor [H, W]    — grayscale</li>
 *   <li>Mat (HxWx3) → Tensor [3, H, W] — RGB BGR conversion</li>
 *   <li>Mat (HxWxC) → Tensor [C, H, W] — multi-channel</li>
 *   <li>Tensor → Mat</li>
 * </ul>
 *
 * <p>All conversions copy data (no sharing).
 */
public final class MatToTensor {

    private MatToTensor() {}

    // ---- Mat → Tensor ----

    /**
     * Convert an OpenCV Mat to a PyTorch tensor.
     *
     * <p>For image Mats (2-D HxW or 3-D HxWxC):
     * <ul>
     *   <li>2-D {@code [H, W]} grayscale → output shape {@code [H, W]}</li>
     *   <li>3-D {@code [H, W, 3]} BGR → output shape {@code [3, H, W]} RGB</li>
     *   <li>3-D {@code [H, W, C]} → output shape {@code [C, H, W]} (de-interleave)</li>
     * </ul>
     *
     * <p>Output dtype: {@code float32}. Values are in range {@code [0, 255]}.
     *
     * @param mat OpenCV Mat (must be contiguous)
     * @return PyTorch tensor
     */
    public static Tensor fromMat(org.bytedeco.opencv.opencv_core.Mat mat) {
        if (mat == null || mat.total() == 0) {
            throw new IllegalArgumentException("mat is null or empty");
        }

        int rows = mat.rows();
        int cols = mat.cols();
        int type = mat.type();

        // Determine channels from OpenCV type: CV_8UCn → n channels
        int ch = CV_MAT_CN(type);
        int depth = CV_MAT_DEPTH(type);

        if (depth != CV_8U && depth != CV_32F) {
            // Convert other depths to float32 or uint8
            throw new IllegalArgumentException("Unsupported Mat depth: " + depth
                    + " (only CV_8U and CV_32F supported)");
        }

        if (mat.dims() != 2) {
            throw new IllegalArgumentException("Only 2-D Mats supported, got dims=" + mat.dims());
        }

        if (ch == 1 && depth == CV_8U) {
            return fromGrayMat8u(mat, rows, cols);
        } else if (ch == 3 && depth == CV_8U) {
            return fromBgrMat8u(mat, rows, cols);
        } else if (depth == CV_8U) {
            return fromMultiChannelMat8u(mat, rows, cols, ch);
        } else if (depth == CV_32F) {
            return fromFloatMat(mat, rows, cols, ch);
        }

        throw new IllegalArgumentException("Unsupported Mat type: " + type);
    }

    private static Tensor fromGrayMat8u(org.bytedeco.opencv.opencv_core.Mat mat, int rows, int cols) {
        Tensor t = torch.empty(new long[]{rows, cols}, new TensorOptions(ScalarType.Float), null);
        BytePointer srcData = mat.data();
        FloatPointer dst = t.data_ptr_float();
        int lineBytes = cols; // contiguous grayscale
        for (int y = 0; y < rows; y++) {
            int rowBase = y * lineBytes;
            for (int x = 0; x < cols; x++) {
                dst.put((long) y * cols + x, (srcData.get(rowBase + x)) & 0xFF);
            }
        }
        return t;
    }

    private static Tensor fromBgrMat8u(org.bytedeco.opencv.opencv_core.Mat mat, int rows, int cols) {
        // BGR [H, W, 3] → RGB [3, H, W]
        Tensor t = torch.empty(new long[]{3, rows, cols}, new TensorOptions(ScalarType.Float), null);
        BytePointer srcData = mat.data();
        FloatPointer dst = t.data_ptr_float();
        int srcStep = (int) mat.step1(); // bytes per row

        for (int y = 0; y < rows; y++) {
            int rowBase = y * srcStep;
            for (int x = 0; x < cols; x++) {
                int pixelBase = rowBase + x * 3;
                byte b = srcData.get(pixelBase + 0);
                byte g = srcData.get(pixelBase + 1);
                byte r = srcData.get(pixelBase + 2);
                // RGB channel-first
                dst.put((long) 0 * rows * cols + y * cols + x, (r) & 0xFF);
                dst.put((long) 1 * rows * cols + y * cols + x, (g) & 0xFF);
                dst.put((long) 2 * rows * cols + y * cols + x, (b) & 0xFF);
            }
        }
        return t;
    }

    private static Tensor fromMultiChannelMat8u(org.bytedeco.opencv.opencv_core.Mat mat,
                                                int rows, int cols, int ch) {
        Tensor t = torch.empty(new long[]{ch, rows, cols}, new TensorOptions(ScalarType.Float), null);
        BytePointer srcData = mat.data();
        FloatPointer dst = t.data_ptr_float();
        int srcStep = (int) mat.step1();

        for (int y = 0; y < rows; y++) {
            int rowBase = y * srcStep;
            for (int x = 0; x < cols; x++) {
                int pixelBase = rowBase + x * ch;
                for (int c = 0; c < ch; c++) {
                    long dstIdx = (long) c * rows * cols + y * cols + x;
                    dst.put(dstIdx, (srcData.get(pixelBase + c)) & 0xFF);
                }
            }
        }
        return t;
    }

    private static Tensor fromFloatMat(org.bytedeco.opencv.opencv_core.Mat mat,
                                       int rows, int cols, int ch) {
        Tensor t = torch.empty(new long[]{ch, rows, cols}, new TensorOptions(ScalarType.Float), null);
        FloatPointer srcData = new FloatPointer(mat.data());
        FloatPointer dst = t.data_ptr_float();
        int srcStepFloats = (int) (mat.step1() / 4); // step in floats

        if (ch == 1) {
            // [H, W] float
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    dst.put((long) y * cols + x, srcData.get((long) y * srcStepFloats + x));
                }
            }
        } else {
            for (int y = 0; y < rows; y++) {
                for (int x = 0; x < cols; x++) {
                    int pixelBase = y * srcStepFloats + x * ch;
                    for (int c = 0; c < ch; c++) {
                        long dstIdx = (long) c * rows * cols + y * cols + x;
                        dst.put(dstIdx, srcData.get(pixelBase + c));
                    }
                }
            }
        }
        return t;
    }

    // ---- Tensor → Mat ----

    /**
     * Convert a PyTorch tensor to an OpenCV Mat.
     *
     * <p>Accepted tensor shapes:
     * <ul>
     *   <li>{@code [H, W]}   → grayscale {@code [H, W]} uint8</li>
     *   <li>{@code [3, H, W]} → BGR {@code [H, W, 3]} uint8</li>
     *   <li>{@code [C, H, W]} → multi-channel {@code [H, W, C]} uint8</li>
     * </ul>
     *
     * <p>Values are assumed to be in {@code [0, 255]} float range.
     *
     * @param t PyTorch tensor (must be on CPU)
     * @return OpenCV Mat (managed by caller; clone to retain)
     */
    public static org.bytedeco.opencv.opencv_core.Mat toMat(Tensor t) {
        if (t == null) throw new IllegalArgumentException("tensor is null");

        long[] shape = sizes(t);
        if (shape.length == 2) {
            // [H, W] → grayscale
            int h = (int) shape[0];
            int w = (int) shape[1];
            org.bytedeco.opencv.opencv_core.Mat mat = new org.bytedeco.opencv.opencv_core.Mat(h, w, CV_8UC1);
            writeGrayTensor(t, mat, h, w);
            return mat;
        } else if (shape.length == 3) {
            // [C, H, W] → image
            int c = (int) shape[0];
            int h = (int) shape[1];
            int w = (int) shape[2];
            int matType = c == 3 ? CV_8UC3 : CV_8UC(c);
            org.bytedeco.opencv.opencv_core.Mat mat = new org.bytedeco.opencv.opencv_core.Mat(h, w, matType);
            writeMultiChannelTensor(t, mat, c, h, w);
            return mat;
        }
        throw new IllegalArgumentException("Unsupported tensor shape: " + java.util.Arrays.toString(shape));
    }

    private static void writeGrayTensor(Tensor t, org.bytedeco.opencv.opencv_core.Mat mat, int h, int w) {
        Tensor cpu = t.contiguous().cpu();
        FloatPointer fp = cpu.data_ptr_float();
        BytePointer dst = mat.data();
        for (int i = 0; i < h * w; i++) {
            float v = fp.get(i);
            dst.put(i, (byte) Math.max(0, Math.min(255, Math.round(v))));
        }
    }

    private static void writeMultiChannelTensor(Tensor t,
                                               org.bytedeco.opencv.opencv_core.Mat mat,
                                               int c, int h, int w) {
        Tensor cpu = t.contiguous().cpu();
        FloatPointer fp = cpu.data_ptr_float();
        BytePointer dst = mat.data();
        int dstStep = (int) mat.step1();
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int dstBase = y * dstStep + x * c;
                for (int ch = 0; ch < c; ch++) {
                    long srcIdx = (long) ch * h * w + y * w + x;
                    float v = fp.get(srcIdx);
                    dst.put(dstBase + ch, (byte) Math.max(0, Math.min(255, Math.round(v))));
                }
            }
        }
    }

    // ---- helpers ----

    private static long[] sizes(Tensor t) {
        long ndim = t.dim();
        long[] out = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) out[i] = t.size(i);
        return out;
    }

    // ---- color space helpers ----

    /**
     * Convert BGR OpenCV Mat to RGB tensor.
     * Convenience that reads BGR and outputs [3, H, W] RGB.
     */
    public static Tensor bgrMatToRgbTensor(org.bytedeco.opencv.opencv_core.Mat bgrMat) {
        return fromMat(bgrMat); // fromMat does BGR→RGB internally
    }

    /**
     * Write an RGB tensor to a BGR OpenCV Mat.
     * Convenience that writes [3, H, W] RGB to BGR Mat.
     */
    public static org.bytedeco.opencv.opencv_core.Mat rgbTensorToBgrMat(Tensor rgb) {
        return toMat(rgb); // toMat writes RGB→BGR internally
    }
}
