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
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.nio.file.Path;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * Torchio/torchvision-style image I/O and transforms via OpenCV (javacpp-opencv).
 *
 * <p>All read operations return tensors of shape {@code [C, H, W]}, dtype float32,
 * values in {@code [0, 255]} (uint8 source). Color images are returned as RGB.
 *
 * <p>All write operations accept tensors of shape {@code [C, H, W]}, dtype float32,
 * values in {@code [0, 255]}.
 *
 * <pre>{@code
 * // Read
 * Tensor img = OpenCVIO.readImage("/path/to/photo.jpg");   // [3, H, W] RGB
 * Tensor gray = OpenCVIO.readImageGray("/path/to/photo.jpg"); // [1, H, W]
 *
 * // Resize
 * Tensor small = OpenCVIO.resize(img, 224, 224);
 *
 * // Color conversion
 * Tensor bgr = OpenCVIO.rgbToBgr(img);
 *
 * // Write
 * OpenCVIO.writeImage("/path/to/out.png", img);
 *
 * // Encode to bytes
 * byte[] jpgBytes = OpenCVIO.encode(img, "jpg");
 * }</pre>
 */
public final class OpenCVIO {

    private OpenCVIO() {}

    // ---- Read ----

    /**
     * Read an image file as an RGB tensor.
     *
     * @param path image file path (JPEG, PNG, BMP, TIFF, …)
     * @return tensor {@code [3, H, W]}, dtype float32, values {@code [0, 255]}
     */
    public static Tensor readImage(String path) {
        Mat mat = imread(path, IMREAD_COLOR);
        if (mat == null || mat.total() == 0) {
            throw new OpenCVException("imread returned empty mat: " + path);
        }
        try {
            // IMREAD_COLOR → BGR; MatToTensor.fromMat does BGR→RGB for 3-channel mats.
            // Do NOT cvtColor here — that would double-swap channels.
            return ensureCHW(MatToTensor.fromMat(mat));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** @see #readImage(String) */
    public static Tensor readImage(Path path) {
        return readImage(path.toString());
    }

    /**
     * Read an image as grayscale (single channel).
     *
     * @param path image file path
     * @return tensor {@code [1, H, W]}, dtype float32, values {@code [0, 255]}
     */
    public static Tensor readImageGray(String path) {
        Mat mat = imread(path, IMREAD_GRAYSCALE);
        if (mat == null || mat.total() == 0) {
            throw new OpenCVException("imread returned empty mat: " + path);
        }
        try {
            // MatToTensor grayscale is [H,W]; expose torch-style [1,H,W]
            return ensureCHW(MatToTensor.fromMat(mat));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** @see #readImageGray(String) */
    public static Tensor readImageGray(Path path) {
        return readImageGray(path.toString());
    }

    /**
     * Read an image with specific channel count.
     *
     * @param path image file path
     * @param channels 1 (grayscale) or 3 (color); forces conversion
     * @return tensor {@code [channels, H, W]}, dtype float32
     */
    public static Tensor readImage(String path, int channels) {
        if (channels == 1) return readImageGray(path);
        if (channels == 3) return readImage(path);
        Mat mat = imread(path, IMREAD_ANYCOLOR);
        if (mat == null || mat.total() == 0) {
            throw new OpenCVException("imread returned empty mat: " + path);
        }
        try {
            // Keep OpenCV native layout (BGR for color, gray for 1-ch).
            // MatToTensor.fromMat handles BGR→RGB for 3-channel mats.
            if (mat.channels() == 1 && channels > 1) {
                Mat converted = new Mat();
                cvtColor(mat, converted, COLOR_GRAY2BGR); // fromMat will BGR→RGB
                return ensureCHW(MatToTensor.fromMat(converted));
            } else if (mat.channels() == 3 && channels == 1) {
                Mat gray = new Mat();
                cvtColor(mat, gray, COLOR_BGR2GRAY);
                return ensureCHW(MatToTensor.fromMat(gray));
            }
            return ensureCHW(MatToTensor.fromMat(mat));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Decode from bytes ----

    /**
     * Decode an image from JPEG/PNG bytes in memory.
     *
     * @param encoded image bytes
     * @return tensor {@code [3, H, W]}, dtype float32, values {@code [0, 255]}
     */
    public static Tensor decodeImage(byte[] encoded) {
        if (encoded == null || encoded.length == 0) {
            throw new IllegalArgumentException("encoded bytes are empty");
        }
        Mat buf = new Mat(1, encoded.length, CV_8UC1);
        buf.data().put(encoded);
        // IMREAD_COLOR forces 3-channel BGR so MatToTensor.fromMat can BGR→RGB consistently
        Mat decoded = imdecode(buf, IMREAD_COLOR);
        if (decoded == null || decoded.total() == 0) {
            throw new OpenCVException("imdecode returned empty mat");
        }
        try {
            // Do NOT cvtColor — fromMat already does BGR→RGB for 3-channel mats.
            return ensureCHW(MatToTensor.fromMat(decoded));
        } finally {
            if (decoded != null) decoded.close();
            buf.close();
        }
    }

    // ---- Write ----

    /**
     * Write a tensor as an image file (PNG by default).
     *
     * <p>Accepts {@code [3, H, W]} (RGB), {@code [1, H, W]} (grayscale).
     * Values must be in {@code [0, 255]}.
     *
     * @param path output file path; extension determines format (png, jpg, bmp, tiff, …)
     * @param tensor tensor {@code [C, H, W]} RGB or grayscale
     */
    public static void writeImage(String path, Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            // toMat writes RGB→BGR internally
            boolean ok = imwrite(path, mat);
            if (!ok) {
                throw new OpenCVException("imwrite failed for: " + path);
            }
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** @see #writeImage(String, Tensor) */
    public static void writeImage(Path path, Tensor tensor) {
        writeImage(path.toString(), tensor);
    }

    // ---- Encode to bytes ----

    /**
     * Encode a tensor to image bytes (JPEG or PNG).
     *
     * @param tensor tensor {@code [C, H, W]}, dtype float32, values {@code [0, 255]}
     * @param format "jpg"/"jpeg" or "png" (case-insensitive)
     * @return encoded image bytes
     */
    public static byte[] encode(Tensor tensor, String format) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            String ext = format == null ? ".png" : format.toLowerCase();
            if (ext.equals("jpeg") || ext.equals("jpg") || ext.equals(".jpeg") || ext.equals(".jpg")) {
                ext = ".jpg";
            } else if (ext.equals("png") || ext.equals(".png")) {
                ext = ".png";
            } else if (!ext.startsWith(".")) {
                ext = "." + ext;
            }

            // JavaCPP imencode writes into a growable BytePointer (std::vector<uchar>).
            BytePointer buf = new BytePointer();
            boolean ok = imencode(ext, mat, buf);
            if (!ok) {
                throw new OpenCVException("imencode failed for format: " + format);
            }
            long lim = buf.limit();
            if (lim <= 0) lim = buf.capacity();
            if (lim <= 0) {
                throw new OpenCVException("imencode produced empty buffer for format: " + format);
            }
            byte[] out = new byte[(int) lim];
            buf.position(0).get(out);
            buf.deallocate();
            return out;
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Resize ----

    /**
     * Resize an image tensor to the given height and width.
     *
     * @param tensor {@code [C, H, W]} image tensor
     * @param height output height in pixels
     * @param width  output width in pixels
     * @return resized tensor {@code [C, height, width]}
     */
    public static Tensor resize(Tensor tensor, int height, int width) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat resized = new Mat();
            org.bytedeco.opencv.global.opencv_imgproc.resize(mat, resized, new Size(width, height), 0, 0, INTER_LINEAR);
            return ensureCHW(MatToTensor.fromMat(resized));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /**
     * Resize by a scaling factor.
     *
     * @param tensor {@code [C, H, W]} image tensor
     * @param fx horizontal scale factor
     * @param fy vertical scale factor
     * @return resized tensor
     */
    public static Tensor resize(Tensor tensor, double fx, double fy) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            int dstW = (int) Math.round(mat.cols() * fx);
            int dstH = (int) Math.round(mat.rows() * fy);
            Mat resized = new Mat();
            org.bytedeco.opencv.global.opencv_imgproc.resize(mat, resized, new Size(dstW, dstH), 0, 0, INTER_LINEAR);
            return ensureCHW(MatToTensor.fromMat(resized));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Color conversion ----

    /**
     * Convert RGB tensor to BGR (channel swap). Useful when handing tensors to pure-OpenCV code.
     */
    public static Tensor rgbToBgr(Tensor tensor) {
        return swapRB(tensor);
    }

    /**
     * Convert BGR tensor to RGB (channel swap).
     */
    public static Tensor bgrToRgb(Tensor tensor) {
        return swapRB(tensor);
    }

    /**
     * Convert a color tensor to grayscale.
     *
     * @param tensor {@code [3, H, W]} RGB tensor
     * @return {@code [1, H, W]} grayscale tensor
     */
    public static Tensor toGrayscale(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat gray = new Mat();
            // toMat emits BGR for 3-channel tensors
            if (mat.channels() == 1) {
                return ensureCHW(MatToTensor.fromMat(mat));
            }
            cvtColor(mat, gray, COLOR_BGR2GRAY);
            return ensureCHW(MatToTensor.fromMat(gray));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Crop ----

    /**
     * Crop a tensor to a rectangular region.
     *
     * @param tensor {@code [C, H, W]}
     * @param y      top-left y coordinate
     * @param x      top-left x coordinate
     * @param height crop height
     * @param width  crop width
     * @return cropped tensor {@code [C, height, width]}
     */
    public static Tensor crop(Tensor tensor, int y, int x, int height, int width) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Rect roi = new Rect(x, y, width, height);
            Mat cropped = new Mat(mat, roi);
            return ensureCHW(MatToTensor.fromMat(cropped));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Horizontal flip ----

    /**
     * Horizontally flip an image tensor.
     *
     * @param tensor {@code [C, H, W]}
     * @return flipped tensor
     */
    public static Tensor hflip(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat flipped = new Mat();
            flip(mat, flipped, 1); // flipCode=1 → horizontal
            return ensureCHW(MatToTensor.fromMat(flipped));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Rotate (90°) ----

    /**
     * Rotate an image tensor by 90° clockwise.
     *
     * @param tensor {@code [C, H, W]}
     * @return rotated tensor
     */
    public static Tensor rotate90(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat rotated = new Mat();
            rotate(mat, rotated, ROTATE_90_CLOCKWISE);
            return ensureCHW(MatToTensor.fromMat(rotated));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Normalize (torchvision-style) ----

    /**
     * Normalize a tensor with mean and std (torchvision.functional.normalize).
     *
     * @param tensor {@code [C, H, W]} tensor, values typically {@code [0, 1]}
     * @param mean   per-channel mean (length = C)
     * @param std    per-channel std  (length = C)
     * @return normalized tensor {@code [C, H, W]}
     */
    public static Tensor normalize(Tensor tensor, float[] mean, float[] std) {
        long[] shape = sizes(tensor);
        if (shape.length != 3) throw new IllegalArgumentException("expected [C,H,W]");
        int c = (int) shape[0];
        if (mean.length != c || std.length != c) {
            throw new IllegalArgumentException("mean/std length must match channels");
        }

        Tensor t = tensor.contiguous().clone();
        Tensor tCpu = t.cpu();

        for (int ch = 0; ch < c; ch++) {
            FloatPointer fp = tCpu.data_ptr_float();
            long chOffset = (long) ch * (int) shape[1] * (int) shape[2];
            for (int i = 0; i < (int) shape[1] * (int) shape[2]; i++) {
                long idx = chOffset + i;
                float v = fp.get(idx);
                fp.put(idx, (v - mean[ch]) / std[ch]);
            }
        }
        return tCpu;
    }

    // ---- helpers ----

    /** Ensure channel-first layout: {@code [H,W]} → {@code [1,H,W]}; pass through {@code [C,H,W]}. */
    private static Tensor ensureCHW(Tensor t) {
        long[] s = sizes(t);
        if (s.length == 2) {
            return t.reshape(1, s[0], s[1]);
        }
        return t;
    }

    /** Swap R↔B on a 3-channel CHW tensor; identity for other ranks/channels. */
    private static Tensor swapRB(Tensor tensor) {
        long[] s = sizes(tensor);
        if (s.length != 3 || s[0] != 3) return tensor;
        // out[0]=in[2], out[1]=in[1], out[2]=in[0] via clone + channel copy
        Tensor out = tensor.contiguous().clone();
        Tensor src = tensor.contiguous().cpu();
        Tensor dst = out.cpu();
        FloatPointer sp = src.data_ptr_float();
        FloatPointer dp = dst.data_ptr_float();
        int h = (int) s[1], w = (int) s[2];
        long plane = (long) h * w;
        for (long i = 0; i < plane; i++) {
            float r = sp.get(i);           // ch0
            float g = sp.get(plane + i);   // ch1
            float b = sp.get(2 * plane + i); // ch2
            dp.put(i, b);
            dp.put(plane + i, g);
            dp.put(2 * plane + i, r);
        }
        return out;
    }

    private static long[] sizes(Tensor t) {
        long ndim = t.dim();
        long[] out = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) out[i] = t.size(i);
        return out;
    }
}
