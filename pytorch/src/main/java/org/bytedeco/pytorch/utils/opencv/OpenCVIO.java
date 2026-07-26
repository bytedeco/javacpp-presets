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
import org.bytedeco.pytorch.data.transforms.*;

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
            // IMREAD_COLOR → BGR; convert to RGB
            Mat bgr = mat;
            if (bgr.channels() == 3) {
                Mat rgb = new Mat();
                cvtColor(bgr, rgb, COLOR_BGR2RGB);
                return MatToTensor.fromMat(rgb);
            }
            return MatToTensor.fromMat(bgr);
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
            return MatToTensor.fromMat(mat);
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
            Mat converted = new Mat();
            if (mat.channels() == 1 && channels > 1) {
                cvtColor(mat, converted, COLOR_GRAY2RGB);
            } else if (mat.channels() == 3 && channels == 1) {
                cvtColor(mat, converted, COLOR_BGR2GRAY);
                converted = converted.reshape(1); // ensure [H,W] → [1,H,W]
            } else if (mat.channels() == 3) {
                cvtColor(mat, converted, COLOR_BGR2RGB);
            } else {
                converted = mat.clone();
            }
            return MatToTensor.fromMat(converted);
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
        Mat decoded = imdecode(buf, IMREAD_UNCHANGED);
        if (decoded == null || decoded.total() == 0) {
            throw new OpenCVException("imdecode returned empty mat");
        }
        try {
            Mat rgb = new Mat();
            if (decoded.channels() == 3) {
                cvtColor(decoded, rgb, COLOR_BGR2RGB);
            } else {
                rgb = decoded.clone();
            }
            return MatToTensor.fromMat(rgb);
        } finally {
            if (decoded != null) decoded.close();
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
            String ext = format.toLowerCase();
            if (ext.equals("jpeg") || ext.equals("jpg")) ext = ".jpg";
            else if (ext.equals("png")) ext = ".png";
            else ext = "." + ext;

            byte[][] buf = new byte[1][];
            boolean ok = imencode(ext, mat, buf[0]);
            if (!ok) {
                throw new OpenCVException("imencode failed for format: " + format);
            }
            return buf[0];
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
            return MatToTensor.fromMat(resized);
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
            return MatToTensor.fromMat(resized);
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Color conversion ----

    /**
     * Convert RGB tensor to BGR OpenCV Mat (and back via {@link #writeImage}).
     * This is a no-op at the tensor level since internal Mat→Tensor always produces RGB.
     */
    public static Tensor rgbToBgr(Tensor tensor) {
        // OpenCV stores BGR; our MatToTensor.fromMat already BGR→RGB on read
        // and MatToTensor.toMat already RGB→BGR on write.
        // So this is a semantic identity for the Java side.
        return tensor;
    }

    /**
     * Convert BGR tensor to RGB.
     */
    public static Tensor bgrToRgb(Tensor tensor) {
        return tensor; // same as rgbToBgr
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
            cvtColor(mat, gray, COLOR_BGR2GRAY);
            Mat reshaped = gray.reshape(1); // [H, W] → [1, H, W]
            return MatToTensor.fromMat(reshaped);
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
            return MatToTensor.fromMat(cropped);
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
            return MatToTensor.fromMat(flipped);
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
            return MatToTensor.fromMat(rotated);
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

    private static long[] sizes(Tensor t) {
        long ndim = t.dim();
        long[] out = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) out[i] = t.size(i);
        return out;
    }
}
