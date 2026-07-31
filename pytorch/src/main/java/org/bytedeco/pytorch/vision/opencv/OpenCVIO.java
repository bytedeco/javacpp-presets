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
package org.bytedeco.pytorch.vision.opencv;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_video.*;

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
            // Fully-qualify: local rotate(Tensor,double) shadows static-imported imgproc.rotate
            org.bytedeco.opencv.global.opencv_core.rotate(mat, rotated, ROTATE_90_CLOCKWISE);
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

    // ---- Vertical flip / extra rotates ----

    /** Vertically flip {@code [C,H,W]}. */
    public static Tensor vflip(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat flipped = new Mat();
            flip(mat, flipped, 0); // flipCode=0 → vertical
            return ensureCHW(MatToTensor.fromMat(flipped));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** Rotate 180°. */
    public static Tensor rotate180(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat rotated = new Mat();
            org.bytedeco.opencv.global.opencv_core.rotate(mat, rotated, ROTATE_180);
            return ensureCHW(MatToTensor.fromMat(rotated));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** Rotate 90° counter-clockwise (270° clockwise). */
    public static Tensor rotate90ccw(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat rotated = new Mat();
            org.bytedeco.opencv.global.opencv_core.rotate(mat, rotated, ROTATE_90_COUNTERCLOCKWISE);
            return ensureCHW(MatToTensor.fromMat(rotated));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /**
     * Arbitrary-angle rotation (degrees, clockwise positive via getRotationMatrix2D).
     * Output canvas same size; corners may be cropped.
     */
    public static Tensor rotate(Tensor tensor, double angleDeg) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            org.bytedeco.opencv.opencv_core.Point2f center =
                    new org.bytedeco.opencv.opencv_core.Point2f(mat.cols() / 2.0f, mat.rows() / 2.0f);
            Mat M = getRotationMatrix2D(center, -angleDeg, 1.0); // OpenCV: positive = CCW
            Mat out = new Mat();
            warpAffine(mat, out, M, new Size(mat.cols(), mat.rows()));
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Blur / filter ----

    /** Gaussian blur; {@code ksize} must be odd positive. */
    public static Tensor gaussianBlur(Tensor tensor, int ksize, double sigma) {
        int k = normalizeOddKernel(ksize, 3);
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat out = new Mat();
            GaussianBlur(mat, out, new Size(k, k), sigma <= 0 ? 0 : sigma);
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    public static Tensor gaussianBlur(Tensor tensor, int ksize) {
        return gaussianBlur(tensor, ksize, 0);
    }

    /** Median blur; {@code ksize} odd. */
    public static Tensor medianBlur(Tensor tensor, int ksize) {
        int k = normalizeOddKernel(ksize, 3);
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat out = new Mat();
            org.bytedeco.opencv.global.opencv_imgproc.medianBlur(mat, out, k);
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** Bilateral filter (edge-preserving denoise). */
    public static Tensor bilateralFilter(Tensor tensor, int d, double sigmaColor, double sigmaSpace) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat out = new Mat();
            org.bytedeco.opencv.global.opencv_imgproc.bilateralFilter(
                    mat, out, d <= 0 ? 9 : d, sigmaColor, sigmaSpace);
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Edges ----

    /**
     * Canny edge detector.
     * @return {@code [1,H,W]} edge map in {@code [0,255]}
     */
    public static Tensor canny(Tensor tensor, double threshold1, double threshold2) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat gray = toGrayMat(mat);
            Mat edges = new Mat();
            Canny(gray, edges, threshold1, threshold2);
            if (gray != mat) gray.close();
            return ensureCHW(MatToTensor.fromMat(edges));
        } finally {
            if (mat != null) mat.close();
        }
    }

    public static Tensor canny(Tensor tensor) {
        return canny(tensor, 50, 150);
    }

    /**
     * Sobel magnitude edge map.
     * @return {@code [1,H,W]} float32 approx magnitude scaled to {@code [0,255]}
     */
    public static Tensor sobel(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat gray = toGrayMat(mat);
            Mat gx = new Mat();
            Mat gy = new Mat();
            Sobel(gray, gx, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
            Sobel(gray, gy, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
            Mat mag = new Mat();
            magnitude(gx, gy, mag);
            Mat out8 = new Mat();
            mag.convertTo(out8, CV_8U, 1.0, 0); // rough scale; ok for viz / features
            if (gray != mat) gray.close();
            gx.close(); gy.close(); mag.close();
            return ensureCHW(MatToTensor.fromMat(out8));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Morphology ----

    public static Tensor dilate(Tensor tensor, int ksize, int iterations) {
        return morph(tensor, MORPH_DILATE, ksize, iterations);
    }

    public static Tensor erode(Tensor tensor, int ksize, int iterations) {
        return morph(tensor, MORPH_ERODE, ksize, iterations);
    }

    public static Tensor morphologyOpen(Tensor tensor, int ksize) {
        return morph(tensor, MORPH_OPEN, ksize, 1);
    }

    public static Tensor morphologyClose(Tensor tensor, int ksize) {
        return morph(tensor, MORPH_CLOSE, ksize, 1);
    }

    private static Tensor morph(Tensor tensor, int op, int ksize, int iterations) {
        int k = normalizeOddKernel(ksize, 3);
        int it = Math.max(1, iterations);
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat kernel = getStructuringElement(MORPH_RECT, new Size(k, k));
            Mat out = new Mat();
            morphologyEx(mat, out, op, kernel, new org.bytedeco.opencv.opencv_core.Point(-1, -1), it,
                    BORDER_CONSTANT, morphologyDefaultBorderValue());
            kernel.close();
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    // ---- Histogram / contrast ----

    /** Per-channel histogram equalization (on Y if color via YCrCb). */
    public static Tensor equalizeHist(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            if (mat.channels() == 1) {
                Mat out = new Mat();
                org.bytedeco.opencv.global.opencv_imgproc.equalizeHist(mat, out);
                return ensureCHW(MatToTensor.fromMat(out));
            }
            Mat ycrcb = new Mat();
            cvtColor(mat, ycrcb, COLOR_BGR2YCrCb);
            org.bytedeco.opencv.opencv_core.MatVector ch = new org.bytedeco.opencv.opencv_core.MatVector();
            split(ycrcb, ch);
            org.bytedeco.opencv.global.opencv_imgproc.equalizeHist(ch.get(0), ch.get(0));
            merge(ch, ycrcb);
            Mat out = new Mat();
            cvtColor(ycrcb, out, COLOR_YCrCb2BGR);
            ycrcb.close();
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /**
     * CLAHE (Contrast Limited Adaptive Histogram Equalization) — enterprise OCR / low-light.
     *
     * @param clipLimit      e.g. 2.0
     * @param tileGridSize   e.g. 8
     */
    public static Tensor clahe(Tensor tensor, double clipLimit, int tileGridSize) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            int g = Math.max(2, tileGridSize);
            org.bytedeco.opencv.opencv_imgproc.CLAHE clahe =
                    createCLAHE(clipLimit <= 0 ? 2.0 : clipLimit, new Size(g, g));
            if (mat.channels() == 1) {
                Mat out = new Mat();
                clahe.apply(mat, out);
                clahe.close();
                return ensureCHW(MatToTensor.fromMat(out));
            }
            Mat lab = new Mat();
            cvtColor(mat, lab, COLOR_BGR2Lab);
            org.bytedeco.opencv.opencv_core.MatVector ch = new org.bytedeco.opencv.opencv_core.MatVector();
            split(lab, ch);
            clahe.apply(ch.get(0), ch.get(0));
            merge(ch, lab);
            Mat out = new Mat();
            cvtColor(lab, out, COLOR_Lab2BGR);
            lab.close();
            clahe.close();
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    public static Tensor clahe(Tensor tensor) {
        return clahe(tensor, 2.0, 8);
    }

    // ---- Color spaces ----

    /**
     * Convert RGB tensor to HSV. Returns {@code [3,H,W]} with channels (H, S, V)
     * (OpenCV scale: H∈[0,180], S/V∈[0,255] as float).
     *
     * <p>Does <em>not</em> go through the BGR↔RGB swap path of {@link MatToTensor#fromMat}
     * so channel order stays H,S,V.
     */
    public static Tensor rgbToHsv(Tensor tensor) {
        return toHsv(tensor);
    }

    /**
     * Convert RGB tensor to HSV without the BGR↔RGB channel-swap heuristic.
     * Returns {@code [3,H,W]} with channels (H, S, V).
     */
    public static Tensor toHsv(Tensor tensor) {
        Mat bgr = MatToTensor.toMat(tensor);
        try {
            Mat hsv = new Mat();
            cvtColor(bgr, hsv, COLOR_BGR2HSV);
            // Split channels then cat — fromMat on 1-ch is identity (no RB swap).
            org.bytedeco.opencv.opencv_core.MatVector ch = new org.bytedeco.opencv.opencv_core.MatVector();
            split(hsv, ch);
            Tensor h = ensureCHW(MatToTensor.fromMat(ch.get(0)));
            Tensor s = ensureCHW(MatToTensor.fromMat(ch.get(1)));
            Tensor v = ensureCHW(MatToTensor.fromMat(ch.get(2)));
            hsv.close();
            return torch.cat(new org.bytedeco.pytorch.TensorVector(h, s, v), 0);
        } finally {
            if (bgr != null) bgr.close();
        }
    }

    // ---- Geometry for multimodal (letterbox / center crop / pad) ----

    /**
     * Letterbox resize keeping aspect ratio, pad to {@code outH x outW} with {@code padValue}
     * (YOLO / DETR / many VLM preprocessors).
     *
     * @return {@code [C, outH, outW]}
     */
    public static Tensor letterbox(Tensor tensor, int outH, int outW, double padValue) {
        long[] s = sizes(tensor);
        if (s.length != 3) throw new IllegalArgumentException("expected [C,H,W]");
        int h = (int) s[1], w = (int) s[2];
        if (h <= 0 || w <= 0) throw new IllegalArgumentException("invalid HxW");
        double scale = Math.min(outH / (double) h, outW / (double) w);
        int nh = Math.max(1, (int) Math.round(h * scale));
        int nw = Math.max(1, (int) Math.round(w * scale));
        Tensor resized = resize(tensor, nh, nw);
        int padT = (outH - nh) / 2;
        int padB = outH - nh - padT;
        int padL = (outW - nw) / 2;
        int padR = outW - nw - padL;
        return pad(resized, padT, padB, padL, padR, padValue);
    }

    public static Tensor letterbox(Tensor tensor, int outH, int outW) {
        return letterbox(tensor, outH, outW, 114.0); // YOLO default gray
    }

    /**
     * Constant-value pad around a {@code [C,H,W]} tensor.
     */
    public static Tensor pad(Tensor tensor, int top, int bottom, int left, int right, double value) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat out = new Mat();
            copyMakeBorder(mat, out, Math.max(0, top), Math.max(0, bottom),
                    Math.max(0, left), Math.max(0, right), BORDER_CONSTANT,
                    new org.bytedeco.opencv.opencv_core.Scalar(value, value, value, 0));
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** Center crop to {@code cropH x cropW} (clamped to image). */
    public static Tensor centerCrop(Tensor tensor, int cropH, int cropW) {
        long[] s = sizes(tensor);
        if (s.length != 3) throw new IllegalArgumentException("expected [C,H,W]");
        int h = (int) s[1], w = (int) s[2];
        int ch = Math.min(cropH, h), cw = Math.min(cropW, w);
        int y = Math.max(0, (h - ch) / 2);
        int x = Math.max(0, (w - cw) / 2);
        return crop(tensor, y, x, ch, cw);
    }

    /**
     * torchvision-style resize shorter side to {@code size} keeping aspect, then center-crop square.
     */
    public static Tensor resizeShortCenterCrop(Tensor tensor, int size) {
        long[] s = sizes(tensor);
        if (s.length != 3) throw new IllegalArgumentException("expected [C,H,W]");
        int h = (int) s[1], w = (int) s[2];
        double scale = size / (double) Math.min(h, w);
        int nh = Math.max(1, (int) Math.round(h * scale));
        int nw = Math.max(1, (int) Math.round(w * scale));
        Tensor resized = resize(tensor, nh, nw);
        return centerCrop(resized, size, size);
    }

    // ---- Brightness / contrast / threshold ----

    /**
     * {@code out = tensor * alpha + beta} clipped conceptually via convertScale on Mat.
     * {@code alpha} contrast, {@code beta} brightness (OpenCV convertScaleAbs style on 8u path).
     */
    public static Tensor adjustBrightnessContrast(Tensor tensor, double alpha, double beta) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat out = new Mat();
            mat.convertTo(out, -1, alpha, beta);
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** Binary threshold on grayscale projection. Returns {@code [1,H,W]}. */
    public static Tensor threshold(Tensor tensor, double thresh, double maxVal) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat gray = toGrayMat(mat);
            Mat out = new Mat();
            org.bytedeco.opencv.global.opencv_imgproc.threshold(
                    gray, out, thresh, maxVal, THRESH_BINARY);
            if (gray != mat) gray.close();
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    public static Tensor adaptiveThreshold(Tensor tensor, int blockSize, double C) {
        int b = normalizeOddKernel(blockSize, 11);
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat gray = toGrayMat(mat);
            Mat out = new Mat();
            org.bytedeco.opencv.global.opencv_imgproc.adaptiveThreshold(
                    gray, out, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, b, C);
            if (gray != mat) gray.close();
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** Blend two same-shaped images: {@code α*a + (1-α)*b}. */
    public static Tensor blend(Tensor a, Tensor b, double alpha) {
        Mat ma = MatToTensor.toMat(a);
        Mat mb = MatToTensor.toMat(b);
        try {
            Mat out = new Mat();
            double al = Math.max(0, Math.min(1, alpha));
            addWeighted(ma, al, mb, 1.0 - al, 0, out);
            return ensureCHW(MatToTensor.fromMat(out));
        } finally {
            if (ma != null) ma.close();
            if (mb != null) mb.close();
        }
    }

    // ---- Batch helpers ----

    /** Resize every tensor in the list to the same HxW. */
    public static List<Tensor> batchResize(List<Tensor> images, int height, int width) {
        if (images == null) return List.of();
        List<Tensor> out = new ArrayList<>(images.size());
        for (Tensor t : images) out.add(resize(t, height, width));
        return out;
    }

    /**
     * Letterbox each frame then stack to {@code [N,C,H,W]} — VLM video preprocessor core.
     */
    public static Tensor batchLetterboxStack(List<Tensor> images, int outH, int outW) {
        if (images == null || images.isEmpty()) {
            return torch.empty(new long[]{0, 3, outH, outW},
                    new org.bytedeco.pytorch.TensorOptions(
                            org.bytedeco.pytorch.global.torch.ScalarType.Float), null);
        }
        List<Tensor> boxed = new ArrayList<>(images.size());
        for (Tensor t : images) boxed.add(letterbox(t, outH, outW));
        org.bytedeco.pytorch.TensorVector tv =
                new org.bytedeco.pytorch.TensorVector(boxed.toArray(new Tensor[0]));
        return torch.stack(tv);
    }

    // ---- Image hash (perceptual near-duplicate) ----

    /**
     * Average hash (aHash) 64-bit as long — fast near-duplicate key for DataFrame dedup.
     * Pipeline: gray → resize 8x8 → mean threshold → bits.
     */
    public static long averageHash(Tensor tensor) {
        Mat mat = MatToTensor.toMat(tensor);
        try {
            Mat gray = toGrayMat(mat);
            Mat small = new Mat();
            org.bytedeco.opencv.global.opencv_imgproc.resize(gray, small, new Size(8, 8), 0, 0, INTER_AREA);
            // mean
            org.bytedeco.opencv.opencv_core.Scalar meanSc = mean(small);
            double m = meanSc.get(0);
            long hash = 0L;
            byte[] buf = new byte[64];
            small.data().get(buf);
            for (int i = 0; i < 64; i++) {
                int v = buf[i] & 0xFF;
                if (v >= m) hash |= (1L << i);
            }
            if (gray != mat) gray.close();
            small.close();
            return hash;
        } finally {
            if (mat != null) mat.close();
        }
    }

    /** Hamming distance between two 64-bit hashes. */
    public static int hamming64(long a, long b) {
        return Long.bitCount(a ^ b);
    }

    // ---- Optical flow (2-frame) ----

    /**
     * Farneback dense optical flow between two frames.
     * @return {@code [2,H,W]} float32 flow (dx, dy) — values are displacement, not 0-255
     */
    public static Tensor opticalFlowFarneback(Tensor prev, Tensor next) {
        Mat m0 = MatToTensor.toMat(prev);
        Mat m1 = MatToTensor.toMat(next);
        try {
            Mat g0 = toGrayMat(m0);
            Mat g1 = toGrayMat(m1);
            Mat flow = new Mat();
            // pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            calcOpticalFlowFarneback(g0, g1, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            // flow is HxW CV_32FC2 → MatToTensor.fromFloatMat yields [2,H,W]
            Tensor out = MatToTensor.fromMat(flow);
            if (g0 != m0) g0.close();
            if (g1 != m1) g1.close();
            flow.close();
            return ensureCHW(out);
        } finally {
            if (m0 != null) m0.close();
            if (m1 != null) m1.close();
        }
    }

    // ---- helpers ----

    private static Mat toGrayMat(Mat mat) {
        if (mat.channels() == 1) return mat;
        Mat gray = new Mat();
        cvtColor(mat, gray, COLOR_BGR2GRAY);
        return gray;
    }

    private static int normalizeOddKernel(int ksize, int defaultOdd) {
        int k = ksize <= 0 ? defaultOdd : ksize;
        if ((k & 1) == 0) k++;
        return Math.max(1, k);
    }

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
