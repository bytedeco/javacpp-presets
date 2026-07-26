/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
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
package org.bytedeco.pytorch.utils.vision.transforms.functional;
import org.bytedeco.pytorch.nn.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.utils.vision.utils.ImageTensors;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.ConvolveOp;
import java.awt.image.Kernel;
import java.util.Objects;

/**
 * Functional image operators (torchvision.transforms.functional / {@code F}).
 * Accepts {@link BufferedImage}, {@link ImageData}, or CHW/NCHW {@link Tensor}.
 */
public final class F {
    private F() {}

    public static BufferedImage asBufferedImage(Object img) {
        if (img instanceof BufferedImage bi) {
            return bi;
        }
        if (img instanceof ImageData id) {
            BufferedImage bi = id.getImage();
            if (bi == null) {
                throw new IllegalArgumentException("ImageData has no image");
            }
            return bi;
        }
        if (img instanceof Tensor t) {
            return ImageTensors.toBufferedImage(t);
        }
        throw new IllegalArgumentException("unsupported image type: " + (img == null ? "null" : img.getClass()));
    }

    public static ImageData asImageData(Object img) {
        if (img instanceof ImageData id) {
            return id;
        }
        return new ImageData(asBufferedImage(img));
    }

    public static BufferedImage resize(Object img, int size) {
        return resize(img, size, size);
    }

    public static BufferedImage resize(Object img, int height, int width) {
        BufferedImage src = asBufferedImage(img);
        BufferedImage dst = new BufferedImage(width, height, rgbType(src));
        Graphics2D g = dst.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, 0, 0, width, height, null);
        g.dispose();
        return dst;
    }

    public static BufferedImage centerCrop(Object img, int size) {
        return centerCrop(img, size, size);
    }

    public static BufferedImage centerCrop(Object img, int cropHeight, int cropWidth) {
        BufferedImage src = asBufferedImage(img);
        int w = src.getWidth();
        int h = src.getHeight();
        int x = Math.max(0, (w - cropWidth) / 2);
        int y = Math.max(0, (h - cropHeight) / 2);
        return crop(src, y, x, cropHeight, cropWidth);
    }

    public static BufferedImage crop(Object img, int top, int left, int height, int width) {
        BufferedImage src = asBufferedImage(img);
        int w = Math.min(width, src.getWidth() - left);
        int h = Math.min(height, src.getHeight() - top);
        if (w <= 0 || h <= 0) {
            throw new IllegalArgumentException("invalid crop");
        }
        return src.getSubimage(left, top, w, h);
    }

    public static BufferedImage hflip(Object img) {
        BufferedImage src = asBufferedImage(img);
        BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), rgbType(src));
        Graphics2D g = dst.createGraphics();
        g.drawImage(src, 0, 0, src.getWidth(), src.getHeight(), src.getWidth(), 0, 0, src.getHeight(), null);
        g.dispose();
        return dst;
    }

    public static BufferedImage vflip(Object img) {
        BufferedImage src = asBufferedImage(img);
        BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), rgbType(src));
        Graphics2D g = dst.createGraphics();
        g.drawImage(src, 0, 0, src.getWidth(), src.getHeight(), 0, src.getHeight(), src.getWidth(), 0, null);
        g.dispose();
        return dst;
    }

    /** Rotate counter-clockwise by degrees (torchvision convention). */
    public static BufferedImage rotate(Object img, double degrees) {
        BufferedImage src = asBufferedImage(img);
        double rad = Math.toRadians(degrees);
        double cos = Math.abs(Math.cos(rad));
        double sin = Math.abs(Math.sin(rad));
        int w = src.getWidth();
        int h = src.getHeight();
        int nw = (int) Math.floor(w * cos + h * sin);
        int nh = (int) Math.floor(w * sin + h * cos);
        BufferedImage dst = new BufferedImage(Math.max(1, nw), Math.max(1, nh), rgbType(src));
        Graphics2D g = dst.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        AffineTransform at = new AffineTransform();
        at.translate(nw / 2.0, nh / 2.0);
        at.rotate(-rad);
        at.translate(-w / 2.0, -h / 2.0);
        g.drawImage(src, at, null);
        g.dispose();
        return dst;
    }

    public static BufferedImage pad(Object img, int padding) {
        return pad(img, padding, padding, padding, padding, 0);
    }

    public static BufferedImage pad(Object img, int top, int left, int bottom, int right, int fill) {
        BufferedImage src = asBufferedImage(img);
        int nw = src.getWidth() + left + right;
        int nh = src.getHeight() + top + bottom;
        BufferedImage dst = new BufferedImage(nw, nh, rgbType(src));
        Graphics2D g = dst.createGraphics();
        g.setColor(new java.awt.Color(fill, fill, fill));
        g.fillRect(0, 0, nw, nh);
        g.drawImage(src, left, top, null);
        g.dispose();
        return dst;
    }

    public static BufferedImage toGrayscale(Object img, int numOutputChannels) {
        BufferedImage src = asBufferedImage(img);
        BufferedImage gray = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = gray.createGraphics();
        g.drawImage(src, 0, 0, null);
        g.dispose();
        if (numOutputChannels <= 1) {
            return gray;
        }
        BufferedImage rgb = new BufferedImage(src.getWidth(), src.getHeight(), BufferedImage.TYPE_INT_RGB);
        rgb.getGraphics().drawImage(gray, 0, 0, null);
        return rgb;
    }

    public static BufferedImage adjustBrightness(Object img, float factor) {
        return asImageData(img).adjustBrightness(factor).getImage();
    }

    public static BufferedImage adjustContrast(Object img, float factor) {
        return asImageData(img).adjustContrast(factor).getImage();
    }

    public static BufferedImage adjustSaturation(Object img, float factor) {
        return asImageData(img).adjustSaturation(factor).getImage();
    }

    public static BufferedImage adjustHue(Object img, float hueShift) {
        // ImageData uses absolute hue shift; torchvision hue in [-0.5,0.5] ~ fraction of circle
        return asImageData(img).adjustHue(hueShift * 360f).getImage();
    }

    public static BufferedImage gaussianBlur(Object img, int kernelSize, double sigma) {
        BufferedImage src = asBufferedImage(img);
        if (kernelSize % 2 == 0) {
            kernelSize++;
        }
        float[] k = gaussianKernel(kernelSize, sigma <= 0 ? 1.0 : sigma);
        Kernel kernel = new Kernel(kernelSize, kernelSize, k);
        ConvolveOp op = new ConvolveOp(kernel, ConvolveOp.EDGE_NO_OP, null);
        BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), rgbType(src));
        return op.filter(src, dst);
    }

    public static Tensor normalize(Tensor tensor, float[] mean, float[] std) {
        Objects.requireNonNull(tensor, "tensor");
        Objects.requireNonNull(mean, "mean");
        Objects.requireNonNull(std, "std");
        Tensor t = tensor.to(org.bytedeco.pytorch.global.torch.ScalarType.Float);
        long[] sizes = ImageTensors.sizes(t);
        if (sizes.length < 3) {
            throw new IllegalArgumentException("normalize expects CHW or NCHW");
        }
        int cIndex = sizes.length == 3 ? 0 : 1;
        int c = (int) sizes[cIndex];
        if (mean.length != c || std.length != c) {
            throw new IllegalArgumentException("mean/std length must equal channels=" + c);
        }
        // (t - mean) / std per channel
        float[] data = ImageTensors.toFloatArray(t);
        int plane;
        int n;
        int h;
        int w;
        if (sizes.length == 3) {
            n = 1;
            h = (int) sizes[1];
            w = (int) sizes[2];
            plane = h * w;
            for (int ch = 0; ch < c; ch++) {
                float m = mean[ch];
                float s = std[ch] == 0f ? 1e-6f : std[ch];
                int off = ch * plane;
                for (int i = 0; i < plane; i++) {
                    data[off + i] = (data[off + i] - m) / s;
                }
            }
            return torch.tensor(data).reshape(c, h, w);
        }
        n = (int) sizes[0];
        h = (int) sizes[2];
        w = (int) sizes[3];
        plane = h * w;
        int sample = c * plane;
        for (int bi = 0; bi < n; bi++) {
            for (int ch = 0; ch < c; ch++) {
                float m = mean[ch];
                float s = std[ch] == 0f ? 1e-6f : std[ch];
                int off = bi * sample + ch * plane;
                for (int i = 0; i < plane; i++) {
                    data[off + i] = (data[off + i] - m) / s;
                }
            }
        }
        return torch.tensor(data).reshape(n, c, h, w);
    }

    public static Tensor toTensor(Object img) {
        if (img instanceof Tensor t) {
            return t;
        }
        return ImageTensors.toTensor(asBufferedImage(img));
    }

    // -------------------------------------------------------------------------
    // Affine / Perspective / Erase / Photometric
    // -------------------------------------------------------------------------

    /**
     * Affine transform matching torchvision.transforms.functional.affine.
     *
     * @param degrees  counter-clockwise rotation in degrees
     * @param translate {tx, ty} in pixels (may be null → 0)
     * @param scale     scale factor (1.0 = identity)
     * @param shear     {shear_x, shear_y} degrees (null → 0); single value = shear_x only
     * @param fill      RGB fill 0–255 (grayscale fill used for all channels if scalar)
     */
    public static BufferedImage affine(Object img, double degrees, double[] translate,
                                       double scale, double[] shear, int fill) {
        BufferedImage src = asBufferedImage(img);
        int w = src.getWidth();
        int h = src.getHeight();
        double tx = translate != null && translate.length > 0 ? translate[0] : 0.0;
        double ty = translate != null && translate.length > 1 ? translate[1] : 0.0;
        double sc = scale <= 0 ? 1.0 : scale;
        double shx = shear != null && shear.length > 0 ? shear[0] : 0.0;
        double shy = shear != null && shear.length > 1 ? shear[1] : 0.0;

        // torchvision builds matrix as: center → rotate → shear → scale → translate → uncenter
        double rad = Math.toRadians(degrees);
        double tanX = Math.tan(Math.toRadians(shx));
        double tanY = Math.tan(Math.toRadians(shy));

        double cx = w / 2.0;
        double cy = h / 2.0;

        AffineTransform at = new AffineTransform();
        // final: un-center + translate
        at.translate(cx + tx, cy + ty);
        // scale
        at.scale(sc, sc);
        // shear (after rotation in torchvision order — apply inverse composition carefully)
        // matrix M = T(center+trans) · S · Sh · R · T(-center)
        // AffineTransform multiplies on the right when concatenating via translate/scale/rotate,
        // so we apply in reverse visual order.
        // shear: x' = x + tanX*y ; y' = y + tanY*x  approximated via shear + slight scale
        at.shear(tanX, tanY);
        // rotate (Java rotate is clockwise for positive; torchvision is CCW → negate)
        at.rotate(-rad);
        at.translate(-cx, -cy);

        BufferedImage dst = new BufferedImage(w, h, rgbType(src));
        Graphics2D g = dst.createGraphics();
        g.setColor(new java.awt.Color(clamp255(fill), clamp255(fill), clamp255(fill)));
        g.fillRect(0, 0, w, h);
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(src, at, null);
        g.dispose();
        return dst;
    }

    public static BufferedImage affine(Object img, double degrees) {
        return affine(img, degrees, null, 1.0, null, 0);
    }

    /**
     * Perspective transform. {@code startpoints} / {@code endpoints} are 4 corner pairs
     * [tl, tr, br, bl] each as {x, y}.
     * <p>
     * Maps the quadrilateral {@code startpoints} in the source image onto
     * {@code endpoints} in the destination (homography, inverse mapping).
     */
    public static BufferedImage perspective(Object img, double[][] startpoints, double[][] endpoints, int fill) {
        Objects.requireNonNull(startpoints, "startpoints");
        Objects.requireNonNull(endpoints, "endpoints");
        if (startpoints.length < 4 || endpoints.length < 4) {
            throw new IllegalArgumentException("need 4 corner points");
        }
        BufferedImage src = asBufferedImage(img);
        int w = src.getWidth();
        int h = src.getHeight();
        // Homography H such that endpoints = H * startpoints
        // For inverse mapping we need H^{-1}: src = Hinv * dst
        double[][] H = computeHomography(startpoints, endpoints);
        double[][] Hinv = invert3x3(H);
        if (Hinv == null) {
            // degenerate — fall back to identity
            BufferedImage copy = new BufferedImage(w, h, rgbType(src));
            Graphics2D g = copy.createGraphics();
            g.drawImage(src, 0, 0, null);
            g.dispose();
            return copy;
        }
        int fillRgb = (0xFF << 24) | (clamp255(fill) << 16) | (clamp255(fill) << 8) | clamp255(fill);
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                // p_src = Hinv * p_dst
                double X = Hinv[0][0] * x + Hinv[0][1] * y + Hinv[0][2];
                double Y = Hinv[1][0] * x + Hinv[1][1] * y + Hinv[1][2];
                double W = Hinv[2][0] * x + Hinv[2][1] * y + Hinv[2][2];
                if (Math.abs(W) < 1e-12) {
                    dst.setRGB(x, y, fillRgb);
                    continue;
                }
                double sx = X / W;
                double sy = Y / W;
                int rgb = sampleBilinear(src, sx, sy, fillRgb);
                dst.setRGB(x, y, rgb);
            }
        }
        return dst;
    }

    /** 4-point homography: maps srcPts → dstPts (each length-4 of {x,y}). */
    private static double[][] computeHomography(double[][] srcPts, double[][] dstPts) {
        // Solve Ah = b for h (8 DOF), H = [[h0,h1,h2],[h3,h4,h5],[h6,h7,1]]
        double[][] A = new double[8][8];
        double[] b = new double[8];
        for (int i = 0; i < 4; i++) {
            double x = srcPts[i][0], y = srcPts[i][1];
            double u = dstPts[i][0], v = dstPts[i][1];
            int r = i * 2;
            A[r][0] = x; A[r][1] = y; A[r][2] = 1;
            A[r][3] = 0; A[r][4] = 0; A[r][5] = 0;
            A[r][6] = -x * u; A[r][7] = -y * u;
            b[r] = u;
            A[r + 1][0] = 0; A[r + 1][1] = 0; A[r + 1][2] = 0;
            A[r + 1][3] = x; A[r + 1][4] = y; A[r + 1][5] = 1;
            A[r + 1][6] = -x * v; A[r + 1][7] = -y * v;
            b[r + 1] = v;
        }
        double[] h = solveLinear8(A, b);
        if (h == null) {
            return identity3();
        }
        return new double[][]{
                {h[0], h[1], h[2]},
                {h[3], h[4], h[5]},
                {h[6], h[7], 1.0}
        };
    }

    /** Gaussian elimination for 8x8. Returns null on failure. */
    private static double[] solveLinear8(double[][] A, double[] b) {
        int n = 8;
        double[][] M = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(A[i], 0, M[i], 0, n);
            M[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) {
                if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
            }
            if (Math.abs(M[pivot][col]) < 1e-12) {
                return null;
            }
            if (pivot != col) {
                double[] tmp = M[col];
                M[col] = M[pivot];
                M[pivot] = tmp;
            }
            double div = M[col][col];
            for (int c = col; c <= n; c++) M[col][c] /= div;
            for (int r = 0; r < n; r++) {
                if (r == col) continue;
                double f = M[r][col];
                for (int c = col; c <= n; c++) M[r][c] -= f * M[col][c];
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = M[i][n];
        return x;
    }

    private static double[][] invert3x3(double[][] m) {
        double a = m[0][0], b = m[0][1], c = m[0][2];
        double d = m[1][0], e = m[1][1], f = m[1][2];
        double g = m[2][0], h = m[2][1], i = m[2][2];
        double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if (Math.abs(det) < 1e-12) {
            return null;
        }
        double invDet = 1.0 / det;
        return new double[][]{
                {(e * i - f * h) * invDet, (c * h - b * i) * invDet, (b * f - c * e) * invDet},
                {(f * g - d * i) * invDet, (a * i - c * g) * invDet, (c * d - a * f) * invDet},
                {(d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet}
        };
    }

    private static double[][] identity3() {
        return new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    }

    private static int sampleBilinear(BufferedImage src, double sx, double sy, int fillRgb) {
        int w = src.getWidth();
        int h = src.getHeight();
        if (sx < 0 || sy < 0 || sx >= w - 1 || sy >= h - 1) {
            if (sx < -0.5 || sy < -0.5 || sx >= w - 0.5 || sy >= h - 0.5) {
                return fillRgb;
            }
            int ix = (int) Math.max(0, Math.min(w - 1, Math.round(sx)));
            int iy = (int) Math.max(0, Math.min(h - 1, Math.round(sy)));
            return src.getRGB(ix, iy);
        }
        int x0 = (int) Math.floor(sx);
        int y0 = (int) Math.floor(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        double dx = sx - x0;
        double dy = sy - y0;
        int c00 = src.getRGB(x0, y0);
        int c10 = src.getRGB(x1, y0);
        int c01 = src.getRGB(x0, y1);
        int c11 = src.getRGB(x1, y1);
        int r = bilinearChan(c00, c10, c01, c11, 16, dx, dy);
        int g = bilinearChan(c00, c10, c01, c11, 8, dx, dy);
        int b = bilinearChan(c00, c10, c01, c11, 0, dx, dy);
        return (0xFF << 24) | (r << 16) | (g << 8) | b;
    }

    private static int bilinearChan(int c00, int c10, int c01, int c11, int shift, double dx, double dy) {
        double v00 = (c00 >> shift) & 0xFF;
        double v10 = (c10 >> shift) & 0xFF;
        double v01 = (c01 >> shift) & 0xFF;
        double v11 = (c11 >> shift) & 0xFF;
        double v0 = v00 * (1 - dx) + v10 * dx;
        double v1 = v01 * (1 - dx) + v11 * dx;
        return clamp255((int) Math.round(v0 * (1 - dy) + v1 * dy));
    }

    /**
     * Solarize: invert all pixel values above threshold (0–255), torchvision.functional.solarize.
     */
    public static BufferedImage solarize(Object img, double threshold) {
        BufferedImage src = asBufferedImage(img);
        int thr = (int) Math.round(Math.max(0, Math.min(255, threshold)));
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage dst = new BufferedImage(w, h, rgbType(src));
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int a = (rgb >>> 24) & 0xFF;
                int r = (rgb >>> 16) & 0xFF;
                int g = (rgb >>> 8) & 0xFF;
                int b = rgb & 0xFF;
                if (r >= thr) r = 255 - r;
                if (g >= thr) g = 255 - g;
                if (b >= thr) b = 255 - b;
                dst.setRGB(x, y, (a << 24) | (r << 16) | (g << 8) | b);
            }
        }
        return dst;
    }

    /**
     * Autocontrast: stretch each channel histogram to full [0,255] range
     * (torchvision.functional.autocontrast).
     */
    public static BufferedImage autocontrast(Object img) {
        BufferedImage src = asBufferedImage(img);
        int w = src.getWidth();
        int h = src.getHeight();
        int[] minC = {255, 255, 255};
        int[] maxC = {0, 0, 0};
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int r = (rgb >>> 16) & 0xFF;
                int g = (rgb >>> 8) & 0xFF;
                int b = rgb & 0xFF;
                if (r < minC[0]) minC[0] = r;
                if (g < minC[1]) minC[1] = g;
                if (b < minC[2]) minC[2] = b;
                if (r > maxC[0]) maxC[0] = r;
                if (g > maxC[1]) maxC[1] = g;
                if (b > maxC[2]) maxC[2] = b;
            }
        }
        int[] scale = new int[3];
        int[] offset = new int[3];
        for (int c = 0; c < 3; c++) {
            int range = maxC[c] - minC[c];
            if (range == 0) {
                scale[c] = 1;
                offset[c] = 0;
            } else {
                // map [min,max] → [0,255]
                scale[c] = range;
                offset[c] = minC[c];
            }
        }
        BufferedImage dst = new BufferedImage(w, h, rgbType(src));
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int a = (rgb >>> 24) & 0xFF;
                int r = scaleChannel((rgb >>> 16) & 0xFF, offset[0], scale[0], maxC[0], minC[0]);
                int g = scaleChannel((rgb >>> 8) & 0xFF, offset[1], scale[1], maxC[1], minC[1]);
                int b = scaleChannel(rgb & 0xFF, offset[2], scale[2], maxC[2], minC[2]);
                dst.setRGB(x, y, (a << 24) | (r << 16) | (g << 8) | b);
            }
        }
        return dst;
    }

    /** Histogram equalization (torchvision.functional.equalize) — per-channel on RGB. */
    public static BufferedImage equalize(Object img) {
        BufferedImage src = asBufferedImage(img);
        if (src.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            return asImageData(src).equalizeHistogram().getImage();
        }
        int w = src.getWidth();
        int h = src.getHeight();
        int[][] hist = new int[3][256];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                hist[0][(rgb >>> 16) & 0xFF]++;
                hist[1][(rgb >>> 8) & 0xFF]++;
                hist[2][rgb & 0xFF]++;
            }
        }
        int[][] lut = new int[3][256];
        int total = w * h;
        for (int c = 0; c < 3; c++) {
            int[] cdf = new int[256];
            cdf[0] = hist[c][0];
            for (int i = 1; i < 256; i++) {
                cdf[i] = cdf[i - 1] + hist[c][i];
            }
            // find first non-zero cdf for proper equalization
            int cdfMin = 0;
            for (int i = 0; i < 256; i++) {
                if (cdf[i] > 0) {
                    cdfMin = cdf[i];
                    break;
                }
            }
            int denom = total - cdfMin;
            if (denom <= 0) {
                for (int i = 0; i < 256; i++) lut[c][i] = i;
            } else {
                for (int i = 0; i < 256; i++) {
                    lut[c][i] = clamp255(Math.round((cdf[i] - cdfMin) * 255f / denom));
                }
            }
        }
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int a = (rgb >>> 24) & 0xFF;
                int r = lut[0][(rgb >>> 16) & 0xFF];
                int g = lut[1][(rgb >>> 8) & 0xFF];
                int b = lut[2][rgb & 0xFF];
                dst.setRGB(x, y, (a << 24) | (r << 16) | (g << 8) | b);
            }
        }
        return dst;
    }

    /**
     * Adjust sharpness (torchvision.functional.adjust_sharpness).
     * factor 0 = blurred, 1 = original, 2 = sharpened.
     */
    public static BufferedImage adjustSharpness(Object img, float factor) {
        BufferedImage src = asBufferedImage(img);
        if (Math.abs(factor - 1f) < 1e-6f) {
            return src;
        }
        // Blend original with a mildly blurred version: out = (1-f)*blur + f*orig
        // equivalently: orig + (f-1)*(orig - blur)
        BufferedImage blurred = gaussianBlur(src, 3, 1.0);
        int w = src.getWidth();
        int h = src.getHeight();
        BufferedImage dst = new BufferedImage(w, h, rgbType(src));
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int o = src.getRGB(x, y);
                int b = blurred.getRGB(x, y);
                int a = (o >>> 24) & 0xFF;
                int r = blendSharp((o >>> 16) & 0xFF, (b >>> 16) & 0xFF, factor);
                int g = blendSharp((o >>> 8) & 0xFF, (b >>> 8) & 0xFF, factor);
                int bl = blendSharp(o & 0xFF, b & 0xFF, factor);
                dst.setRGB(x, y, (a << 24) | (r << 16) | (g << 8) | bl);
            }
        }
        return dst;
    }

    /**
     * Random erasing / cutout on a CHW FloatTensor in [0,1] (or image converted to tensor).
     * Returns Tensor always (matches common training usage of RandomErasing after ToTensor).
     *
     * @param value per-channel fill, or single value broadcast; null → zeros
     */
    public static Tensor erase(Tensor img, int i, int j, int h, int w, float[] value, boolean inplace) {
        Objects.requireNonNull(img, "img");
        Tensor t = inplace ? img : img.clone();
        long[] sizes = ImageTensors.sizes(t);
        if (sizes.length < 3) {
            throw new IllegalArgumentException("erase expects CHW tensor");
        }
        int c = (int) sizes[sizes.length == 3 ? 0 : 1];
        int H = (int) sizes[sizes.length == 3 ? 1 : 2];
        int W = (int) sizes[sizes.length == 3 ? 2 : 3];
        float[] data = ImageTensors.toFloatArray(t);
        float[] fill = value == null ? new float[c] : value;
        if (fill.length == 1 && c > 1) {
            float v = fill[0];
            fill = new float[c];
            java.util.Arrays.fill(fill, v);
        }
        int plane = H * W;
        // only handle CHW (3D); for NCHW apply on first sample
        int base = 0;
        if (sizes.length == 4) {
            // operate on all batch items identically region
            int n = (int) sizes[0];
            int sample = c * plane;
            for (int bi = 0; bi < n; bi++) {
                base = bi * sample;
                fillRect(data, base, c, H, W, plane, i, j, h, w, fill);
            }
        } else {
            fillRect(data, 0, c, H, W, plane, i, j, h, w, fill);
        }
        if (sizes.length == 3) {
            return torch.tensor(data).reshape(c, H, W);
        }
        return torch.tensor(data).reshape(sizes[0], c, H, W);
    }

    /** Erase a rectangular region on a BufferedImage with constant fill. */
    public static BufferedImage erase(Object img, int top, int left, int height, int width, int fill) {
        BufferedImage src = asBufferedImage(img);
        BufferedImage dst = new BufferedImage(src.getWidth(), src.getHeight(), rgbType(src));
        Graphics2D g = dst.createGraphics();
        g.drawImage(src, 0, 0, null);
        g.setColor(new java.awt.Color(clamp255(fill), clamp255(fill), clamp255(fill)));
        g.fillRect(left, top, width, height);
        g.dispose();
        return dst;
    }

    /** Invert colors (helper for solarize edge cases / RandomInvert). */
    public static BufferedImage invert(Object img) {
        return asImageData(img).invert().getImage();
    }

    private static void fillRect(float[] data, int base, int c, int H, int W, int plane,
                                 int i, int j, int h, int w, float[] fill) {
        int i1 = Math.min(H, Math.max(0, i));
        int j1 = Math.min(W, Math.max(0, j));
        int i2 = Math.min(H, i1 + Math.max(0, h));
        int j2 = Math.min(W, j1 + Math.max(0, w));
        for (int ch = 0; ch < c; ch++) {
            float v = ch < fill.length ? fill[ch] : 0f;
            int off = base + ch * plane;
            for (int yy = i1; yy < i2; yy++) {
                int row = off + yy * W;
                for (int xx = j1; xx < j2; xx++) {
                    data[row + xx] = v;
                }
            }
        }
    }

    private static int scaleChannel(int v, int offset, int scale, int maxC, int minC) {
        if (maxC == minC) {
            return v;
        }
        return clamp255(Math.round((v - offset) * 255f / scale));
    }

    private static int blendSharp(int orig, int blur, float factor) {
        // out = orig + (factor - 1) * (orig - blur)
        float v = orig + (factor - 1f) * (orig - blur);
        return clamp255(Math.round(v));
    }

    private static int clamp255(int v) {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return v;
    }

    private static int rgbType(BufferedImage src) {
        if (src.getType() == BufferedImage.TYPE_BYTE_GRAY) {
            return BufferedImage.TYPE_BYTE_GRAY;
        }
        return BufferedImage.TYPE_INT_RGB;
    }

    private static float[] gaussianKernel(int k, double sigma) {
        float[] kernel = new float[k * k];
        double mean = (k - 1) / 2.0;
        double sum = 0;
        for (int y = 0; y < k; y++) {
            for (int x = 0; x < k; x++) {
                double v = Math.exp(-0.5 * (Math.pow((x - mean) / sigma, 2) + Math.pow((y - mean) / sigma, 2)));
                kernel[y * k + x] = (float) v;
                sum += v;
            }
        }
        for (int i = 0; i < kernel.length; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }
}
