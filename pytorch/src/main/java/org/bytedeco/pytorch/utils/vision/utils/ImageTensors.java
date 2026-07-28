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
package org.bytedeco.pytorch.utils.vision.utils;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.util.Objects;

/**
 * Converters between {@link BufferedImage} / {@link ImageData} and CHW / NCHW
 * {@link Tensor}s (torchvision-style, float in {@code [0, 1]} by default).
 */
public final class ImageTensors {
    private ImageTensors() {}

    /** Degrees → radians (ImageData.rotate uses radians; torchvision uses degrees). */
    public static double deg2rad(double degrees) {
        return Math.toRadians(degrees);
    }

    public static double rad2deg(double radians) {
        return Math.toDegrees(radians);
    }

    /**
     * Convert a {@link BufferedImage} to float CHW tensor in {@code [0, 1]}.
     * Shape {@code [C, H, W]} with C=1 (gray) or C=3 (RGB). Alpha is dropped.
     */
    public static Tensor toTensor(BufferedImage image) {
        Objects.requireNonNull(image, "image");
        BufferedImage rgb = ensureRgbOrGray(image);
        int w = rgb.getWidth();
        int h = rgb.getHeight();
        int type = rgb.getType();
        boolean gray = type == BufferedImage.TYPE_BYTE_GRAY;
        int c = gray ? 1 : 3;
        float[] data = new float[c * h * w];
        if (gray) {
            byte[] pixels = ((DataBufferByte) rgb.getRaster().getDataBuffer()).getData();
            for (int i = 0; i < h * w; i++) {
                data[i] = (pixels[i] & 0xff) / 255.0f;
            }
        } else {
            int[] pixels;
            if (rgb.getRaster().getDataBuffer() instanceof DataBufferInt) {
                pixels = ((DataBufferInt) rgb.getRaster().getDataBuffer()).getData();
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int p = pixels[y * w + x];
                        int r = (p >> 16) & 0xff;
                        int g = (p >> 8) & 0xff;
                        int b = p & 0xff;
                        int idx = y * w + x;
                        data[0 * h * w + idx] = r / 255.0f;
                        data[1 * h * w + idx] = g / 255.0f;
                        data[2 * h * w + idx] = b / 255.0f;
                    }
                }
            } else {
                for (int y = 0; y < h; y++) {
                    for (int x = 0; x < w; x++) {
                        int p = rgb.getRGB(x, y);
                        int r = (p >> 16) & 0xff;
                        int g = (p >> 8) & 0xff;
                        int b = p & 0xff;
                        int idx = y * w + x;
                        data[0 * h * w + idx] = r / 255.0f;
                        data[1 * h * w + idx] = g / 255.0f;
                        data[2 * h * w + idx] = b / 255.0f;
                    }
                }
            }
        }
        return torch.tensor(data).reshape(c, h, w);
    }

    public static Tensor toTensor(ImageData image) {
        Objects.requireNonNull(image, "image");
        BufferedImage bi = image.getImage();
        if (bi == null) {
            throw new IllegalArgumentException("ImageData has no BufferedImage loaded");
        }
        return toTensor(bi);
    }

    /** Float CHW {@code [C,H,W]} or NCHW {@code [N,C,H,W]} in roughly {@code [0,1]} → BufferedImage (first batch item). */
    public static BufferedImage toBufferedImage(Tensor t) {
        Objects.requireNonNull(t, "tensor");
        Tensor cpu = t.contiguous().cpu().to(ScalarType.Float);
        long[] sizes = sizes(cpu);
        int n = 1, c, h, w;
        if (sizes.length == 3) {
            c = (int) sizes[0];
            h = (int) sizes[1];
            w = (int) sizes[2];
        } else if (sizes.length == 4) {
            n = (int) sizes[0];
            c = (int) sizes[1];
            h = (int) sizes[2];
            w = (int) sizes[3];
            if (n < 1) {
                throw new IllegalArgumentException("empty batch");
            }
            cpu = cpu.narrow(0, 0, 1).squeeze(0);
        } else if (sizes.length == 2) {
            c = 1;
            h = (int) sizes[0];
            w = (int) sizes[1];
            cpu = cpu.reshape(1, h, w);
        } else {
            throw new IllegalArgumentException("expected CHW/NCHW/HW, got rank " + sizes.length);
        }
        float[] data = toFloatArray(cpu);
        if (c == 1) {
            BufferedImage gray = new BufferedImage(w, h, BufferedImage.TYPE_BYTE_GRAY);
            byte[] pixels = ((DataBufferByte) gray.getRaster().getDataBuffer()).getData();
            for (int i = 0; i < h * w; i++) {
                int v = Math.round(clamp01(data[i]) * 255.0f);
                pixels[i] = (byte) v;
            }
            return gray;
        }
        BufferedImage rgb = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = y * w + x;
                int r = Math.round(clamp01(data[0 * h * w + idx]) * 255.0f);
                int g = Math.round(clamp01(c > 1 ? data[1 * h * w + idx] : data[idx]) * 255.0f);
                int b = Math.round(clamp01(c > 2 ? data[2 * h * w + idx] : data[idx]) * 255.0f);
                rgb.setRGB(x, y, (r << 16) | (g << 8) | b);
            }
        }
        return rgb;
    }

    public static ImageData toImageData(Tensor t) {
        return new ImageData(toBufferedImage(t));
    }

    /**
     * Pack a batch of CHW images already laid out as float arrays of length {@code C*H*W}.
     * {@code chwFlat[i]} is one image in channel-first order; result shape {@code [N,C,H,W]}.
     */
    public static Tensor stackCHW(float[][] chwFlat, int c, int h, int w) {
        Objects.requireNonNull(chwFlat, "batch");
        if (c <= 0 || h <= 0 || w <= 0) {
            throw new IllegalArgumentException("c,h,w must be > 0, got " + c + "," + h + "," + w);
        }
        int n = chwFlat.length;
        int plane = c * h * w;
        float[] all = new float[n * plane];
        for (int i = 0; i < n; i++) {
            float[] src = Objects.requireNonNull(chwFlat[i], "chwFlat[" + i + "]");
            if (src.length != plane) {
                throw new IllegalArgumentException(
                        "chwFlat[" + i + "] length " + src.length + " != C*H*W=" + plane);
            }
            System.arraycopy(src, 0, all, i * plane, plane);
        }
        return torch.tensor(all).reshape(n, c, h, w);
    }

    public static float[] toFloatArray(Tensor t) {
        Tensor cpu = t.contiguous().cpu().to(ScalarType.Float);
        long n = cpu.numel();
        float[] data = new float[(int) n];
        FloatPointer ptr = cpu.data_ptr_float();
        for (int i = 0; i < n; i++) {
            data[i] = ptr.get(i);
        }
        return data;
    }

    public static long[] sizes(Tensor t) {
        long ndim = t.dim();
        long[] out = new long[(int) ndim];
        for (int i = 0; i < ndim; i++) {
            out[i] = t.size(i);
        }
        return out;
    }

    private static float clamp01(float v) {
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
    }

    private static BufferedImage ensureRgbOrGray(BufferedImage image) {
        int type = image.getType();
        if (type == BufferedImage.TYPE_BYTE_GRAY
                || type == BufferedImage.TYPE_INT_RGB
                || type == BufferedImage.TYPE_INT_ARGB
                || type == BufferedImage.TYPE_3BYTE_BGR
                || type == BufferedImage.TYPE_4BYTE_ABGR) {
            if (type == BufferedImage.TYPE_BYTE_GRAY) {
                return image;
            }
            if (type == BufferedImage.TYPE_INT_RGB || type == BufferedImage.TYPE_INT_ARGB) {
                return image;
            }
        }
        BufferedImage rgb = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
        rgb.getGraphics().drawImage(image, 0, 0, null);
        return rgb;
    }
}
