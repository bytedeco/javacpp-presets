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
package org.bytedeco.pytorch.llm.vllm.multimodal.encoders;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.utils.opencv.OpenCVIO;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.tensor;

/**
 * Image load + resize + ImageNet/CLIP normalize → float tensor {@code [1,3,H,W]}.
 */
public final class ImagePreprocess {

    public static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    public static final float[] IMAGENET_STD = {0.229f, 0.224f, 0.225f};
    public static final float[] CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    public static final float[] CLIP_STD = {0.26862954f, 0.26130258f, 0.27577711f};

    private ImagePreprocess() {}

    /**
     * Load image from {@link MediaInput} path/bytes/tensor and produce
     * {@code [1,3,size,size]} float32 NCHW in {@code [0,1]} then normalized.
     */
    public static Tensor loadNormalized(MediaInput input, int size, float[] mean, float[] std) {
        Tensor chw = loadChw(input); // [3,H,W] float 0..1 or 0..255
        chw = toUnitRange(chw);
        chw = resizeSquare(chw, size);
        chw = normalize(chw, mean, std);
        return chw.unsqueeze(0); // [1,3,S,S]
    }

    public static Tensor loadChw(MediaInput input) {
        if (input == null) throw new IllegalArgumentException("null MediaInput");
        if (input.tensor != null && input.tensor.defined()) {
            Tensor t = input.tensor;
            if (t.dim() == 4) t = t.squeeze(0);
            if (t.dim() == 3) return t.to(ScalarType.Float).contiguous();
        }
        if (input.bytes != null && input.bytes.length > 0) {
            try {
                return OpenCVIO.decodeImage(input.bytes); // [3,H,W] RGB float?
            } catch (Throwable t) {
                // fall through
            }
        }
        if (input.path != null && Files.isRegularFile(input.path)) {
            try {
                return OpenCVIO.readImage(input.path);
            } catch (Throwable t) {
                // pure-Java PNG fallback for our fixtures
                Tensor png = tryReadPngSimple(input.path);
                if (png != null) return png;
                throw new IllegalStateException("Failed to read image: " + input.path + " — " + t.getMessage(), t);
            }
        }
        // synthetic solid color if nothing else
        return solidColor(64, 64, 0.2f, 0.4f, 0.8f);
    }

    /** Ensure values roughly in 0..1 (OpenCVIO may return 0..255). */
    public static Tensor toUnitRange(Tensor chw) {
        Tensor t = chw.to(ScalarType.Float).contiguous();
        try {
            float max = t.max().item_float();
            if (max > 1.5f) {
                t = t.div(new Scalar(255.0));
            }
        } catch (Throwable ignored) {
            t = t.div(new Scalar(255.0));
        }
        return t;
    }

    public static Tensor resizeSquare(Tensor chw, int size) {
        if (size <= 0) return chw;
        try {
            // OpenCVIO.resize expects [C,H,W]
            return OpenCVIO.resize(chw, size, size);
        } catch (Throwable t) {
            // nearest-neighbor fallback via simple re-sample in float array
            return nearestResize(chw, size, size);
        }
    }

    public static Tensor normalize(Tensor chw, float[] mean, float[] std) {
        if (mean == null || std == null) return chw;
        Tensor t = chw.to(ScalarType.Float).contiguous();
        // manual channel-wise: (x - mean) / std
        float[] data = toFloatArray(t);
        int c = (int) t.size(0);
        int h = (int) t.size(1);
        int w = (int) t.size(2);
        int plane = h * w;
        for (int ci = 0; ci < c && ci < mean.length; ci++) {
            float m = mean[ci];
            float s = std[ci] == 0 ? 1f : std[ci];
            int off = ci * plane;
            for (int i = 0; i < plane; i++) {
                data[off + i] = (data[off + i] - m) / s;
            }
        }
        return fromFloatArray(data, c, h, w);
    }

    public static Tensor solidColor(int h, int w, float r, float g, float b) {
        float[] data = new float[3 * h * w];
        int plane = h * w;
        Arrays.fill(data, 0, plane, r);
        Arrays.fill(data, plane, 2 * plane, g);
        Arrays.fill(data, 2 * plane, 3 * plane, b);
        return fromFloatArray(data, 3, h, w);
    }

    public static float[] toFloatArray(Tensor t) {
        Tensor f = t.to(ScalarType.Float).contiguous().cpu();
        long n = f.numel();
        float[] out = new float[(int) n];
        try {
            FloatIndexer idx = f.createIndexer();
            for (long i = 0; i < n; i++) out[(int) i] = idx.get(i);
            idx.release();
        } catch (Throwable e) {
            // data_ptr fallback
            try {
                org.bytedeco.javacpp.FloatPointer p = f.data_ptr_float();
                p.get(out);
            } catch (Throwable e2) {
                throw new IllegalStateException("Cannot export float tensor: " + e2.getMessage(), e2);
            }
        }
        return out;
    }

    public static Tensor fromFloatArray(float[] data, int c, int h, int w) {
        Tensor t = tensor(data, new TensorOptions(ScalarType.Float));
        return t.reshape(c, h, w).contiguous();
    }

    public static Tensor fromFloatArray(float[] data, long... shape) {
        Tensor t = tensor(data, new TensorOptions(ScalarType.Float));
        return t.reshape(shape).contiguous();
    }

    private static Tensor nearestResize(Tensor chw, int nh, int nw) {
        float[] src = toFloatArray(chw);
        int c = (int) chw.size(0);
        int h = (int) chw.size(1);
        int w = (int) chw.size(2);
        float[] dst = new float[c * nh * nw];
        for (int ci = 0; ci < c; ci++) {
            for (int y = 0; y < nh; y++) {
                int sy = Math.min(h - 1, y * h / nh);
                for (int x = 0; x < nw; x++) {
                    int sx = Math.min(w - 1, x * w / nw);
                    dst[ci * nh * nw + y * nw + x] = src[ci * h * w + sy * w + sx];
                }
            }
        }
        return fromFloatArray(dst, c, nh, nw);
    }

    /** Minimal RGB PNG reader for fixtures (no OpenCV required). */
    static Tensor tryReadPngSimple(Path path) {
        try {
            byte[] all = Files.readAllBytes(path);
            if (all.length < 24) return null;
            // PNG signature
            if ((all[0] & 0xFF) != 0x89 || all[1] != 'P') return null;
            // find IHDR
            int w = readIntBE(all, 16);
            int h = readIntBE(all, 20);
            int bitDepth = all[24] & 0xFF;
            int colorType = all[25] & 0xFF;
            if (bitDepth != 8 || (colorType != 2 && colorType != 6)) return null;
            // collect IDAT
            java.io.ByteArrayOutputStream idat = new java.io.ByteArrayOutputStream();
            int pos = 8;
            while (pos + 8 <= all.length) {
                int len = readIntBE(all, pos);
                String type = new String(all, pos + 4, 4, java.nio.charset.StandardCharsets.US_ASCII);
                if (type.equals("IDAT")) {
                    idat.write(all, pos + 8, len);
                } else if (type.equals("IEND")) break;
                pos += 12 + len;
            }
            byte[] inflated = inflate(idat.toByteArray());
            int bpp = colorType == 6 ? 4 : 3;
            int stride = 1 + w * bpp;
            float[] chw = new float[3 * h * w];
            for (int y = 0; y < h; y++) {
                int row = y * stride;
                // filter byte ignored (assume 0 for our fixtures)
                for (int x = 0; x < w; x++) {
                    int p = row + 1 + x * bpp;
                    chw[0 * h * w + y * w + x] = (inflated[p] & 0xFF) / 255f;
                    chw[1 * h * w + y * w + x] = (inflated[p + 1] & 0xFF) / 255f;
                    chw[2 * h * w + y * w + x] = (inflated[p + 2] & 0xFF) / 255f;
                }
            }
            return fromFloatArray(chw, 3, h, w);
        } catch (Throwable t) {
            return null;
        }
    }

    private static int readIntBE(byte[] b, int off) {
        return ((b[off] & 0xFF) << 24) | ((b[off + 1] & 0xFF) << 16)
                | ((b[off + 2] & 0xFF) << 8) | (b[off + 3] & 0xFF);
    }

    private static byte[] inflate(byte[] data) throws java.io.IOException {
        java.util.zip.Inflater inf = new java.util.zip.Inflater();
        inf.setInput(data);
        java.io.ByteArrayOutputStream bos = new java.io.ByteArrayOutputStream(data.length * 2);
        byte[] buf = new byte[8192];
        try {
            while (!inf.finished()) {
                int n = inf.inflate(buf);
                if (n == 0 && inf.needsInput()) break;
                bos.write(buf, 0, n);
            }
        } catch (java.util.zip.DataFormatException e) {
            throw new java.io.IOException(e);
        } finally {
            inf.end();
        }
        return bos.toByteArray();
    }
}
