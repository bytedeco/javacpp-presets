/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
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
package org.bytedeco.pytorch.utils.opencv;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Enterprise multimodal OpenCV ops on top of {@link OpenCVIO} / javacpp-opencv.
 *
 * <p>Designed for full-pipeline CV that multi-modal systems need (Daft, Meta
 * torchvision, Google/ByteDance VLM preprocessors):
 * <ul>
 *   <li>VLM image/video preprocess — letterbox, short-side resize + center crop, ImageNet norm</li>
 *   <li>OCR / document — CLAHE, adaptive threshold, denoise</li>
 *   <li>Dedup / retrieval — average hash + hamming</li>
 *   <li>Motion / video — Farneback optical flow, frame-diff magnitude</li>
 *   <li>Augmentation pack — flip / rotate / blur / brightness for training</li>
 *   <li>Batch stack helpers for DataLoader-style tensors</li>
 * </ul>
 *
 * <pre>{@code
 * // VLM single image → [3,224,224] normalized
 * Tensor x = OpenCVOps.preprocessImagenet("photo.jpg", 224);
 *
 * // Video frames → letterboxed batch [N,3,336,336]
 * Tensor batch = OpenCVOps.preprocessVlmFrames(frames, 336);
 *
 * // OCR binarize
 * Tensor bin = OpenCVOps.ocrBinarize(OpenCVIO.readImage("doc.png"));
 *
 * // Near-duplicate
 * long h1 = OpenCVOps.ahash(img1), h2 = OpenCVOps.ahash(img2);
 * boolean near = OpenCVOps.hamming(h1, h2) <= 5;
 * }</pre>
 */
public final class OpenCVOps {

    /** ImageNet mean (RGB, for tensors in {@code [0,1]}). */
    public static final float[] IMAGENET_MEAN = {0.485f, 0.456f, 0.406f};
    /** ImageNet std (RGB, for tensors in {@code [0,1]}). */
    public static final float[] IMAGENET_STD  = {0.229f, 0.224f, 0.225f};

    /** CLIP mean/std (OpenAI CLIP / many VLMs). */
    public static final float[] CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
    public static final float[] CLIP_STD  = {0.26862954f, 0.26130258f, 0.27577711f};

    private OpenCVOps() {}

    // ═══════════════════════════════════════════════════════════════════════
    // VLM / classification preprocess
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Classic torchvision classification preprocess:
     * short-side resize → center-crop square → scale to [0,1] → ImageNet normalize.
     *
     * @param path image path
     * @param size square side (e.g. 224)
     * @return {@code [3,size,size]} float32 normalized
     */
    public static Tensor preprocessImagenet(String path, int size) {
        Tensor img = OpenCVIO.readImage(path); // [3,H,W] in [0,255]
        return preprocessImagenet(img, size);
    }

    public static Tensor preprocessImagenet(Tensor img255, int size) {
        Objects.requireNonNull(img255, "img");
        int s = size > 0 ? size : 224;
        Tensor cropped = OpenCVIO.resizeShortCenterCrop(img255, s);
        Tensor unit = scaleToUnit(cropped);
        return OpenCVIO.normalize(unit, IMAGENET_MEAN, IMAGENET_STD);
    }

    /**
     * CLIP / many open-VLM preprocess: short-side + center crop + CLIP mean/std.
     */
    public static Tensor preprocessClip(Tensor img255, int size) {
        int s = size > 0 ? size : 224;
        Tensor cropped = OpenCVIO.resizeShortCenterCrop(img255, s);
        return OpenCVIO.normalize(scaleToUnit(cropped), CLIP_MEAN, CLIP_STD);
    }

    /**
     * YOLO / DETR-style letterbox to square (pad 114) then optional [0,1] scale.
     * Does <em>not</em> apply ImageNet norm (detectors usually stay in 0-1 or 0-255).
     */
    public static Tensor preprocessLetterbox(Tensor img255, int size, boolean toUnit) {
        int s = size > 0 ? size : 640;
        Tensor boxed = OpenCVIO.letterbox(img255, s, s, 114.0);
        return toUnit ? scaleToUnit(boxed) : boxed;
    }

    /**
     * Batch-letterbox a list of video/key frames for VLM video towers
     * (Qwen2-VL / LLaVA-Video / InternVL style fixed canvas).
     *
     * @return {@code [N,3,out,out]} in {@code [0,255]} (caller may scale/normalize)
     */
    public static Tensor preprocessVlmFrames(List<Tensor> frames, int size) {
        int s = size > 0 ? size : 336;
        return OpenCVIO.batchLetterboxStack(frames, s, s);
    }

    /**
     * Full VLM pack: letterbox → [0,1] → optional CLIP norm → stacked batch.
     *
     * @param applyClipNorm if true apply {@link #CLIP_MEAN}/{@link #CLIP_STD}
     */
    public static Tensor preprocessVlmFramesNorm(List<Tensor> frames, int size, boolean applyClipNorm) {
        Tensor batch = preprocessVlmFrames(frames, size); // [N,3,H,W] 0-255
        if (batch.size(0) == 0) return batch;
        // scale per-frame then optionally normalize
        long n = batch.size(0);
        List<Tensor> out = new ArrayList<>((int) n);
        for (int i = 0; i < n; i++) {
            Tensor f = scaleToUnit(batch.select(0, i));
            if (applyClipNorm) f = OpenCVIO.normalize(f, CLIP_MEAN, CLIP_STD);
            out.add(f);
        }
        return torch.stack(new TensorVector(out.toArray(new Tensor[0])));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // OCR / document enhancement
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Document / OCR binarization pipeline: grayscale → CLAHE → bilateral denoise
     * → adaptive threshold. Returns {@code [1,H,W]} 0/255.
     */
    public static Tensor ocrBinarize(Tensor img255) {
        Tensor gray = OpenCVIO.toGrayscale(img255);
        Tensor eq = OpenCVIO.clahe(gray, 2.0, 8);
        Tensor den = OpenCVIO.bilateralFilter(eq, 7, 50, 50);
        return OpenCVIO.adaptiveThreshold(den, 11, 2);
    }

    /** Low-light enhance: CLAHE on color (Lab L-channel). */
    public static Tensor enhanceLowLight(Tensor img255) {
        return OpenCVIO.clahe(img255, 3.0, 8);
    }

    /** Denoise pack: bilateral then light unsharp via blend with gaussian. */
    public static Tensor denoise(Tensor img255, int strength) {
        int d = strength <= 0 ? 7 : Math.min(15, strength | 1);
        return OpenCVIO.bilateralFilter(img255, d, 50 + strength * 5.0, 50 + strength * 5.0);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Dedup / perceptual hash
    // ═══════════════════════════════════════════════════════════════════════

    public static long ahash(Tensor img255) {
        return OpenCVIO.averageHash(img255);
    }

    public static long ahash(String path) {
        return ahash(OpenCVIO.readImage(path));
    }

    public static int hamming(long a, long b) {
        return OpenCVIO.hamming64(a, b);
    }

    /** True if two images are near-duplicates under aHash (default threshold 5). */
    public static boolean isNearDuplicate(Tensor a, Tensor b, int maxDistance) {
        return hamming(ahash(a), ahash(b)) <= Math.max(0, maxDistance);
    }

    public static boolean isNearDuplicate(Tensor a, Tensor b) {
        return isNearDuplicate(a, b, 5);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Motion / video analytics
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Dense Farneback flow {@code [2,H,W]} between consecutive frames.
     */
    public static Tensor opticalFlow(Tensor prev, Tensor next) {
        return OpenCVIO.opticalFlowFarneback(prev, next);
    }

    /**
     * Mean absolute flow magnitude (scalar motion energy). Useful as a
     * cheap scene-activity / keyframe score without full decode analytics.
     */
    public static double meanFlowMagnitude(Tensor prev, Tensor next) {
        Tensor flow = opticalFlow(prev, next); // [2,H,W]
        Tensor dx = flow.select(0, 0);
        Tensor dy = flow.select(0, 1);
        // mag = sqrt(dx^2 + dy^2); mean over HxW
        Tensor mag = torch.sqrt(dx.mul(dx).add(dy.mul(dy)));
        return mag.mean().item_float();
    }

    /**
     * Absolute frame difference mean (faster motion proxy than optical flow).
     * Both inputs {@code [C,H,W]} 0-255; returns mean |a-b| in 0-255 units.
     */
    public static double frameDiffEnergy(Tensor a, Tensor b) {
        Tensor ga = OpenCVIO.toGrayscale(a);
        Tensor gb = OpenCVIO.toGrayscale(b);
        Tensor diff = ga.sub(gb).abs();
        return diff.mean().item_float();
    }

    /**
     * Pairwise consecutive flow energy for a frame list (length N-1).
     * Empty / single-frame → empty list.
     */
    public static List<Double> motionProfile(List<Tensor> frames) {
        List<Double> out = new ArrayList<>();
        if (frames == null || frames.size() < 2) return out;
        for (int i = 1; i < frames.size(); i++) {
            out.add(frameDiffEnergy(frames.get(i - 1), frames.get(i)));
        }
        return out;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Augmentation pack (training)
    // ═══════════════════════════════════════════════════════════════════════

    /** Horizontal flip. */
    public static Tensor hflip(Tensor t) { return OpenCVIO.hflip(t); }

    /** Vertical flip. */
    public static Tensor vflip(Tensor t) { return OpenCVIO.vflip(t); }

    /** Rotate degrees (canvas preserved). */
    public static Tensor rotate(Tensor t, double deg) { return OpenCVIO.rotate(t, deg); }

    /** Gaussian blur augmentation. */
    public static Tensor blur(Tensor t, int ksize) { return OpenCVIO.gaussianBlur(t, ksize); }

    /** Brightness/contrast jitter: alpha∈contrast, beta∈brightness offset. */
    public static Tensor colorJitter(Tensor t, double contrast, double brightness) {
        return OpenCVIO.adjustBrightnessContrast(t, contrast, brightness);
    }

    /**
     * Apply a small fixed augmentation recipe (deterministic demo / baseline):
     * optional hflip, light blur, slight contrast.
     */
    public static Tensor augmentBasic(Tensor t, boolean flip, boolean blur, boolean contrast) {
        Tensor x = t;
        if (flip) x = OpenCVIO.hflip(x);
        if (blur) x = OpenCVIO.gaussianBlur(x, 3);
        if (contrast) x = OpenCVIO.adjustBrightnessContrast(x, 1.1, 5);
        return x;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Edges / morphology facade
    // ═══════════════════════════════════════════════════════════════════════

    public static Tensor edges(Tensor t) { return OpenCVIO.canny(t); }

    public static Tensor sobel(Tensor t) { return OpenCVIO.sobel(t); }

    public static Tensor sharpen(Tensor t) {
        // unsharp-ish: blend original with blurred inverse
        Tensor blurred = OpenCVIO.gaussianBlur(t, 3);
        // out = t + (t - blurred) = 2t - blurred
        return OpenCVIO.blend(t, blurred, 1.5); // alpha>1 exaggerates; clamp via convert in blend path
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Batch / stack
    // ═══════════════════════════════════════════════════════════════════════

    /** Resize all to HxW then stack {@code [N,C,H,W]}. */
    public static Tensor stackResized(List<Tensor> images, int height, int width) {
        List<Tensor> r = OpenCVIO.batchResize(images, height, width);
        if (r.isEmpty()) {
            return torch.empty(new long[]{0, 3, height, width},
                    new TensorOptions(ScalarType.Float), null);
        }
        return torch.stack(new TensorVector(r.toArray(new Tensor[0])));
    }

    /** Scale {@code [0,255]} → {@code [0,1]} (clone). */
    public static Tensor scaleToUnit(Tensor img255) {
        return img255.div(new Scalar(255.0f));
    }

    /** Scale {@code [0,1]} → {@code [0,255]} (clone). */
    public static Tensor scaleTo255(Tensor imgUnit) {
        return imgUnit.mul(new Scalar(255.0f));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Capability / recipe catalog (for DataFrame.media / docs)
    // ═══════════════════════════════════════════════════════════════════════

    public static Map<String, Object> capabilities() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("imagenetMean", IMAGENET_MEAN);
        m.put("clipMean", CLIP_MEAN);
        m.put("pipelines", List.of(
                "preprocessImagenet", "preprocessClip", "preprocessLetterbox",
                "preprocessVlmFrames", "preprocessVlmFramesNorm",
                "ocrBinarize", "enhanceLowLight", "denoise",
                "ahash", "isNearDuplicate",
                "opticalFlow", "frameDiffEnergy", "motionProfile",
                "augmentBasic", "edges", "stackResized"
        ));
        m.put("opencvIoExtended", List.of(
                "vflip", "rotate", "gaussianBlur", "medianBlur", "bilateralFilter",
                "canny", "sobel", "dilate", "erode", "morphologyOpen", "morphologyClose",
                "equalizeHist", "clahe", "toHsv", "letterbox", "pad", "centerCrop",
                "resizeShortCenterCrop", "adjustBrightnessContrast", "threshold",
                "adaptiveThreshold", "blend", "batchLetterboxStack", "averageHash",
                "opticalFlowFarneback"
        ));
        return m;
    }
}
