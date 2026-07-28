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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaInput;
import org.bytedeco.pytorch.llm.vllm.multimodal.MediaType;

import java.util.Objects;

/**
 * OCR-oriented image encoder wrapper.
 *
 * <p>Does not require a dedicated OCR checkpoint: applies document-friendly
 * preprocess (contrast stretch + mild grayscale blend) then runs a real vision
 * tower (DINOv2 / CLIP / SmolVLM). Features are denser than generic image encode
 * so token-id projection varies more for text-heavy images.
 *
 * <p>Use for OCR stress path: image of text / UI / screenshot → continuous
 * features → feature-hash token ids into the text LM.
 */
public final class OcrEncoder implements MediaEncoder {

    /** Higher token density for OCR vs generic image path. */
    public static final int DEFAULT_TOKEN_BUDGET_HINT = 96;

    private final MediaEncoder vision;
    private final String name;
    private final boolean enhanceContrast;

    public OcrEncoder(MediaEncoder vision) {
        this(vision, true);
    }

    public OcrEncoder(MediaEncoder vision, boolean enhanceContrast) {
        this.vision = Objects.requireNonNull(vision, "vision");
        this.enhanceContrast = enhanceContrast;
        this.name = "ocr/" + vision.encoderName();
    }

    public static OcrEncoder wrap(MediaEncoder vision) {
        return new OcrEncoder(vision, true);
    }

    public MediaEncoder visionEncoder() {
        return vision;
    }

    @Override
    public MediaType modality() {
        // OCR is an image specialization; processor routes IMAGE + OCR role.
        return MediaType.IMAGE;
    }

    @Override
    public String encoderName() {
        return name;
    }

    @Override
    public int featureDim() {
        return vision.featureDim();
    }

    @Override
    public boolean supports(MediaInput input) {
        return input != null && (input.type == MediaType.IMAGE || input.type == MediaType.VIDEO);
    }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        if (input == null) {
            return EncoderFeatures.empty(encoderName());
        }
        try {
            MediaInput prepared = enhanceContrast ? preprocessForOcr(input) : input;
            EncoderFeatures base = vision.encode(prepared);
            if (base == null || base.isEmpty()) {
                // retry without preprocess
                base = vision.encode(input);
            }
            if (base == null || base.isEmpty()) {
                return EncoderFeatures.empty(encoderName());
            }
            // Slight re-scale of pooled features to keep OCR path distinct from plain image
            float[] pooled = base.pooled.clone();
            emphasizeTextEnergy(pooled);
            float[][] seq = base.sequence;
            if (seq != null && seq.length > 0) {
                float[][] copy = new float[seq.length][];
                for (int i = 0; i < seq.length; i++) {
                    copy[i] = seq[i] == null ? null : seq[i].clone();
                    if (copy[i] != null) emphasizeTextEnergy(copy[i]);
                }
                seq = copy;
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pooled, seq, encoderName(), ms);
        } catch (Exception e) {
            System.out.println("[OcrEncoder] encode failed: " + e.getMessage());
            return EncoderFeatures.empty(encoderName());
        }
    }

    /**
     * Load image, boost contrast, blend toward grayscale (documents are often mono).
     * Returns a MediaInput carrying a float CHW tensor in 0..255 or 0..1.
     */
    public static MediaInput preprocessForOcr(MediaInput input) {
        try {
            Tensor chw = ImagePreprocess.loadChw(input); // [3,H,W]
            chw = ImagePreprocess.toUnitRange(chw);       // 0..1
            float[] data = ImagePreprocess.toFloatArray(chw);
            int c = (int) chw.size(0);
            int h = (int) chw.size(1);
            int w = (int) chw.size(2);
            int plane = h * w;
            // per-channel min/max for contrast stretch
            float minV = Float.POSITIVE_INFINITY, maxV = Float.NEGATIVE_INFINITY;
            for (float v : data) {
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
            }
            float range = Math.max(1e-6f, maxV - minV);
            float[] out = new float[data.length];
            for (int i = 0; i < plane; i++) {
                float r = data[i];
                float g = c > 1 ? data[plane + i] : r;
                float b = c > 2 ? data[2 * plane + i] : r;
                // contrast stretch
                r = (r - minV) / range;
                g = (g - minV) / range;
                b = (b - minV) / range;
                // luminance
                float y = 0.299f * r + 0.587f * g + 0.114f * b;
                // blend 70% gray / 30% color (keeps some hue for colored UI text)
                float a = 0.70f;
                out[i] = a * y + (1 - a) * r;
                if (c > 1) out[plane + i] = a * y + (1 - a) * g;
                if (c > 2) out[2 * plane + i] = a * y + (1 - a) * b;
            }
            Tensor t = ImagePreprocess.fromFloatArray(out, c, h, w);
            return MediaInput.builder().type(MediaType.IMAGE).tensor(t)
                    .width(w).height(h).build();
        } catch (Throwable t) {
            // fall back to original
            return input;
        }
    }

    /** Amplify mid-band energy so text-like textures hash differently. */
    private static void emphasizeTextEnergy(float[] feat) {
        if (feat == null || feat.length == 0) return;
        double mean = 0;
        for (float v : feat) mean += v;
        mean /= feat.length;
        double var = 0;
        for (float v : feat) {
            double d = v - mean;
            var += d * d;
        }
        var = Math.sqrt(var / feat.length) + 1e-6;
        for (int i = 0; i < feat.length; i++) {
            float z = (float) ((feat[i] - mean) / var);
            // soft-sign emphasis
            feat[i] = (float) (z / (1.0 + Math.abs(z)));
        }
    }

    @Override
    public void close() {
        // vision owned by registry
    }
}
