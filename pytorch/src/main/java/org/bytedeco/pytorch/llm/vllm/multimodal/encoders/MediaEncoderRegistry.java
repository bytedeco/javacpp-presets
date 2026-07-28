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

import org.bytedeco.pytorch.llm.vllm.multimodal.MediaType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Discover and load real multimodal encoders from a local {@code models/} tree.
 *
 * <p>Default directory names (HF cache style with {@code __}):
 * <ul>
 *   <li>image: {@code facebook__dinov2-small}, {@code openai__clip-vit-base-patch32},
 *       {@code HuggingFaceTB__SmolVLM-256M-Instruct}</li>
 *   <li>VL towers: {@code Qwen__Qwen3-VL-2B-Instruct-FP8} (qwen3vl / qwen-vl),
 *       {@code deepseek-ai__deepseek-vl-1.3b-chat} or {@code google__siglip-base-patch16-224}</li>
 *   <li>audio / ASR: {@code openai__whisper-tiny}</li>
 *   <li>derived wrappers: {@code video}, {@code ocr}, {@code asr}</li>
 * </ul>
 */
public final class MediaEncoderRegistry implements AutoCloseable {

    private final Map<String, MediaEncoder> byKey = new LinkedHashMap<>();
    private final List<MediaEncoder> imageEncoders = new ArrayList<>();
    private final List<MediaEncoder> audioEncoders = new ArrayList<>();
    private final List<MediaEncoder> videoEncoders = new ArrayList<>();
    private final List<MediaEncoder> ocrEncoders = new ArrayList<>();
    private final List<String> loadLog = new ArrayList<>();

    public MediaEncoderRegistry() {}

    /**
     * Load all known encoders under {@code modelsRoot} (best-effort; failures are logged).
     */
    public static MediaEncoderRegistry loadDefault(Path modelsRoot) {
        MediaEncoderRegistry reg = new MediaEncoderRegistry();
        if (modelsRoot == null || !Files.isDirectory(modelsRoot)) {
            reg.loadLog.add("models root missing: " + modelsRoot);
            return reg;
        }
        // Prefer FunctionalVisionEncoder for vision towers (avoids Module/ExpandingArray
        // peer GC SIGSEGV seen with register_module Conv2d on Mac). Fall back to
        // Module-based loaders if functional path fails.
        reg.tryLoad("dinov2", modelsRoot.resolve("facebook__dinov2-small"), MediaType.IMAGE,
                dir -> {
                    try { return FunctionalVisionEncoder.dinov2(dir); }
                    catch (Throwable t) {
                        System.out.println("[MediaEncoderRegistry] functional dinov2 failed: "
                                + t.getMessage() + " — try Module Dinov2Encoder");
                        return Dinov2Encoder.fromDirectory(dir);
                    }
                });
        reg.tryLoad("clip", modelsRoot.resolve("openai__clip-vit-base-patch32"), MediaType.IMAGE,
                dir -> {
                    try { return FunctionalVisionEncoder.clip(dir); }
                    catch (Throwable t) {
                        System.out.println("[MediaEncoderRegistry] functional clip failed: "
                                + t.getMessage() + " — try Module ClipEncoder");
                        return ClipEncoder.fromDirectory(dir);
                    }
                });
        reg.tryLoad("smolvlm", modelsRoot.resolve("HuggingFaceTB__SmolVLM-256M-Instruct"), MediaType.IMAGE,
                dir -> {
                    try { return FunctionalVisionEncoder.smolVision(dir); }
                    catch (Throwable t) {
                        System.out.println("[MediaEncoderRegistry] functional smol failed: "
                                + t.getMessage() + " — try Module SmolVlmEncoder");
                        return SmolVlmEncoder.fromDirectory(dir);
                    }
                });
        // Qwen3-VL / Qwen2-VL vision tower (prefer dedicated Qwen3VLEncoder)
        reg.tryLoad("qwen3vl", modelsRoot.resolve("Qwen__Qwen3-VL-2B-Instruct-FP8"), MediaType.IMAGE,
                Qwen3VLEncoder::fromDirectory);
        reg.tryLoad("qwen2vl", modelsRoot.resolve("Qwen__Qwen2-VL-2B-Instruct"), MediaType.IMAGE,
                dir -> {
                    try { return Qwen3VLEncoder.fromDirectory(dir); }
                    catch (Throwable t1) {
                        try { return FunctionalVisionEncoder.smolVision(dir); }
                        catch (Throwable t2) { return SmolVlmEncoder.fromDirectory(dir); }
                    }
                });
        // DeepSeek-VL: full deepseek-vl-1.3b or Mac-friendly SigLIP-base stand-in
        reg.tryLoad("deepseek-vl", modelsRoot.resolve("deepseek-ai__deepseek-vl-1.3b-chat"), MediaType.IMAGE,
                DeepSeekVLEncoder::fromDirectory);
        reg.tryLoad("siglip", modelsRoot.resolve("google__siglip-base-patch16-224"), MediaType.IMAGE,
                DeepSeekVLEncoder::fromDirectory);
        reg.tryLoad("whisper", modelsRoot.resolve("openai__whisper-tiny"), MediaType.AUDIO,
                WhisperEncoder::fromDirectory);
        reg.attachDerivedEncoders();
        return reg;
    }

    /**
     * Build video / OCR / ASR wrappers from already-loaded base encoders.
     * Safe to call multiple times (idempotent keys).
     */
    public void attachDerivedEncoders() {
        // VL alias priority: qwen3vl > qwen2vl > smolvlm
        MediaEncoder qwen3vl = byKey.get("qwen3vl");
        if (qwen3vl != null && !byKey.containsKey("qwen-vl")) {
            byKey.put("qwen-vl", qwen3vl);
            loadLog.add("qwen-vl: ALIAS qwen3vl");
        }
        MediaEncoder qwen2vl = byKey.get("qwen2vl");
        if (qwen2vl != null && !byKey.containsKey("qwen-vl")) {
            byKey.put("qwen-vl", qwen2vl);
            loadLog.add("qwen-vl: ALIAS qwen2vl");
        }
        MediaEncoder smol = byKey.get("smolvlm");
        if (smol != null && !byKey.containsKey("qwen-vl")) {
            byKey.put("qwen-vl", smol);
            loadLog.add("qwen-vl: ALIAS smolvlm (small VL stand-in for Mac)");
        }
        // DeepSeek-VL alias: full deepseek-vl or siglip stand-in
        MediaEncoder dsvl = byKey.get("deepseek-vl");
        if (dsvl == null && byKey.get("siglip") != null) {
            byKey.put("deepseek-vl", byKey.get("siglip"));
            loadLog.add("deepseek-vl: ALIAS siglip (DeepSeek-VL vision stand-in)");
        }

        MediaEncoder img = preferredImage();
        if (img != null && !byKey.containsKey("video")) {
            VideoEncoder video = VideoEncoder.wrap(img, VideoEncoder.DEFAULT_MAX_FRAMES);
            byKey.put("video", video);
            videoEncoders.add(video);
            loadLog.add("video: OK " + video.encoderName() + " dim=" + video.featureDim()
                    + " maxFrames=" + video.maxFrames());
        } else if (img == null) {
            loadLog.add("video: SKIP no image encoder base");
        }

        if (img != null && !byKey.containsKey("ocr")) {
            // Prefer DINOv2 for OCR texture sensitivity, else preferred image
            MediaEncoder ocrBase = byKey.get("dinov2") != null ? byKey.get("dinov2") : img;
            OcrEncoder ocr = OcrEncoder.wrap(ocrBase);
            byKey.put("ocr", ocr);
            ocrEncoders.add(ocr);
            loadLog.add("ocr: OK " + ocr.encoderName() + " dim=" + ocr.featureDim());
        } else if (img == null) {
            loadLog.add("ocr: SKIP no image encoder base");
        }

        MediaEncoder aud = primaryAudio();
        if (aud != null && !byKey.containsKey("asr")) {
            AsrEncoder asr = AsrEncoder.wrap(aud);
            byKey.put("asr", asr);
            // keep under audio list for primaryAudio? No — asr is separate key
            loadLog.add("asr: OK " + asr.encoderName() + " dim=" + asr.featureDim());
        } else if (aud == null) {
            loadLog.add("asr: SKIP no audio encoder base");
        }
    }

    private void tryLoad(String key, Path dir, MediaType type, MediaEncoder.Factory factory) {
        if (dir == null || !Files.isDirectory(dir)) {
            loadLog.add(key + ": SKIP missing dir " + dir);
            return;
        }
        boolean hasWeight = Files.isRegularFile(dir.resolve("model.safetensors"))
                || Files.isRegularFile(dir.resolve("pytorch_model.bin"))
                || Files.isRegularFile(dir.resolve("vision_weights.safetensors"));
        if (!hasWeight) {
            loadLog.add(key + ": SKIP no weights in " + dir.getFileName());
            return;
        }
        try {
            MediaEncoder enc = factory.load(dir);
            byKey.put(key, enc);
            if (type == MediaType.IMAGE) imageEncoders.add(enc);
            else if (type == MediaType.AUDIO) audioEncoders.add(enc);
            else if (type == MediaType.VIDEO) videoEncoders.add(enc);
            loadLog.add(key + ": OK " + enc.encoderName() + " dim=" + enc.featureDim());
        } catch (Throwable t) {
            loadLog.add(key + ": FAIL " + t.getClass().getSimpleName() + ": " + t.getMessage());
            System.out.println("[MediaEncoderRegistry] " + key + " load failed: " + t.getMessage());
            t.printStackTrace(System.out);
        }
    }

    public MediaEncoder get(String key) {
        return byKey.get(key);
    }

    public MediaEncoder primaryImage() {
        return imageEncoders.isEmpty() ? null : imageEncoders.get(0);
    }

    public MediaEncoder primaryAudio() {
        return audioEncoders.isEmpty() ? null : audioEncoders.get(0);
    }

    public MediaEncoder primaryVideo() {
        MediaEncoder v = byKey.get("video");
        if (v != null) return v;
        return videoEncoders.isEmpty() ? null : videoEncoders.get(0);
    }

    public MediaEncoder preferredOcr() {
        MediaEncoder o = byKey.get("ocr");
        if (o != null) return o;
        return ocrEncoders.isEmpty() ? preferredImage() : ocrEncoders.get(0);
    }

    public MediaEncoder preferredAsr() {
        MediaEncoder a = byKey.get("asr");
        if (a != null) return a;
        return primaryAudio();
    }

    /** Prefer CLIP for image if present (joint vision-text), else DINO / SmolVLM / first. */
    public MediaEncoder preferredImage() {
        MediaEncoder clip = byKey.get("clip");
        if (clip != null) return clip;
        MediaEncoder dino = byKey.get("dinov2");
        if (dino != null) return dino;
        MediaEncoder smol = byKey.get("smolvlm");
        if (smol != null) return smol;
        MediaEncoder qvl = byKey.get("qwen-vl");
        if (qvl != null) return qvl;
        return primaryImage();
    }

    /** Prefer dedicated VL tower (Qwen3-VL / Qwen2-VL / DeepSeek-VL / SmolVLM). */
    public MediaEncoder preferredVl() {
        MediaEncoder q3 = byKey.get("qwen3vl");
        if (q3 != null) return q3;
        MediaEncoder q = byKey.get("qwen-vl");
        if (q != null) return q;
        MediaEncoder q2 = byKey.get("qwen2vl");
        if (q2 != null) return q2;
        MediaEncoder ds = byKey.get("deepseek-vl");
        if (ds != null) return ds;
        MediaEncoder s = byKey.get("smolvlm");
        if (s != null) return s;
        MediaEncoder sig = byKey.get("siglip");
        if (sig != null) return sig;
        return preferredImage();
    }

    /** Prefer DeepSeek-VL tower (full or SigLIP stand-in). */
    public MediaEncoder preferredDeepSeekVl() {
        MediaEncoder ds = byKey.get("deepseek-vl");
        if (ds != null) return ds;
        MediaEncoder sig = byKey.get("siglip");
        if (sig != null) return sig;
        return preferredVl();
    }

    public List<MediaEncoder> imageEncoders() {
        return List.copyOf(imageEncoders);
    }

    public List<MediaEncoder> audioEncoders() {
        return List.copyOf(audioEncoders);
    }

    public List<MediaEncoder> videoEncoders() {
        return List.copyOf(videoEncoders);
    }

    public List<MediaEncoder> ocrEncoders() {
        return List.copyOf(ocrEncoders);
    }

    public Map<String, MediaEncoder> all() {
        return Map.copyOf(byKey);
    }

    public List<String> loadLog() {
        return List.copyOf(loadLog);
    }

    public boolean hasImage() {
        return !imageEncoders.isEmpty();
    }

    public boolean hasAudio() {
        return !audioEncoders.isEmpty();
    }

    public boolean hasVideo() {
        return primaryVideo() != null;
    }

    public boolean hasOcr() {
        return preferredOcr() != null;
    }

    public boolean hasAsr() {
        return preferredAsr() != null;
    }

    public void printStatus() {
        System.out.println("[MediaEncoderRegistry] loaded=" + byKey.keySet());
        for (String line : loadLog) {
            System.out.println("  " + line);
        }
    }

    @Override
    public void close() {
        for (MediaEncoder e : byKey.values()) {
            try {
                e.close();
            } catch (Exception ignored) {}
        }
        byKey.clear();
        imageEncoders.clear();
        audioEncoders.clear();
        videoEncoders.clear();
        ocrEncoders.clear();
    }
}
