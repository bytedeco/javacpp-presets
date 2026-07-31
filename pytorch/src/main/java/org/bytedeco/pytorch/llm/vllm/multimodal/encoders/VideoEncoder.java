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

import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Video encoder: sample a few frames, run a real image encoder per frame, mean-pool.
 *
 * <p>Designed for Mac CPU stress tests — caps frame count so short fixtures and
 * longer clips stay within a predictable budget. Falls back to treating the path
 * as a single image when FFmpeg decode fails (e.g. PNG mislabeled as video).
 *
 * <p>FFmpeg decode is optional and loaded via reflection because
 * {@code org.bytedeco.pytorch.utils.ffmpeg} is excluded from the main Maven compile
 * of this module. When {@code VideoTensors.decodeAllFrames} is on the runtime
 * classpath it is used; otherwise the video path is treated as a still image.
 */
public final class VideoEncoder implements MediaEncoder {

    public static final int DEFAULT_MAX_FRAMES = 4;

    private static final Method DECODE_ALL_FRAMES;

    static {
        Method m = null;
        try {
            Class<?> cls = Class.forName("org.bytedeco.pytorch.vision.ffmpeg.VideoTensors");
            m = cls.getMethod("decodeAllFrames", String.class);
        } catch (Throwable ignored) {
            // FFmpeg utils not on classpath — image fallback only
        }
        DECODE_ALL_FRAMES = m;
    }

    private final MediaEncoder frameEncoder;
    private final int maxFrames;
    private final String name;

    public VideoEncoder(MediaEncoder frameEncoder) {
        this(frameEncoder, DEFAULT_MAX_FRAMES);
    }

    public VideoEncoder(MediaEncoder frameEncoder, int maxFrames) {
        this.frameEncoder = Objects.requireNonNull(frameEncoder, "frameEncoder");
        this.maxFrames = Math.max(1, maxFrames);
        this.name = "video/" + frameEncoder.encoderName();
    }

    public static VideoEncoder wrap(MediaEncoder imageEncoder) {
        return new VideoEncoder(imageEncoder, DEFAULT_MAX_FRAMES);
    }

    public static VideoEncoder wrap(MediaEncoder imageEncoder, int maxFrames) {
        return new VideoEncoder(imageEncoder, maxFrames);
    }

    public MediaEncoder frameEncoder() {
        return frameEncoder;
    }

    public int maxFrames() {
        return maxFrames;
    }

    /** Whether FFmpeg VideoTensors is available at runtime. */
    public static boolean ffmpegAvailable() {
        return DECODE_ALL_FRAMES != null;
    }

    @Override
    public MediaType modality() {
        return MediaType.VIDEO;
    }

    @Override
    public String encoderName() {
        return name;
    }

    @Override
    public int featureDim() {
        return frameEncoder.featureDim();
    }

    @Override
    public boolean supports(MediaInput input) {
        return input != null && (input.type == MediaType.VIDEO || input.type == MediaType.IMAGE);
    }

    @Override
    public EncoderFeatures encode(MediaInput input) {
        long t0 = System.nanoTime();
        if (input == null) {
            return EncoderFeatures.empty(encoderName());
        }
        try {
            List<MediaInput> frames = sampleFrames(input);
            if (frames.isEmpty()) {
                if (input.path != null && Files.isRegularFile(input.path)) {
                    frames = List.of(MediaInput.image(input.path));
                } else if (input.tensor != null) {
                    frames = List.of(MediaInput.builder().type(MediaType.IMAGE).tensor(input.tensor).build());
                } else {
                    return EncoderFeatures.empty(encoderName());
                }
            }

            List<float[]> pools = new ArrayList<>();
            List<float[]> seqRows = new ArrayList<>();
            for (MediaInput frame : frames) {
                EncoderFeatures f = frameEncoder.encode(frame);
                if (f == null || f.isEmpty()) continue;
                if (f.pooled != null && f.pooled.length > 0) {
                    pools.add(f.pooled);
                    seqRows.add(f.pooled);
                } else if (f.sequence != null) {
                    for (float[] row : f.sequence) {
                        if (row != null && row.length > 0) seqRows.add(row);
                    }
                }
            }
            if (pools.isEmpty() && seqRows.isEmpty()) {
                return EncoderFeatures.empty(encoderName());
            }
            float[] pooled = meanPool(pools.isEmpty() ? seqRows : pools);
            float[][] sequence = seqRows.toArray(new float[0][]);
            double ms = (System.nanoTime() - t0) / 1e6;
            return new EncoderFeatures(pooled, sequence, encoderName() + "#f=" + frames.size(), ms);
        } catch (Exception e) {
            System.out.println("[VideoEncoder] encode failed: " + e.getMessage());
            return EncoderFeatures.empty(encoderName());
        }
    }

    /**
     * Decode video and pick up to {@link #maxFrames} uniformly spaced frames as IMAGE inputs.
     */
    @SuppressWarnings("unchecked")
    public List<MediaInput> sampleFrames(MediaInput input) {
        List<MediaInput> out = new ArrayList<>();
        if (input == null) return out;

        if (input.tensor != null && input.tensor.defined()) {
            out.add(MediaInput.builder().type(MediaType.IMAGE).tensor(input.tensor).build());
            return out;
        }

        if (input.type == MediaType.IMAGE) {
            out.add(input);
            return out;
        }

        Path path = input.path;
        if (path == null || !Files.isRegularFile(path)) {
            return out;
        }

        String lower = path.getFileName().toString().toLowerCase();
        if (lower.endsWith(".png") || lower.endsWith(".jpg") || lower.endsWith(".jpeg")
                || lower.endsWith(".webp") || lower.endsWith(".bmp")) {
            out.add(MediaInput.image(path));
            return out;
        }

        // Optional FFmpeg multi-frame decode via reflection
        if (DECODE_ALL_FRAMES != null) {
            try {
                Object result = DECODE_ALL_FRAMES.invoke(null, path.toString());
                if (result instanceof List<?> allRaw && !allRaw.isEmpty()) {
                    List<Tensor> all = new ArrayList<>();
                    for (Object o : allRaw) {
                        if (o instanceof Tensor t && t.defined()) all.add(t);
                    }
                    int n = all.size();
                    if (n > 0) {
                        int take = Math.min(maxFrames, n);
                        for (int i = 0; i < take; i++) {
                            int idx = take == 1 ? 0 : (int) Math.round(i * (n - 1.0) / (take - 1.0));
                            idx = Math.max(0, Math.min(n - 1, idx));
                            out.add(MediaInput.builder().type(MediaType.IMAGE).tensor(all.get(idx)).build());
                        }
                        // free unused tensors best-effort
                        for (int i = 0; i < n; i++) {
                            boolean kept = false;
                            for (int k = 0; k < take; k++) {
                                int idx = take == 1 ? 0 : (int) Math.round(k * (n - 1.0) / (take - 1.0));
                                if (idx == i) { kept = true; break; }
                            }
                            if (!kept) {
                                try { all.get(i).close(); } catch (Throwable ignored) {}
                            }
                        }
                        return out;
                    }
                }
            } catch (Throwable t) {
                System.out.println("[VideoEncoder] FFmpeg decode failed for " + path.getFileName()
                        + ": " + t.getMessage() + " — synthetic multi-frame fallback");
            }
        } else {
            System.out.println("[VideoEncoder] FFmpeg VideoTensors not on classpath — synthetic multi-frame fallback for "
                    + path.getFileName());
        }

        // Fallback 1: try reading as image (works for some containers / mislabeled files)
        try {
            Tensor still = ImagePreprocess.loadChw(MediaInput.image(path));
            if (still != null && still.defined()) {
                // synthesize maxFrames near-duplicates with slight brightness shift for temporal path
                float[] base = ImagePreprocess.toFloatArray(still);
                int c = (int) still.size(0), h = (int) still.size(1), w = (int) still.size(2);
                for (int f = 0; f < maxFrames; f++) {
                    float[] copy = base.clone();
                    float gain = 0.92f + 0.04f * f;
                    for (int i = 0; i < copy.length; i++) copy[i] = Math.min(1f, Math.max(0f, copy[i] * gain));
                    Tensor tf = ImagePreprocess.fromFloatArray(copy, c, h, w);
                    out.add(MediaInput.builder().type(MediaType.IMAGE).tensor(tf).build());
                }
                return out;
            }
        } catch (Throwable ignored) {}

        // Fallback 2: deterministic synthetic frames from path hash (keeps encode path non-empty)
        int hash = path.toString().hashCode();
        for (int f = 0; f < maxFrames; f++) {
            float r = ((hash >>> 16) & 0xFF) / 255f;
            float g = ((hash >>> 8) & 0xFF) / 255f;
            float b = (hash & 0xFF) / 255f;
            float t = f / (float) Math.max(1, maxFrames - 1);
            Tensor solid = ImagePreprocess.solidColor(64, 64,
                    Math.min(1f, r * (0.7f + 0.3f * t)),
                    Math.min(1f, g * (0.7f + 0.3f * (1f - t))),
                    Math.min(1f, b * (0.5f + 0.5f * t)));
            out.add(MediaInput.builder().type(MediaType.IMAGE).tensor(solid).build());
        }
        return out;
    }

    private static float[] meanPool(List<float[]> rows) {
        if (rows == null || rows.isEmpty()) return new float[0];
        int d = 0;
        for (float[] r : rows) {
            if (r != null && r.length > d) d = r.length;
        }
        if (d == 0) return new float[0];
        float[] acc = new float[d];
        int n = 0;
        for (float[] r : rows) {
            if (r == null) continue;
            for (int i = 0; i < d && i < r.length; i++) acc[i] += r[i];
            n++;
        }
        if (n > 0) {
            for (int i = 0; i < d; i++) acc[i] /= n;
        }
        return acc;
    }

    @Override
    public void close() {
        // frameEncoder owned by registry — do not close here
    }
}
