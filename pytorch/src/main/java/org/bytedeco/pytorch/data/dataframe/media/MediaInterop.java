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
package org.bytedeco.pytorch.data.dataframe.media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.data.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.data.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.utils.audio.utils.AudioTensors;
import org.bytedeco.pytorch.utils.vision.utils.ImageTensors;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Interoperability between DataFrame multimodal columns and
 * torchvision / torchaudio / torchtext style utilities already present under
 * {@code org.bytedeco.pytorch.utils.vision|audio|text}.
 *
 * <p>Typical flows:
 * <pre>{@code
 * // DataFrame images → torchvision transforms → tensor batch
 * DataFrame df = MultimodalIO.readImages("imgs/");
 * DataFrame aug = MediaInterop.applyVisionTransform(df, "image",
 *     new Transforms.Resize(224, 224));
 * Tensor batch = MediaInterop.toVisionBatch(aug, "image");   // [N,C,H,W] in [0,1]
 *
 * // DataFrame audio → torchaudio waveform list
 * List&lt;Tensor&gt; waves = MediaInterop.toAudioWaveforms(df, "audio");
 *
 * // Round-trip OpenCV tensor ↔ ImageData column
 * DataFrame back = MediaInterop.fromVisionBatch(batch, "image");
 * }</pre>
 */
public final class MediaInterop {

    private MediaInterop() {}

    // ── torchvision ───────────────────────────────────────────────────────

    /**
     * Apply a torchvision-style transform to every non-null image cell.
     * Accepts {@code Transforms.*} instances, {@link Function}, or anything with
     * {@code apply/forward/call(Object)}.
     */
    public static DataFrame applyVisionTransform(DataFrame df, String imageCol, Object transform)
            throws Exception {
        Objects.requireNonNull(df, "df");
        String col = imageCol == null ? "image" : imageCol;
        if (!df.hasColumn(col)) throw new IllegalArgumentException("missing column: " + col);
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, col);
            if (cell instanceof ImageData id) {
                out.set(r, col, MediaBridge.applyImageTransform(id, transform));
            } else if (cell instanceof Tensor t) {
                ImageData id = MediaBridge.tensorToImage(t);
                out.set(r, col, MediaBridge.applyImageTransform(id, transform));
            }
        }
        return out;
    }

    /**
     * Map image column through a pure function {@code ImageData → ImageData}.
     */
    public static DataFrame mapImages(DataFrame df, String imageCol,
                                      Function<ImageData, ImageData> fn) throws Exception {
        Objects.requireNonNull(fn, "fn");
        String col = imageCol == null ? "image" : imageCol;
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, col);
            if (cell instanceof ImageData id) {
                out.set(r, col, fn.apply(id));
            }
        }
        return out;
    }

    /**
     * Common vision preprocess chain: resize → (optional) center-ish crop via resize → to tensor batch.
     * Returns NCHW float [0,1].
     */
    public static Tensor visionPreprocess(DataFrame df, String imageCol, int size) {
        Objects.requireNonNull(df, "df");
        String col = imageCol == null ? "image" : imageCol;
        List<ImageData> imgs = new ArrayList<>();
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = df.get(r, col);
            if (!(cell instanceof ImageData id) || id.getImage() == null) continue;
            ImageData use = id;
            if (size > 0 && (id.getWidth() != size || id.getHeight() != size)) {
                use = id.resize(size, size);
            }
            imgs.add(use);
        }
        if (imgs.isEmpty()) {
            throw new IllegalArgumentException("no decodable images in column " + col);
        }
        return MediaBridge.stackImages(imgs);
    }

    /** Stack image column as NCHW [0,1] without resize. */
    public static Tensor toVisionBatch(DataFrame df, String imageCol) {
        return MultimodalIO.imagesToBatchTensor(df, imageCol);
    }

    /** Inverse of {@link #toVisionBatch}: NCHW tensor → image DataFrame. */
    public static DataFrame fromVisionBatch(Tensor batch, String imageCol) {
        Objects.requireNonNull(batch, "batch");
        String col = imageCol == null ? "image" : imageCol;
        long[] shape = MediaBridgeSizes.sizes(batch);
        DataFrame df = DataFrame.create();
        df.addColumn(col, Column.DType.IMAGE);
        if (shape.length == 3) {
            // single CHW
            int ri = df.addEmptyRow();
            df.set(ri, col, MediaBridge.tensorToImage(batch));
            return df;
        }
        if (shape.length != 4) {
            throw new IllegalArgumentException("expected CHW or NCHW, got rank " + shape.length);
        }
        int n = (int) shape[0];
        for (int i = 0; i < n; i++) {
            Tensor slice = batch.select(0, i);
            int ri = df.addEmptyRow();
            df.set(ri, col, MediaBridge.tensorToImage(slice));
        }
        return df;
    }

    /** Row-wise ImageData → torchvision CHW tensor list (float [0,1]). */
    public static List<Tensor> toVisionTensors(DataFrame df, String imageCol) {
        String col = imageCol == null ? "image" : imageCol;
        List<Tensor> out = new ArrayList<>();
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = df.get(r, col);
            if (cell instanceof ImageData id && id.getImage() != null) {
                out.add(ImageTensors.toTensor(id));
            } else if (cell instanceof Tensor t) {
                out.add(t);
            } else {
                out.add(null);
            }
        }
        return out;
    }

    // ── torchaudio ────────────────────────────────────────────────────────

    /** Apply a function {@code AudioData → AudioData} to every audio cell. */
    public static DataFrame mapAudio(DataFrame df, String audioCol,
                                     Function<AudioData, AudioData> fn) throws Exception {
        Objects.requireNonNull(fn, "fn");
        String col = audioCol == null ? "audio" : audioCol;
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, col);
            if (cell instanceof AudioData ad) {
                out.set(r, col, fn.apply(ad));
            }
        }
        return out;
    }

    /** Resample all audio cells to {@code sampleRate}. */
    public static DataFrame resampleAudio(DataFrame df, String audioCol, int sampleRate) throws Exception {
        return mapAudio(df, audioCol, a -> MediaBridge.resample(a, sampleRate));
    }

    /** Convert all audio cells to mono. */
    public static DataFrame toMonoAudio(DataFrame df, String audioCol) throws Exception {
        return mapAudio(df, audioCol, MediaBridge::toMono);
    }

    /** Row-wise AudioData → torchaudio [C,T] tensors. */
    public static List<Tensor> toAudioWaveforms(DataFrame df, String audioCol) {
        return MultimodalIO.audioToTensors(df, audioCol);
    }

    /** Build DataFrame from torchaudio-style waveforms. */
    public static DataFrame fromAudioWaveforms(List<Tensor> waves, int sampleRate) {
        return MultimodalIO.fromAudioTensors("audio", waves, sampleRate);
    }

    /**
     * Audio preprocess for ASR-style pipelines: mono + resample + optional max duration trim.
     */
    public static DataFrame audioPreprocess(DataFrame df, String audioCol,
                                            int sampleRate, double maxSeconds) throws Exception {
        String col = audioCol == null ? "audio" : audioCol;
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, col);
            if (!(cell instanceof AudioData ad)) continue;
            AudioData a = MediaBridge.toMono(ad);
            a = MediaBridge.resample(a, sampleRate);
            if (maxSeconds > 0 && a.getSamples() != null) {
                int maxSamples = (int) (maxSeconds * sampleRate) * Math.max(1, a.getChannels());
                float[] s = a.getSamples();
                if (s.length > maxSamples) {
                    float[] cut = new float[maxSamples];
                    System.arraycopy(s, 0, cut, 0, maxSamples);
                    a = new AudioData(cut, sampleRate, a.getChannels());
                    a.setPath(ad.getPath());
                    a.setDuration(maxSeconds);
                }
            }
            out.set(r, col, a);
        }
        return out;
    }

    // ── torchtext ─────────────────────────────────────────────────────────

    /**
     * Tokenize a text column with a tokenizer object exposing {@code encode(String)}
     * or {@code tokenize(String)} (returns List&lt;String&gt; or int[]/long[]).
     * Result adds {@code tokens} column (LIST) or reuses {@code outCol}.
     */
    public static DataFrame tokenizeText(DataFrame df, String textCol, String outCol,
                                         Object tokenizer) throws Exception {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(tokenizer, "tokenizer");
        String tc = textCol == null ? "text" : textCol;
        String oc = outCol == null ? "tokens" : outCol;
        DataFrame out = df.copy();
        MultimodalIO.ensureColumn(out, oc, Column.DType.LIST);
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, tc);
            if (cell == null) continue;
            String text = cell.toString();
            Object tokens = invokeTokenizer(tokenizer, text);
            out.set(r, oc, tokens);
        }
        return out;
    }

    private static Object invokeTokenizer(Object tokenizer, String text) {
        for (String m : new String[]{"tokenize", "encode", "encodeToIds", "call", "apply"}) {
            try {
                java.lang.reflect.Method method = tokenizer.getClass().getMethod(m, String.class);
                try {
                    method.setAccessible(true);
                } catch (Exception ignored) {}
                return method.invoke(tokenizer, text);
            } catch (NoSuchMethodException ignored) {
            } catch (Exception e) {
                // IllegalAccess on anonymous classes etc. → try next / fallback
            }
        }
        // fallback: whitespace split (also used when tokenizer is inaccessible)
        if (tokenizer instanceof java.util.function.Function<?, ?> fn) {
            try {
                @SuppressWarnings("unchecked")
                Object r = ((java.util.function.Function<String, Object>) fn).apply(text);
                if (r != null) return r;
            } catch (Exception ignored) {}
        }
        String[] parts = text.trim().isEmpty() ? new String[0] : text.trim().split("\\s+");
        List<String> list = new ArrayList<>(parts.length);
        for (String p : parts) list.add(p);
        return list;
    }

    /**
     * Lowercase + strip a text column (torchtext basic_english style cleanup, lightweight).
     */
    public static DataFrame basicEnglishNormalize(DataFrame df, String textCol) throws Exception {
        String tc = textCol == null ? "text" : textCol;
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, tc);
            if (cell == null) continue;
            String s = cell.toString().toLowerCase(java.util.Locale.ROOT);
            s = s.replaceAll("[^a-z0-9\\s']", " ");
            s = s.replaceAll("\\s+", " ").trim();
            out.set(r, tc, s);
        }
        return out;
    }

    // ── video interop ─────────────────────────────────────────────────────

    /** Explode video → frames then apply vision transform on frames. */
    public static DataFrame videoFramesTransformed(DataFrame df, String videoCol,
                                                   double fps, Object transform) throws Exception {
        DataFrame frames = MultimodalIO.extractVideoFrames(df, videoCol, fps);
        if (transform != null) {
            frames = applyVisionTransform(frames, "frame", transform);
        }
        return frames;
    }

    /** Decode video column (path stubs) via FFmpeg/MediaBridge in place. */
    public static DataFrame decodeVideos(DataFrame df, String videoCol,
                                         MediaBridge.VideoOptions opts) throws Exception {
        String vc = videoCol == null ? "video" : videoCol;
        DataFrame out = df.copy();
        MediaBridge.VideoOptions use = opts == null ? MediaBridge.VideoOptions.defaults() : opts;
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, vc);
            VideoData vid;
            if (cell instanceof VideoData vd) {
                if (vd.getFrames() != null && !vd.getFrames().isEmpty()) continue;
                if (vd.getPath() == null) continue;
                vid = MediaBridge.loadVideo(vd.getPath(), use);
            } else if (cell instanceof String path) {
                vid = MediaBridge.loadVideo(path, use);
            } else {
                continue;
            }
            out.set(r, vc, vid);
        }
        return out;
    }

    // ── OpenCV / FFmpeg direct ────────────────────────────────────────────

    /**
     * Re-decode image column with OpenCV backend (no-op cells that already have pixels
     * unless {@code force=true}).
     */
    public static DataFrame redecodeImagesOpenCv(DataFrame df, String imageCol, boolean force)
            throws Exception {
        String col = imageCol == null ? "image" : imageCol;
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, col);
            String path = null;
            if (cell instanceof ImageData id) {
                if (!force && id.getImage() != null) continue;
                path = id.getPath();
            } else if (cell instanceof String s) {
                path = s;
            }
            if (path == null) continue;
            try {
                out.set(r, col, MediaBridge.loadImageOpenCv(path, false));
            } catch (Exception ignored) {}
        }
        return out;
    }

    /** Convert image column to list of OpenCV Mats (as Object to keep optional dep soft). */
    public static List<Object> toOpenCvMats(DataFrame df, String imageCol) throws Exception {
        String col = imageCol == null ? "image" : imageCol;
        List<Object> mats = new ArrayList<>();
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = df.get(r, col);
            if (cell instanceof ImageData id && id.getImage() != null) {
                mats.add(MediaBridge.imageToMat(id));
            } else {
                mats.add(null);
            }
        }
        return mats;
    }

    /** Waveform tensor utilities re-export. */
    public static Tensor audioDataToTensor(AudioData a) {
        return AudioTensors.toTensor(a);
    }

    public static ImageData tensorToImageData(Tensor t) {
        return ImageTensors.toImageData(t);
    }

    // package-local size helper without depending on TensorBridge internals
    static final class MediaBridgeSizes {
        static long[] sizes(Tensor t) {
            return org.bytedeco.pytorch.data.dataframe.tensor.TensorBridge.sizesOf(t);
        }
    }
}
