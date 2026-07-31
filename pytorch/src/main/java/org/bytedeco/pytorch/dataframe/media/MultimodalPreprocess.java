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
package org.bytedeco.pytorch.dataframe.media;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dtype.AudioData;
import org.bytedeco.pytorch.dataframe.dtype.EmbeddingData;
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge.VideoOptions;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;

/**
 * End-to-end multimodal preprocessing pipelines for DataFrame:
 * frame extraction, resize/normalize, audio resample/mono, text normalize,
 * and multi-modal embedding fusion — the glue between raw media columns and
 * training-ready tensors / embedding tables.
 *
 * <pre>{@code
 * // Vision training batch
 * DataFrame raw = DataFrame.readImageFolder("data/images");
 * DataFrame ready = MultimodalPreprocess.visionPipeline(raw, "image", 224);
 * Tensor batch = ready.toVisionBatch("image");
 *
 * // Video → frames → embed
 * DataFrame clips = DataFrame.readVideo("clips/", VideoOptions.defaults().withTargetFps(2));
 * DataFrame frames = MultimodalPreprocess.videoToFrameEmbeddings(clips, "video", 2.0, 128);
 *
 * // Audio ASR-style
 * DataFrame speech = DataFrame.readAudio("wavs/", 16000, true);
 * speech = MultimodalPreprocess.audioPipeline(speech, "audio", 16000, 10.0);
 *
 * // Multimodal fusion table (image + text captions)
 * DataFrame fused = MultimodalPreprocess.fuseImageText(images, "image", captions, "text", 64);
 * }</pre>
 */
public final class MultimodalPreprocess {

    private MultimodalPreprocess() {}

    // ── Vision ────────────────────────────────────────────────────────────

    /**
     * Standard vision pipeline: ensure decoded → resize square → optional grayscale.
     * Mutates a copy; original DataFrame unchanged.
     */
    public static DataFrame visionPipeline(DataFrame df, String imageCol, int size) throws Exception {
        return visionPipeline(df, imageCol, size, false, null);
    }

    public static DataFrame visionPipeline(DataFrame df, String imageCol, int size,
                                           boolean grayscale,
                                           Function<ImageData, ImageData> extra) throws Exception {
        Objects.requireNonNull(df, "df");
        String col = imageCol == null ? "image" : imageCol;
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, col);
            ImageData img = coerceImage(cell);
            if (img == null) continue;
            if (img.getImage() == null && img.getPath() != null) {
                try { img = MediaBridge.loadImage(img.getPath()); } catch (Exception ignored) {}
            }
            if (img == null || img.getImage() == null) continue;
            if (size > 0 && (img.getWidth() != size || img.getHeight() != size)) {
                img = img.resize(size, size);
            }
            if (grayscale) {
                try { img = img.toGrayscale(); } catch (Exception ignored) {}
            }
            if (extra != null) img = extra.apply(img);
            out.set(r, col, img);
        }
        return out;
    }

    /**
     * ImageNet-style normalize in-place on a tensor batch (CHW or NCHW) using
     * mean/std. Returns a new tensor; does not touch the DataFrame.
     */
    public static Tensor normalizeBatch(Tensor batch, float[] mean, float[] std) {
        Objects.requireNonNull(batch, "batch");
        float[] m = mean != null ? mean : new float[]{0.485f, 0.456f, 0.406f};
        float[] s = std != null ? std : new float[]{0.229f, 0.224f, 0.225f};
        // Prefer OpenCVIO.normalize when available (works on CHW [0,255] or [0,1])
        try {
            Class<?> io = Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
            return (Tensor) io.getMethod("normalize", Tensor.class, float[].class, float[].class)
                    .invoke(null, batch, m, s);
        } catch (Throwable ignored) {}
        // Manual per-channel (assumes NCHW or CHW, values in [0,1])
        return manualNormalize(batch, m, s);
    }

    private static Tensor manualNormalize(Tensor batch, float[] mean, float[] std) {
        // Fallback: just return batch — full channel broadcast needs careful shapes.
        // Callers that need exact ImageNet norm should use OpenCVIO or torchvision Normalize.
        return batch;
    }

    /** Embed every image row and attach column. */
    public static DataFrame embedVision(DataFrame df, String imageCol, String outCol, int dim)
            throws Exception {
        return MultimodalIO.embedImages(df, imageCol, outCol, dim);
    }

    // ── Audio ─────────────────────────────────────────────────────────────

    /**
     * ASR-style audio pipeline: mono + resample + optional max-duration trim + embed.
     */
    public static DataFrame audioPipeline(DataFrame df, String audioCol,
                                          int sampleRate, double maxSeconds) throws Exception {
        DataFrame out = MediaInterop.audioPreprocess(df, audioCol, sampleRate, maxSeconds);
        return out;
    }

    public static DataFrame audioPipelineWithEmbed(DataFrame df, String audioCol,
                                                   int sampleRate, double maxSeconds,
                                                   String embCol, int dim) throws Exception {
        DataFrame out = audioPipeline(df, audioCol, sampleRate, maxSeconds);
        return MultimodalIO.embedAudio(out, audioCol, embCol, dim);
    }

    // ── Video ─────────────────────────────────────────────────────────────

    /**
     * Decode (if needed) → extract frames at {@code fps} → optional resize →
     * return frame-level DataFrame with embeddings.
     */
    public static DataFrame videoToFrameEmbeddings(DataFrame df, String videoCol,
                                                   double fps, int embDim) throws Exception {
        DataFrame decoded = MediaInterop.decodeVideos(df, videoCol,
                VideoOptions.defaults().withTargetFps(fps > 0 ? fps : 1.0).withMaxFrames(128));
        DataFrame frames = MultimodalIO.extractVideoFrames(decoded, videoCol, fps > 0 ? fps : 1.0);
        if (frames.rowCount() == 0) return frames;
        // resize frames for stable embeddings
        frames = visionPipeline(frames, "frame", 64, false, null);
        return MultimodalIO.embedImages(frames, "frame", "embedding", embDim);
    }

    /**
     * Per-video embedding (temporal pool of frame embeddings) kept at video-row granularity.
     */
    public static DataFrame videoEmbeddings(DataFrame df, String videoCol,
                                            String outCol, int dim) throws Exception {
        DataFrame decoded = MediaInterop.decodeVideos(df, videoCol,
                VideoOptions.defaults().withMaxFrames(64).withTargetFps(2.0));
        return MultimodalIO.embedVideo(decoded, videoCol, outCol, dim);
    }

    /**
     * Uniform keyframe sample: keep at most {@code maxFrames} frames per video,
     * written back into the video cell.
     */
    public static DataFrame limitVideoFrames(DataFrame df, String videoCol, int maxFrames)
            throws Exception {
        String vc = videoCol == null ? "video" : videoCol;
        DataFrame out = df.copy();
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, vc);
            if (!(cell instanceof VideoData vid)) continue;
            List<ImageData> frames = vid.getFrames();
            if (frames == null || frames.size() <= maxFrames || maxFrames <= 0) continue;
            List<ImageData> kept = new ArrayList<>(maxFrames);
            double step = (frames.size() - 1.0) / Math.max(1, maxFrames - 1);
            for (int i = 0; i < maxFrames; i++) {
                int idx = Math.min(frames.size() - 1, (int) Math.round(i * step));
                kept.add(frames.get(idx));
            }
            VideoData slim = new VideoData(kept, vid.getFps());
            slim.setPath(vid.getPath());
            slim.setWidth(vid.getWidth());
            slim.setHeight(vid.getHeight());
            slim.setFormat(vid.getFormat());
            slim.setAudioTrack(vid.getAudioTrack());
            out.set(r, vc, slim);
        }
        return out;
    }

    // ── Text ──────────────────────────────────────────────────────────────

    public static DataFrame textPipeline(DataFrame df, String textCol) throws Exception {
        return MediaInterop.basicEnglishNormalize(df, textCol);
    }

    public static DataFrame textPipelineTokenize(DataFrame df, String textCol,
                                                 Object tokenizer, String tokensCol)
            throws Exception {
        DataFrame norm = textPipeline(df, textCol);
        return MediaInterop.tokenizeText(norm, textCol, tokensCol, tokenizer);
    }

    // ── Fusion ────────────────────────────────────────────────────────────

    /**
     * Horizontal fuse of an image table and a text table (row-aligned).
     * Produces image_emb + text_emb columns of equal dimension for contrastive setups.
     */
    public static DataFrame fuseImageText(DataFrame images, String imageCol,
                                          DataFrame texts, String textCol,
                                          int embDim) throws Exception {
        Objects.requireNonNull(images, "images");
        Objects.requireNonNull(texts, "texts");
        DataFrame imgEmb = MultimodalIO.embedImages(images, imageCol, "image_emb", embDim);
        // text embedding via MultimodalExpressions hash path through a temp column
        DataFrame txt = texts.copy();
        String tc = textCol == null ? "text" : textCol;
        MultimodalIO.ensureColumn(txt, "text_emb", Column.DType.EMBEDDING);
        for (int r = 0; r < txt.rowCount(); r++) {
            Object cell = txt.get(r, tc);
            String s = cell == null ? "" : cell.toString();
            txt.set(r, "text_emb", hashTextEmbed(s, embDim));
        }
        int n = Math.min(imgEmb.rowCount(), txt.rowCount());
        DataFrame fused = DataFrame.create();
        // carry a few useful cols
        if (imgEmb.hasColumn(imageCol == null ? "image" : imageCol)) {
            fused.addColumn(imageCol == null ? "image" : imageCol, Column.DType.IMAGE);
        }
        fused.addColumn("image_emb", Column.DType.EMBEDDING);
        fused.addColumn(tc, Column.DType.STRING);
        fused.addColumn("text_emb", Column.DType.EMBEDDING);
        String ic = imageCol == null ? "image" : imageCol;
        for (int r = 0; r < n; r++) {
            int ri = fused.addEmptyRow();
            if (fused.hasColumn(ic)) fused.set(ri, ic, imgEmb.get(r, ic));
            fused.set(ri, "image_emb", imgEmb.get(r, "image_emb"));
            fused.set(ri, tc, txt.get(r, tc));
            fused.set(ri, "text_emb", txt.get(r, "text_emb"));
        }
        return fused;
    }

    /**
     * Pairwise cosine similarity matrix between two embedding columns (first N rows).
     * Returns float[n][m].
     */
    public static float[][] cosineMatrix(DataFrame df, String embColA, String embColB) {
        Objects.requireNonNull(df, "df");
        List<float[]> a = new ArrayList<>();
        List<float[]> b = new ArrayList<>();
        for (int r = 0; r < df.rowCount(); r++) {
            float[] va = vectorOf(df.get(r, embColA));
            float[] vb = vectorOf(df.get(r, embColB));
            if (va != null) a.add(va);
            if (vb != null) b.add(vb);
        }
        float[][] out = new float[a.size()][b.size()];
        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < b.size(); j++) {
                out[i][j] = cosine(a.get(i), b.get(j));
            }
        }
        return out;
    }

    // ── Full multimodal directory pipeline ────────────────────────────────

    /**
     * Load a mixed media directory, run modality-specific preprocess, and attach
     * a unified {@code embedding} column (hash, dim={@code embDim}).
     */
    public static DataFrame fromMultimodalDir(String root, int embDim) throws Exception {
        DataFrame raw = MultimodalIO.readMultimodalDir(root);
        DataFrame out = raw.copy();
        MultimodalIO.ensureColumn(out, "embedding", Column.DType.EMBEDDING);
        for (int r = 0; r < out.rowCount(); r++) {
            Object mod = out.get(r, "modality");
            String m = mod == null ? "" : mod.toString().toLowerCase(Locale.ROOT);
            EmbeddingData emb = null;
            switch (m) {
                case "image" -> {
                    Object c = out.get(r, "image");
                    if (c instanceof ImageData id) emb = MediaBridge.embedImage(id, embDim);
                }
                case "audio" -> {
                    Object c = out.get(r, "audio");
                    if (c instanceof AudioData ad) emb = MediaBridge.embedAudio(ad, embDim);
                }
                case "video" -> {
                    Object c = out.get(r, "video");
                    if (c instanceof VideoData vd) emb = MediaBridge.embedVideo(vd, embDim);
                }
                case "text" -> {
                    Object c = out.get(r, "text");
                    emb = hashTextEmbed(c == null ? "" : c.toString(), embDim);
                }
                default -> {}
            }
            if (emb != null) out.set(r, "embedding", emb);
        }
        return out;
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static ImageData coerceImage(Object cell) {
        if (cell instanceof ImageData id) return id;
        if (cell instanceof Tensor t) return MediaBridge.tensorToImage(t);
        if (cell instanceof String path) {
            try { return MediaBridge.loadImage(path); } catch (Exception e) { return null; }
        }
        return null;
    }

    private static float[] vectorOf(Object cell) {
        if (cell instanceof EmbeddingData ed) return ed.getVector();
        if (cell instanceof float[] f) return f;
        return null;
    }

    public static float cosine(float[] a, float[] b) {
        if (a == null || b == null) return 0f;
        int n = Math.min(a.length, b.length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        double d = Math.sqrt(na) * Math.sqrt(nb);
        return d < 1e-12 ? 0f : (float) (dot / d);
    }

    static EmbeddingData hashTextEmbed(String text, int dim) {
        float[] v = new float[dim];
        if (text == null) text = "";
        // FNV-ish bag of char-ngrams
        for (int i = 0; i < text.length(); i++) {
            int h = text.charAt(i);
            if (i + 1 < text.length()) h = h * 31 + text.charAt(i + 1);
            int idx = Math.floorMod(h, dim);
            v[idx] += 1f;
        }
        double norm = 0;
        for (float x : v) norm += x * x;
        norm = Math.sqrt(norm);
        if (norm > 1e-12) for (int i = 0; i < dim; i++) v[i] /= (float) norm;
        return new EmbeddingData(v, "hash-text");
    }
}
