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
import org.bytedeco.pytorch.dataframe.dtype.ImageData;
import org.bytedeco.pytorch.dataframe.dtype.VideoData;
import org.bytedeco.pytorch.dataframe.media.MediaBridge.AudioOptions;
import org.bytedeco.pytorch.dataframe.media.MediaBridge.ImageOptions;
import org.bytedeco.pytorch.dataframe.media.MediaBridge.VideoOptions;

import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * High-level multimodal batch I/O for {@link DataFrame}.
 *
 * <p>Wraps {@link MediaBridge} (OpenCV / FFmpeg / pure-Java) and dataset-folder
 * layouts used by torchvision / torchaudio / torchtext:
 * <ul>
 *   <li>{@code readImages / readAudio / readVideo} — flat dir or glob</li>
 *   <li>{@code readImageFolder} — {@code root/class_x/*.jpg} → path, image, label, class</li>
 *   <li>{@code readAudioFolder} — {@code root/class_x/*.wav} → path, audio, label, class</li>
 *   <li>{@code readTextFolder} — {@code root/class_x/*.txt} → path, text, label, class</li>
 *   <li>{@code fromOpenCV / fromFFmpeg / fromVision / fromAudio / fromText} — in-memory builders</li>
 *   <li>{@code extractVideoFrames} — explode video column into per-frame image rows</li>
 *   <li>{@code embed*Column} — batch hash / model embeddings</li>
 * </ul>
 */
public final class MultimodalIO {

    private MultimodalIO() {}

    public static final String[] IMAGE_EXTS = {
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff", ".tif"
    };
    public static final String[] AUDIO_EXTS = {
            ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".wave"
    };
    public static final String[] VIDEO_EXTS = {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"
    };
    public static final String[] TEXT_EXTS = {
            ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl"
    };

    // ── Flat directory / glob loaders ─────────────────────────────────────

    /**
     * Batch-load images with OpenCV preferred.
     * Columns: {@code path}, {@code image}, plus optional {@code width}/{@code height}/{@code channels}.
     */
    public static DataFrame readImages(String pathOrGlob) throws Exception {
        return readImages(pathOrGlob, ImageOptions.defaults(), true);
    }

    public static DataFrame readImages(String pathOrGlob, ImageOptions opts) throws Exception {
        return readImages(pathOrGlob, opts, true);
    }

    public static DataFrame readImages(String pathOrGlob, ImageOptions opts, boolean withMeta) throws Exception {
        List<Path> files = expand(pathOrGlob, IMAGE_EXTS);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("image", Column.DType.IMAGE);
        if (withMeta) {
            df.addColumn("width", Column.DType.INT32);
            df.addColumn("height", Column.DType.INT32);
            df.addColumn("channels", Column.DType.INT32);
        }
        ImageOptions use = opts == null ? ImageOptions.defaults() : opts;
        for (Path p : files) {
            try {
                ImageData img = MediaBridge.loadImage(p.toString(), use);
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "image", img);
                if (withMeta) {
                    df.set(ri, "width", img.getWidth());
                    df.set(ri, "height", img.getHeight());
                    df.set(ri, "channels", img.getChannels());
                }
            } catch (Exception ignored) {}
        }
        return df;
    }

    public static DataFrame readAudio(String pathOrGlob) throws Exception {
        return readAudio(pathOrGlob, AudioOptions.defaults(), true);
    }

    public static DataFrame readAudio(String pathOrGlob, int sampleRate, boolean mono) throws Exception {
        return readAudio(pathOrGlob, AudioOptions.of(sampleRate, mono), true);
    }

    public static DataFrame readAudio(String pathOrGlob, AudioOptions opts, boolean withMeta) throws Exception {
        List<Path> files = expand(pathOrGlob, AUDIO_EXTS);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("audio", Column.DType.AUDIO);
        if (withMeta) {
            df.addColumn("sample_rate", Column.DType.INT32);
            df.addColumn("channels", Column.DType.INT32);
            df.addColumn("duration", Column.DType.FLOAT64);
            df.addColumn("num_samples", Column.DType.INT64);
        }
        AudioOptions use = opts == null ? AudioOptions.defaults() : opts;
        for (Path p : files) {
            try {
                AudioData aud = MediaBridge.loadAudio(p.toString(), use);
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "audio", aud);
                if (withMeta) {
                    df.set(ri, "sample_rate", aud.getSampleRate());
                    df.set(ri, "channels", aud.getChannels());
                    df.set(ri, "duration", aud.getDuration());
                    float[] s = aud.getSamples();
                    df.set(ri, "num_samples", s == null ? 0L : (long) s.length);
                }
            } catch (Exception ignored) {}
        }
        return df;
    }

    public static DataFrame readVideo(String pathOrGlob) throws Exception {
        return readVideo(pathOrGlob, VideoOptions.defaults(), true);
    }

    public static DataFrame readVideo(String pathOrGlob, VideoOptions opts) throws Exception {
        return readVideo(pathOrGlob, opts, true);
    }

    public static DataFrame readVideo(String pathOrGlob, VideoOptions opts, boolean withMeta) throws Exception {
        List<Path> files = expand(pathOrGlob, VIDEO_EXTS);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("video", Column.DType.VIDEO);
        if (withMeta) {
            df.addColumn("width", Column.DType.INT32);
            df.addColumn("height", Column.DType.INT32);
            df.addColumn("fps", Column.DType.FLOAT64);
            df.addColumn("duration", Column.DType.FLOAT64);
            df.addColumn("frame_count", Column.DType.INT32);
        }
        VideoOptions use = opts == null ? VideoOptions.defaults() : opts;
        for (Path p : files) {
            try {
                VideoData vid = MediaBridge.loadVideo(p.toString(), use);
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "video", vid);
                if (withMeta) {
                    df.set(ri, "width", vid.getWidth());
                    df.set(ri, "height", vid.getHeight());
                    df.set(ri, "fps", vid.getFps());
                    df.set(ri, "duration", vid.getDuration());
                    df.set(ri, "frame_count", vid.getFrameCount());
                }
            } catch (Exception e) {
                // path stub so the row is still present
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "video", new VideoData(p.toString()));
            }
        }
        return df;
    }

    // ── Folder datasets (torchvision / torchaudio / torchtext layouts) ────

    /**
     * {@code root/class_name/*.jpg} → columns path, image, label (int), class (string).
     */
    public static DataFrame readImageFolder(String root) throws Exception {
        return readImageFolder(root, ImageOptions.defaults());
    }

    public static DataFrame readImageFolder(String root, ImageOptions opts) throws Exception {
        Path rootPath = Path.of(root);
        List<String> classes = listClassDirs(rootPath);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("image", Column.DType.IMAGE);
        df.addColumn("label", Column.DType.INT32);
        df.addColumn("class", Column.DType.STRING);
        ImageOptions use = opts == null ? ImageOptions.defaults() : opts;
        for (int ci = 0; ci < classes.size(); ci++) {
            String cls = classes.get(ci);
            Path classDir = rootPath.resolve(cls);
            List<Path> files = listFiles(classDir, IMAGE_EXTS);
            for (Path p : files) {
                try {
                    ImageData img = MediaBridge.loadImage(p.toString(), use);
                    int ri = df.addEmptyRow();
                    df.set(ri, "path", p.toString());
                    df.set(ri, "image", img);
                    df.set(ri, "label", ci);
                    df.set(ri, "class", cls);
                } catch (Exception ignored) {}
            }
        }
        return df;
    }

    public static DataFrame readAudioFolder(String root) throws Exception {
        return readAudioFolder(root, AudioOptions.defaults());
    }

    public static DataFrame readAudioFolder(String root, AudioOptions opts) throws Exception {
        Path rootPath = Path.of(root);
        List<String> classes = listClassDirs(rootPath);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("audio", Column.DType.AUDIO);
        df.addColumn("label", Column.DType.INT32);
        df.addColumn("class", Column.DType.STRING);
        AudioOptions use = opts == null ? AudioOptions.defaults() : opts;
        for (int ci = 0; ci < classes.size(); ci++) {
            String cls = classes.get(ci);
            Path classDir = rootPath.resolve(cls);
            List<Path> files = listFiles(classDir, AUDIO_EXTS);
            for (Path p : files) {
                try {
                    AudioData aud = MediaBridge.loadAudio(p.toString(), use);
                    int ri = df.addEmptyRow();
                    df.set(ri, "path", p.toString());
                    df.set(ri, "audio", aud);
                    df.set(ri, "label", ci);
                    df.set(ri, "class", cls);
                } catch (Exception ignored) {}
            }
        }
        return df;
    }

    /**
     * {@code root/class_name/*.txt} → path, text, label, class.
     */
    public static DataFrame readTextFolder(String root) throws Exception {
        Path rootPath = Path.of(root);
        List<String> classes = listClassDirs(rootPath);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("text", Column.DType.STRING);
        df.addColumn("label", Column.DType.INT32);
        df.addColumn("class", Column.DType.STRING);
        for (int ci = 0; ci < classes.size(); ci++) {
            String cls = classes.get(ci);
            Path classDir = rootPath.resolve(cls);
            List<Path> files = listFiles(classDir, TEXT_EXTS);
            for (Path p : files) {
                try {
                    String text = Files.readString(p, StandardCharsets.UTF_8);
                    int ri = df.addEmptyRow();
                    df.set(ri, "path", p.toString());
                    df.set(ri, "text", text);
                    df.set(ri, "label", ci);
                    df.set(ri, "class", cls);
                } catch (Exception ignored) {}
            }
        }
        return df;
    }

    /**
     * Mixed multimodal table from a directory that may contain images, audio, video and text.
     * Columns: path, modality, image, audio, video, text (only one media cell filled per row).
     */
    public static DataFrame readMultimodalDir(String root) throws Exception {
        Path rootPath = Path.of(root);
        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("modality", Column.DType.STRING);
        df.addColumn("image", Column.DType.IMAGE);
        df.addColumn("audio", Column.DType.AUDIO);
        df.addColumn("video", Column.DType.VIDEO);
        df.addColumn("text", Column.DType.STRING);

        if (!Files.isDirectory(rootPath)) {
            // single file
            addMultimodalFile(df, rootPath);
            return df;
        }
        try (var stream = Files.walk(rootPath)) {
            List<Path> files = stream.filter(Files::isRegularFile).sorted().collect(Collectors.toList());
            for (Path p : files) addMultimodalFile(df, p);
        }
        return df;
    }

    private static void addMultimodalFile(DataFrame df, Path p) {
        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
        try {
            if (endsWithAny(name, IMAGE_EXTS)) {
                ImageData img = MediaBridge.loadImage(p.toString());
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "modality", "image");
                df.set(ri, "image", img);
            } else if (endsWithAny(name, AUDIO_EXTS)) {
                AudioData aud = MediaBridge.loadAudio(p.toString());
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "modality", "audio");
                df.set(ri, "audio", aud);
            } else if (endsWithAny(name, VIDEO_EXTS)) {
                VideoData vid = MediaBridge.loadVideo(p.toString(),
                        VideoOptions.defaults().withMaxFrames(32).withTargetFps(2.0));
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "modality", "video");
                df.set(ri, "video", vid);
            } else if (endsWithAny(name, TEXT_EXTS)) {
                String text = Files.readString(p, StandardCharsets.UTF_8);
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "modality", "text");
                df.set(ri, "text", text);
            }
        } catch (Exception ignored) {}
    }

    // ── In-memory builders / interop entry points ─────────────────────────

    /** Build image DataFrame from already-decoded {@link ImageData} cells. */
    public static DataFrame fromImages(String imageCol, List<ImageData> images) {
        return DataFrame.fromImages(imageCol, images);
    }

    /** Build from OpenCV-decoded tensors (CHW [0,255] or [0,1]). */
    public static DataFrame fromOpenCVTensors(String imageCol, List<Tensor> tensors) {
        DataFrame df = DataFrame.create();
        String col = imageCol == null ? "image" : imageCol;
        df.addColumn(col, Column.DType.IMAGE);
        if (tensors != null) {
            for (Tensor t : tensors) {
                if (t == null) continue;
                int ri = df.addEmptyRow();
                df.set(ri, col, MediaBridge.tensorToImage(t));
            }
        }
        return df;
    }

    /** Build from FFmpeg video frame tensors. */
    public static DataFrame fromFFmpegFrames(String imageCol, List<Tensor> frames, double fps) {
        DataFrame df = DataFrame.create();
        String col = imageCol == null ? "frame" : imageCol;
        df.addColumn(col, Column.DType.IMAGE);
        df.addColumn("frame_idx", Column.DType.INT32);
        df.addColumn("time_sec", Column.DType.FLOAT64);
        if (frames != null) {
            double useFps = fps > 0 ? fps : 30.0;
            for (int i = 0; i < frames.size(); i++) {
                Tensor t = frames.get(i);
                if (t == null) continue;
                int ri = df.addEmptyRow();
                df.set(ri, col, MediaBridge.tensorToImage(t));
                df.set(ri, "frame_idx", i);
                df.set(ri, "time_sec", i / useFps);
            }
        }
        return df;
    }

    /** Build audio DataFrame from waveform tensors [C,T]. */
    public static DataFrame fromAudioTensors(String audioCol, List<Tensor> waves, int sampleRate) {
        DataFrame df = DataFrame.create();
        String col = audioCol == null ? "audio" : audioCol;
        df.addColumn(col, Column.DType.AUDIO);
        df.addColumn("sample_rate", Column.DType.INT32);
        if (waves != null) {
            int sr = sampleRate > 0 ? sampleRate : 16000;
            for (Tensor w : waves) {
                if (w == null) continue;
                int ri = df.addEmptyRow();
                df.set(ri, col, MediaBridge.tensorToAudio(w, sr));
                df.set(ri, "sample_rate", sr);
            }
        }
        return df;
    }

    /** Build text classification table. */
    public static DataFrame fromText(List<String> texts, List<String> labels) {
        DataFrame df = DataFrame.create();
        df.addColumn("text", Column.DType.STRING);
        df.addColumn("label", Column.DType.STRING);
        int n = texts == null ? 0 : texts.size();
        for (int i = 0; i < n; i++) {
            int ri = df.addEmptyRow();
            df.set(ri, "text", texts.get(i));
            if (labels != null && i < labels.size()) df.set(ri, "label", labels.get(i));
        }
        return df;
    }

    /**
     * Convert a torchvision {@code ImageFolder}-compatible object (or any object
     * exposing {@code samples()}/{@code targets()}/{@code classes()}) into a DataFrame.
     */
    public static DataFrame fromVisionDataset(Object dataset) throws Exception {
        Objects.requireNonNull(dataset, "dataset");
        @SuppressWarnings("unchecked")
        List<Path> samples = (List<Path>) dataset.getClass().getMethod("samples").invoke(dataset);
        @SuppressWarnings("unchecked")
        List<Integer> targets = (List<Integer>) dataset.getClass().getMethod("targets").invoke(dataset);
        @SuppressWarnings("unchecked")
        List<String> classes = (List<String>) dataset.getClass().getMethod("classes").invoke(dataset);

        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("image", Column.DType.IMAGE);
        df.addColumn("label", Column.DType.INT32);
        df.addColumn("class", Column.DType.STRING);
        if (samples == null) return df;
        for (int i = 0; i < samples.size(); i++) {
            Path p = samples.get(i);
            try {
                ImageData img = MediaBridge.loadImage(p.toString());
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "image", img);
                int label = targets != null && i < targets.size() ? targets.get(i) : -1;
                df.set(ri, "label", label);
                String cls = (classes != null && label >= 0 && label < classes.size())
                        ? classes.get(label) : null;
                df.set(ri, "class", cls);
            } catch (Exception ignored) {}
        }
        return df;
    }

    public static DataFrame fromAudioDataset(Object dataset) throws Exception {
        Objects.requireNonNull(dataset, "dataset");
        @SuppressWarnings("unchecked")
        List<Path> samples = (List<Path>) dataset.getClass().getMethod("samples").invoke(dataset);
        @SuppressWarnings("unchecked")
        List<Integer> targets = (List<Integer>) dataset.getClass().getMethod("targets").invoke(dataset);
        @SuppressWarnings("unchecked")
        List<String> classes = (List<String>) dataset.getClass().getMethod("classes").invoke(dataset);
        int sr = 16000;
        boolean mono = true;
        try { sr = ((Number) dataset.getClass().getMethod("sampleRate").invoke(dataset)).intValue(); } catch (Exception ignored) {}
        try { mono = (Boolean) dataset.getClass().getMethod("mono").invoke(dataset); } catch (Exception ignored) {}

        DataFrame df = DataFrame.create();
        df.addColumn("path", Column.DType.STRING);
        df.addColumn("audio", Column.DType.AUDIO);
        df.addColumn("label", Column.DType.INT32);
        df.addColumn("class", Column.DType.STRING);
        if (samples == null) return df;
        AudioOptions opts = AudioOptions.of(sr, mono);
        for (int i = 0; i < samples.size(); i++) {
            Path p = samples.get(i);
            try {
                AudioData aud = MediaBridge.loadAudio(p.toString(), opts);
                int ri = df.addEmptyRow();
                df.set(ri, "path", p.toString());
                df.set(ri, "audio", aud);
                int label = targets != null && i < targets.size() ? targets.get(i) : -1;
                df.set(ri, "label", label);
                String cls = (classes != null && label >= 0 && label < classes.size())
                        ? classes.get(label) : null;
                df.set(ri, "class", cls);
            } catch (Exception ignored) {}
        }
        return df;
    }

    public static DataFrame fromTextDataset(Object dataset) throws Exception {
        Objects.requireNonNull(dataset, "dataset");
        @SuppressWarnings("unchecked")
        List<?> samples = (List<?>) dataset.getClass().getMethod("samples").invoke(dataset);
        DataFrame df = DataFrame.create();
        df.addColumn("text", Column.DType.STRING);
        df.addColumn("label", Column.DType.INT32);
        df.addColumn("class", Column.DType.STRING);
        if (samples == null) return df;
        for (Object s : samples) {
            try {
                String text = (String) s.getClass().getField("text").get(s);
                int label = ((Number) s.getClass().getField("label").get(s)).intValue();
                String labelName = (String) s.getClass().getField("labelName").get(s);
                int ri = df.addEmptyRow();
                df.set(ri, "text", text);
                df.set(ri, "label", label);
                df.set(ri, "class", labelName);
            } catch (Exception e) {
                try {
                    String text = (String) s.getClass().getMethod("text").invoke(s);
                    int label = ((Number) s.getClass().getMethod("label").invoke(s)).intValue();
                    int ri = df.addEmptyRow();
                    df.set(ri, "text", text);
                    df.set(ri, "label", label);
                } catch (Exception ignored) {}
            }
        }
        return df;
    }

    // ── Frame explosion / embedding columns ───────────────────────────────

    /**
     * Explode a video column into one row per extracted frame.
     * Result columns: original non-video cols + {@code frame}, {@code frame_idx}, {@code time_sec}.
     */
    public static DataFrame extractVideoFrames(DataFrame src, String videoCol, double fps) throws Exception {
        Objects.requireNonNull(src, "src");
        String vcol = videoCol == null ? "video" : videoCol;
        if (!src.hasColumn(vcol)) {
            throw new IllegalArgumentException("missing video column: " + vcol);
        }
        List<String> keep = new ArrayList<>();
        for (String c : src.getColumnNames()) {
            if (!c.equals(vcol)) keep.add(c);
        }
        DataFrame out = DataFrame.create();
        for (String c : keep) {
            out.addColumn(c, src.getColumn(c).dtype());
        }
        out.addColumn("frame", Column.DType.IMAGE);
        out.addColumn("frame_idx", Column.DType.INT32);
        out.addColumn("time_sec", Column.DType.FLOAT64);
        out.addColumn("src_path", Column.DType.STRING);

        for (int r = 0; r < src.rowCount(); r++) {
            Object cell = src.get(r, vcol);
            VideoData vid = cell instanceof VideoData vd ? vd : null;
            if (vid == null) continue;
            // ensure frames present
            if ((vid.getFrames() == null || vid.getFrames().isEmpty()) && vid.getPath() != null) {
                try {
                    vid = MediaBridge.loadVideo(vid.getPath(),
                            VideoOptions.defaults().withTargetFps(fps > 0 ? fps : 1.0));
                } catch (Exception ignored) {}
            }
            List<ImageData> frames = MediaBridge.extractFrames(vid, fps > 0 ? fps : vid.getFps());
            double useFps = fps > 0 ? fps : (vid.getFps() > 0 ? vid.getFps() : 1.0);
            for (int fi = 0; fi < frames.size(); fi++) {
                int ri = out.addEmptyRow();
                for (String c : keep) {
                    out.set(ri, c, src.get(r, c));
                }
                out.set(ri, "frame", frames.get(fi));
                out.set(ri, "frame_idx", fi);
                out.set(ri, "time_sec", fi / useFps);
                out.set(ri, "src_path", vid.getPath());
            }
        }
        return out;
    }

    /**
     * Ensure a column exists and is padded to {@code df.rowCount()} nulls so
     * subsequent {@link DataFrame#set(int, String, Object)} calls are in-bounds.
     * {@link DataFrame#addColumn(String, Column.DType)} does not auto-pad.
     */
    public static void ensureColumn(DataFrame df, String name, Column.DType dtype) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(name, "name");
        if (!df.hasColumn(name)) {
            df.addColumn(name, dtype);
        }
        Column col = df.getColumn(name);
        int need = df.rowCount();
        while (col.size() < need) col.add(null);
    }

    /** Add embedding column from image column. */
    public static DataFrame embedImages(DataFrame df, String imageCol, String outCol, int dim) throws Exception {
        Objects.requireNonNull(df, "df");
        String ic = imageCol == null ? "image" : imageCol;
        String oc = outCol == null ? "embedding" : outCol;
        DataFrame out = df.copy();
        ensureColumn(out, oc, Column.DType.EMBEDDING);
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, ic);
            if (cell instanceof ImageData id) {
                out.set(r, oc, MediaBridge.embedImage(id, dim));
            }
        }
        return out;
    }

    public static DataFrame embedAudio(DataFrame df, String audioCol, String outCol, int dim) throws Exception {
        Objects.requireNonNull(df, "df");
        String ac = audioCol == null ? "audio" : audioCol;
        String oc = outCol == null ? "embedding" : outCol;
        DataFrame out = df.copy();
        ensureColumn(out, oc, Column.DType.EMBEDDING);
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, ac);
            if (cell instanceof AudioData ad) {
                out.set(r, oc, MediaBridge.embedAudio(ad, dim));
            }
        }
        return out;
    }

    public static DataFrame embedVideo(DataFrame df, String videoCol, String outCol, int dim) throws Exception {
        Objects.requireNonNull(df, "df");
        String vc = videoCol == null ? "video" : videoCol;
        String oc = outCol == null ? "embedding" : outCol;
        DataFrame out = df.copy();
        ensureColumn(out, oc, Column.DType.EMBEDDING);
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, vc);
            if (cell instanceof VideoData vd) {
                out.set(r, oc, MediaBridge.embedVideo(vd, dim));
            }
        }
        return out;
    }


    /** Embed image column with a named neural / registry model (e.g. {@code "resnet18"}). */
    public static DataFrame embedImagesModel(DataFrame df, String imageCol, String outCol, String modelId)
            throws Exception {
        Objects.requireNonNull(df, "df");
        String ic = imageCol == null ? "image" : imageCol;
        String oc = outCol == null ? "embedding" : outCol;
        DataFrame out = df.copy();
        ensureColumn(out, oc, Column.DType.EMBEDDING);
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, ic);
            if (cell instanceof ImageData id) {
                out.set(r, oc, MediaBridge.embedImageModel(id, modelId));
            }
        }
        return out;
    }

    /** Embed audio column with a named neural / registry model (e.g. {@code "m5"}). */
    public static DataFrame embedAudioModel(DataFrame df, String audioCol, String outCol, String modelId)
            throws Exception {
        Objects.requireNonNull(df, "df");
        String ac = audioCol == null ? "audio" : audioCol;
        String oc = outCol == null ? "embedding" : outCol;
        DataFrame out = df.copy();
        ensureColumn(out, oc, Column.DType.EMBEDDING);
        for (int r = 0; r < out.rowCount(); r++) {
            Object cell = out.get(r, ac);
            if (cell instanceof AudioData ad) {
                out.set(r, oc, MediaBridge.embedAudioModel(ad, modelId));
            }
        }
        return out;
    }

    /**
     * Convert image column to a stacked NCHW tensor (all rows, resized to first non-null).
     */
    public static Tensor imagesToBatchTensor(DataFrame df, String imageCol) {
        Objects.requireNonNull(df, "df");
        String ic = imageCol == null ? "image" : imageCol;
        List<ImageData> imgs = new ArrayList<>();
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = df.get(r, ic);
            if (cell instanceof ImageData id && id.getImage() != null) imgs.add(id);
        }
        return MediaBridge.stackImages(imgs);
    }

    /** Convert audio column row-wise to list of [C,T] tensors. */
    public static List<Tensor> audioToTensors(DataFrame df, String audioCol) {
        Objects.requireNonNull(df, "df");
        String ac = audioCol == null ? "audio" : audioCol;
        List<Tensor> out = new ArrayList<>();
        for (int r = 0; r < df.rowCount(); r++) {
            Object cell = df.get(r, ac);
            if (cell instanceof AudioData ad && ad.getSamples() != null) {
                out.add(MediaBridge.audioToTensor(ad));
            } else {
                out.add(null);
            }
        }
        return out;
    }

    // ── path expansion (shared with DataFrame) ────────────────────────────

    public static List<Path> expand(String pathOrGlob, String... extensions) throws Exception {
        List<Path> out = new ArrayList<>();
        if (pathOrGlob == null || pathOrGlob.isBlank()) return out;
        Set<String> exts = new HashSet<>();
        for (String e : extensions) exts.add(e.toLowerCase(Locale.ROOT));

        String[] parts = pathOrGlob.split(",");
        for (String part : parts) {
            String p = part.trim();
            if (p.isEmpty()) continue;
            Path path = Path.of(p);
            if (Files.isRegularFile(path)) {
                out.add(path);
            } else if (Files.isDirectory(path)) {
                try (var stream = Files.list(path)) {
                    stream.filter(Files::isRegularFile)
                            .filter(f -> {
                                String name = f.getFileName().toString().toLowerCase(Locale.ROOT);
                                for (String e : exts) if (name.endsWith(e)) return true;
                                return false;
                            })
                            .sorted()
                            .forEach(out::add);
                }
            } else {
                Path parent = path.getParent() != null ? path.getParent() : Path.of(".");
                String pattern = path.getFileName() != null ? path.getFileName().toString() : "*";
                if (Files.isDirectory(parent)) {
                    try (DirectoryStream<Path> stream = Files.newDirectoryStream(parent, pattern)) {
                        for (Path f : stream) {
                            if (Files.isRegularFile(f)) out.add(f);
                        }
                    } catch (Exception ignored) {}
                }
            }
        }
        return out;
    }

    private static List<String> listClassDirs(Path root) throws Exception {
        List<String> classes = new ArrayList<>();
        if (!Files.isDirectory(root)) return classes;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(root)) {
            for (Path p : ds) {
                if (Files.isDirectory(p) && !p.getFileName().toString().startsWith(".")) {
                    classes.add(p.getFileName().toString());
                }
            }
        }
        Collections.sort(classes);
        return classes;
    }

    private static List<Path> listFiles(Path dir, String... exts) throws Exception {
        List<Path> files = new ArrayList<>();
        if (!Files.isDirectory(dir)) return files;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(dir)) {
            for (Path p : ds) {
                if (Files.isRegularFile(p)
                        && endsWithAny(p.getFileName().toString().toLowerCase(Locale.ROOT), exts)) {
                    files.add(p);
                }
            }
        }
        Collections.sort(files);
        return files;
    }

    private static boolean endsWithAny(String name, String... exts) {
        for (String e : exts) {
            if (name.endsWith(e.toLowerCase(Locale.ROOT))) return true;
        }
        return false;
    }

    /** Capability summary for diagnostics / benchmarks. */

    public static Map<String, Object> capabilities() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("opencv", MediaBridge.isOpenCvAvailable());
        m.put("ffmpeg", MediaBridge.isFFmpegAvailable());
        m.put("image_exts", List.of(IMAGE_EXTS));
        m.put("audio_exts", List.of(AUDIO_EXTS));
        m.put("video_exts", List.of(VIDEO_EXTS));
        return m;
    }
}
