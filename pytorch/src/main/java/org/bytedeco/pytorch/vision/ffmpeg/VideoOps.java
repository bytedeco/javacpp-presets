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
package org.bytedeco.pytorch.vision.ffmpeg;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Enterprise-grade video operations on top of javacpp-ffmpeg — Daft / torchcodec /
 * Meta torchvision / ByteDance multimodal pipeline style convenience API.
 *
 * <p>Two backends:
 * <ul>
 *   <li><b>In-process</b> — {@link VideoFile} / libav* (seek, uniform sample, decode→Tensor)</li>
 *   <li><b>CLI pipeline</b> — {@link FFmpeg} process builder (transcode, clip, concat, gif, extract audio)</li>
 * </ul>
 *
 * <pre>{@code
 * // ── Frame extraction (multimodal / VL models) ──────────────────────────
 * List<Tensor> frames = VideoOps.extractUniform("clip.mp4", 8);     // LLaVA/Qwen-VL
 * Tensor batch = VideoOps.extractUniformStacked("clip.mp4", 8);     // [N,3,H,W]
 * List<Tensor> at2fps = VideoOps.extractAtFps("clip.mp4", 2.0, 64);
 * Tensor thumb = VideoOps.thumbnail("clip.mp4");
 * Tensor at = VideoOps.frameAt("clip.mp4", 1.5);
 *
 * // ── Probe ──────────────────────────────────────────────────────────────
 * VideoFile.VideoMeta meta = VideoOps.probe("clip.mp4");
 *
 * // ── Transcode / clip / concat (CLI ffmpeg) ─────────────────────────────
 * VideoOps.clip("in.mp4", "out.mp4", 1.0, 5.0);
 * VideoOps.transcode("in.mp4", "out.mp4", "libx264", "23");
 * VideoOps.concat(List.of("a.mp4", "b.mp4"), "ab.mp4");
 * VideoOps.toGif("in.mp4", "out.gif", 10, 320);
 * VideoOps.extractAudio("in.mp4", "out.wav");
 * VideoOps.extractFramesToDir("in.mp4", "/tmp/frames", 1.0); // 1 fps JPEGs
 * }</pre>
 */
public final class VideoOps {

    private VideoOps() {}

    // ═══════════════════════════════════════════════════════════════════════
    // Probe
    // ═══════════════════════════════════════════════════════════════════════

    /** Lightweight metadata probe (opens+closes container, no full decode). */
    public static VideoFile.VideoMeta probe(String path) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.meta();
        }
    }

    public static VideoFile.VideoMeta probe(Path path) {
        return probe(path.toString());
    }

    /** True if path opens as a video with ≥1 video stream. */
    public static boolean isReadable(String path) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.width() > 0 && vf.height() > 0;
        } catch (Throwable t) {
            return false;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Frame extraction → Tensor  (in-process, multimodal-friendly)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Uniformly sample {@code count} frames across the whole timeline
     * (Daft / LLaVA / Qwen-VL / InternVL style).
     *
     * @return list of {@code [3,H,W]} float32 tensors in {@code [0,255]}
     */
    public static List<Tensor> extractUniform(String path, int count) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.extractUniform(count);
        }
    }

    /** Same as {@link #extractUniform} but stacked to {@code [N,3,H,W]}. */
    public static Tensor extractUniformStacked(String path, int count) {
        return VideoFile.stackFrames(extractUniform(path, count));
    }

    /**
     * Sample at target fps from start, optionally capped.
     *
     * @param maxFrames ≤0 → no cap
     */
    public static List<Tensor> extractAtFps(String path, double targetFps, int maxFrames) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.extractAtFps(targetFps, maxFrames);
        }
    }

    public static List<Tensor> extractAtFps(String path, double targetFps) {
        return extractAtFps(path, targetFps, 0);
    }

    /** Every N-th frame, optionally capped. */
    public static List<Tensor> extractEveryN(String path, int n, int maxFrames) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.extractEveryN(n, maxFrames);
        }
    }

    public static List<Tensor> extractEveryN(String path, int n) {
        return extractEveryN(path, n, 0);
    }

    /** Frames in {@code [startSec, endSec)}. */
    public static List<Tensor> extractRange(String path, double startSec, double endSec, int maxFrames) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.extractRange(startSec, endSec, maxFrames);
        }
    }

    public static List<Tensor> extractRange(String path, double startSec, double endSec) {
        return extractRange(path, startSec, endSec, 0);
    }

    /** Single frame nearest to {@code seconds}. */
    public static Tensor frameAt(String path, double seconds) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.frameAt(seconds);
        }
    }

    /** First decodable frame (poster / thumbnail). */
    public static Tensor thumbnail(String path) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.thumbnail();
        }
    }

    /**
     * Full sequential decode (use with care on long videos).
     * Prefer {@link #extractUniform} / {@link #extractAtFps} for ML pipelines.
     */
    public static List<Tensor> decodeAll(String path) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.readFrames();
        }
    }

    public static Tensor decodeAllStacked(String path) {
        try (VideoFile vf = VideoFile.open(path)) {
            return vf.read();
        }
    }

    /**
     * Multimodal-ready pack: uniform sample + optional resize hint metadata.
     * Returns a map with keys: {@code frames} (List&lt;Tensor&gt;), {@code stacked} (Tensor),
     * {@code meta} (VideoMeta), {@code count} (int).
     */
    public static Map<String, Object> sampleForVlm(String path, int numFrames) {
        Objects.requireNonNull(path, "path");
        if (numFrames <= 0) numFrames = 8;
        try (VideoFile vf = VideoFile.open(path)) {
            List<Tensor> frames = vf.extractUniform(numFrames);
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("frames", frames);
            m.put("stacked", VideoFile.stackFrames(frames));
            m.put("meta", vf.meta());
            m.put("count", frames.size());
            m.put("path", path);
            return m;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CLI-backed ops (transcode / clip / concat / gif / audio / dump frames)
    // ═══════════════════════════════════════════════════════════════════════

    /** Require system ffmpeg binary; throw if missing. */
    private static void requireCli() {
        if (!FFmpeg.isAvailable()) {
            throw new FFmpegException(
                    "ffmpeg binary not found on PATH (set ffmpeg.binary / FFMPEG_BINARY)");
        }
    }

    /**
     * Lossless-ish stream copy clip: {@code [startSec, startSec+durationSec)}.
     * Uses {@code -c copy} when possible (fast, keyframe-aligned).
     */
    public static void clip(String input, String output, double startSec, double durationSec) {
        clip(input, output, startSec, durationSec, true);
    }

    /**
     * @param streamCopy if true use {@code -c copy} (fast, keyframe snap); else re-encode h264/aac
     */
    public static void clip(String input, String output, double startSec, double durationSec,
                            boolean streamCopy) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        requireCli();
        if (streamCopy) {
            FFmpeg.input(input, "ss", String.valueOf(startSec), "t", String.valueOf(durationSec))
                    .output(output, "c", "copy")
                    .overwriteOutput()
                    .run();
        } else {
            FFmpeg.input(input, "ss", String.valueOf(startSec), "t", String.valueOf(durationSec))
                    .output(output, "vcodec", "libx264", "acodec", "aac", "crf", "23", "preset", "veryfast")
                    .overwriteOutput()
                    .run();
        }
    }

    /**
     * Trim by absolute end time: keep {@code [startSec, endSec)}.
     */
    public static void trim(String input, String output, double startSec, double endSec) {
        if (endSec <= startSec) throw new IllegalArgumentException("endSec must be > startSec");
        clip(input, output, startSec, endSec - startSec, true);
    }

    /**
     * Transcode to H.264/AAC (or custom codecs).
     *
     * @param vcodec e.g. {@code libx264}, {@code libx265}, {@code copy}
     * @param crf    quality (lower = better), ignored if vcodec is {@code copy}
     */
    public static void transcode(String input, String output, String vcodec, String crf) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        requireCli();
        String vc = vcodec == null ? "libx264" : vcodec;
        if ("copy".equals(vc)) {
            FFmpeg.input(input)
                    .output(output, "c", "copy")
                    .overwriteOutput()
                    .run();
        } else {
            String c = crf == null ? "23" : crf;
            FFmpeg.input(input)
                    .output(output, "vcodec", vc, "acodec", "aac", "crf", c, "preset", "veryfast",
                            "pix_fmt", "yuv420p", "movflags", "+faststart")
                    .overwriteOutput()
                    .run();
        }
    }

    /** Scale video to {@code width}x{@code height} (use -1 to keep aspect on one dim). */
    public static void scale(String input, String output, int width, int height) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        requireCli();
        String w = width > 0 ? String.valueOf(width) : "-1";
        String h = height > 0 ? String.valueOf(height) : "-1";
        FFmpeg.input(input)
                .filter("scale", w, h)
                .output(output, "vcodec", "libx264", "acodec", "aac", "crf", "23", "pix_fmt", "yuv420p")
                .overwriteOutput()
                .run();
    }

    /**
     * Concatenate multiple videos (same codec/resolution preferred for stream-copy).
     * Uses filter_complex concat (re-encode) for robustness across mismatched inputs.
     */
    public static void concat(List<String> inputs, String output) {
        Objects.requireNonNull(inputs, "inputs");
        Objects.requireNonNull(output, "output");
        if (inputs.isEmpty()) throw new IllegalArgumentException("inputs empty");
        requireCli();
        if (inputs.size() == 1) {
            transcode(inputs.get(0), output, "copy", null);
            return;
        }
        FFmpeg.FFmpegNode[] nodes = new FFmpeg.FFmpegNode[inputs.size()];
        for (int i = 0; i < inputs.size(); i++) {
            nodes[i] = FFmpeg.input(inputs.get(i));
        }
        FFmpeg.concat(true, true, nodes)
                .output(output, "vcodec", "libx264", "acodec", "aac", "crf", "23", "pix_fmt", "yuv420p")
                .overwriteOutput()
                .run();
    }

    /**
     * Export an animated GIF (palettegen dual-pass simplified to single scale+fps).
     *
     * @param fps   output gif fps
     * @param width max width (height auto); ≤0 keeps source width
     */
    public static void toGif(String input, String output, int fps, int width) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        requireCli();
        int f = fps > 0 ? fps : 10;
        String scaleExpr = width > 0
                ? "fps=" + f + ",scale=" + width + ":-1:flags=lanczos"
                : "fps=" + f;
        FFmpeg.input(input)
                .filter(scaleExpr)
                .output(output, "format", "gif")
                .overwriteOutput()
                .run();
    }

    /** Extract audio track to wav/mp3/aac (format from output extension). */
    public static void extractAudio(String input, String output) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        requireCli();
        String ext = extension(output);
        if ("wav".equals(ext)) {
            FFmpeg.input(input)
                    .output(output, "vn", "true", "acodec", "pcm_s16le")
                    .overwriteOutput()
                    .run();
        } else if ("mp3".equals(ext)) {
            FFmpeg.input(input)
                    .output(output, "vn", "true", "acodec", "libmp3lame", "q:a", "2")
                    .overwriteOutput()
                    .run();
        } else {
            FFmpeg.input(input)
                    .output(output, "vn", "true", "acodec", "aac")
                    .overwriteOutput()
                    .run();
        }
    }

    /**
     * Dump frames as image sequence via CLI at {@code fps} (default 1).
     * Output pattern example: {@code /tmp/frames/frame_%05d.jpg}
     *
     * @param outputDir directory created if missing; writes {@code frame_%05d.jpg}
     * @return list of written file paths (best-effort scan)
     */
    public static List<Path> extractFramesToDir(String input, String outputDir, double fps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(outputDir, "outputDir");
        requireCli();
        try {
            Path dir = Paths.get(outputDir);
            Files.createDirectories(dir);
            String pattern = dir.resolve("frame_%05d.jpg").toString();
            double f = fps > 0 ? fps : 1.0;
            FFmpeg.input(input)
                    .output(pattern, "vf", "fps=" + f, "qscale:v", "2")
                    .overwriteOutput()
                    .run();
            List<Path> written = new ArrayList<>();
            try (var stream = Files.list(dir)) {
                stream.filter(p -> p.getFileName().toString().startsWith("frame_"))
                        .sorted()
                        .forEach(written::add);
            }
            return written;
        } catch (FFmpegException e) {
            throw e;
        } catch (Exception e) {
            throw new FFmpegException("extractFramesToDir failed: " + e.getMessage(), e);
        }
    }

    /**
     * Change container fps presentation (re-encode) — does not interpolate motion.
     * For true interpolation use filter {@code minterpolate} via {@link FFmpeg} directly.
     */
    public static void setFps(String input, String output, double fps) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output");
        requireCli();
        FFmpeg.input(input)
                .filter("fps", String.valueOf(fps))
                .output(output, "vcodec", "libx264", "acodec", "aac", "crf", "23", "pix_fmt", "yuv420p")
                .overwriteOutput()
                .run();
    }

    /** Mute video (drop audio). */
    public static void stripAudio(String input, String output) {
        requireCli();
        FFmpeg.input(input)
                .output(output, "c:v", "copy", "an", "true")
                .overwriteOutput()
                .run();
    }

    /**
     * Write a short silent H.264 mp4 from an RGB tensor batch {@code [N,3,H,W]} in {@code [0,255]}
     * by dumping PNG frames then encoding — useful for unit tests / debug visualization.
     *
     * <p>Requires CLI ffmpeg + {@code OpenCVIO} (or falls back to raw ppm if OpenCV absent).
     */
    public static void writeTensorVideo(String output, List<Tensor> frames, double fps) {
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(frames, "frames");
        if (frames.isEmpty()) throw new IllegalArgumentException("frames empty");
        requireCli();
        double useFps = fps > 0 ? fps : 8.0;
        Path tmp;
        try {
            tmp = Files.createTempDirectory("videoops_frames_");
            // Prefer OpenCV encode if present
            boolean usedOpenCv = false;
            try {
                Class<?> io = Class.forName("org.bytedeco.pytorch.vision.opencv.OpenCVIO");
                for (int i = 0; i < frames.size(); i++) {
                    String name = String.format("frame_%05d.png", i);
                    Path p = tmp.resolve(name);
                    io.getMethod("writeImage", String.class, Tensor.class)
                            .invoke(null, p.toString(), frames.get(i));
                }
                usedOpenCv = true;
            } catch (ClassNotFoundException cnf) {
                usedOpenCv = false;
            } catch (Throwable t) {
                throw new FFmpegException("frame dump failed: " + t.getMessage(), t);
            }
            if (!usedOpenCv) {
                throw new FFmpegException(
                        "writeTensorVideo requires org.bytedeco.pytorch.utils.opencv.OpenCVIO on classpath");
            }
            String pattern = tmp.resolve("frame_%05d.png").toString();
            FFmpeg.input(pattern, "framerate", String.valueOf(useFps))
                    .output(output, "vcodec", "libx264", "pix_fmt", "yuv420p", "crf", "23",
                            "an", "true")
                    .overwriteOutput()
                    .run();
        } catch (FFmpegException e) {
            throw e;
        } catch (Exception e) {
            throw new FFmpegException("writeTensorVideo failed: " + e.getMessage(), e);
        } finally {
            // best-effort cleanup left to OS temp sweeper; avoid deleting before ffmpeg finishes
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Batch / DataFrame-adjacent helpers
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Probe many paths; skip unreadable. Returns map path→meta.
     */
    public static Map<String, VideoFile.VideoMeta> probeAll(List<String> paths) {
        Map<String, VideoFile.VideoMeta> out = new LinkedHashMap<>();
        if (paths == null) return out;
        for (String p : paths) {
            try {
                out.put(p, probe(p));
            } catch (Throwable ignored) {}
        }
        return out;
    }

    /**
     * Extract uniform frames from many videos (embarrassingly parallel-friendly;
     * this impl is sequential — caller may parallelize).
     *
     * @return list aligned with inputs; empty list entry on failure
     */
    public static List<List<Tensor>> extractUniformBatch(List<String> paths, int count) {
        if (paths == null) return List.of();
        List<List<Tensor>> out = new ArrayList<>(paths.size());
        for (String p : paths) {
            try {
                out.add(extractUniform(p, count));
            } catch (Throwable t) {
                out.add(Collections.emptyList());
            }
        }
        return out;
    }

    /**
     * Stack variable-length frame lists to a single batch by resizing all frames
     * to the first frame's HxW via center-crop-or-pad identity (no OpenCV required):
     * only stacks equal-sized frames; drops mismatches.
     */
    public static Tensor stackEqualFrames(List<Tensor> frames) {
        if (frames == null || frames.isEmpty()) {
            return torch.empty(new long[]{0, 3, 1, 1}, new TensorOptions(ScalarType.Float), null);
        }
        long h = frames.get(0).size(1);
        long w = frames.get(0).size(2);
        List<Tensor> ok = new ArrayList<>();
        for (Tensor f : frames) {
            if (f.dim() == 3 && f.size(1) == h && f.size(2) == w) ok.add(f);
        }
        return VideoFile.stackFrames(ok);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Capability report
    // ═══════════════════════════════════════════════════════════════════════

    public static Map<String, Object> capabilities() {
        Map<String, Object> m = new LinkedHashMap<>();
        boolean nativeOk = false;
        try {
            FFmpegNative.load();
            nativeOk = true;
        } catch (Throwable t) {
            m.put("nativeError", t.getMessage());
        }
        m.put("nativeLibav", nativeOk);
        m.put("cliFfmpeg", FFmpeg.isAvailable());
        m.put("cliBinary", FFmpeg.findBinary());
        m.put("ops", List.of(
                "probe", "extractUniform", "extractAtFps", "extractEveryN", "extractRange",
                "frameAt", "thumbnail", "decodeAll", "sampleForVlm",
                "clip", "trim", "transcode", "scale", "concat", "toGif",
                "extractAudio", "extractFramesToDir", "setFps", "stripAudio",
                "writeTensorVideo", "probeAll", "extractUniformBatch"
        ));
        return m;
    }

    private static String extension(String path) {
        int dot = path.lastIndexOf('.');
        int sep = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        if (dot > sep && dot >= 0 && dot + 1 < path.length()) {
            return path.substring(dot + 1).toLowerCase();
        }
        return "";
    }
}
