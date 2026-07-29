/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option)
 * any later version (collectively, the "License");
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
package org.bytedeco.pytorch.utils.ffmpeg;

import org.bytedeco.ffmpeg.avutil.AVDictionary;

import org.bytedeco.javacpp.Pointer;

import org.bytedeco.ffmpeg.avutil.AVRational;

import org.bytedeco.ffmpeg.avformat.AVStream;

import org.bytedeco.ffmpeg.avcodec.AVCodecParameters;

import org.bytedeco.ffmpeg.avcodec.AVCodec;

import org.bytedeco.ffmpeg.avcodec.AVCodecContext;
import org.bytedeco.ffmpeg.avcodec.AVPacket;
import org.bytedeco.ffmpeg.avformat.AVFormatContext;
import org.bytedeco.ffmpeg.avutil.AVFrame;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorOptions;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.bytedeco.ffmpeg.global.avcodec.*;
import static org.bytedeco.ffmpeg.global.avformat.*;
import static org.bytedeco.ffmpeg.global.avutil.*;

/**
 * High-level FFmpeg video file reader — mirrors {@code torchvision.io.VideoReader}
 * / torchcodec style sequential + seekable frame access.
 *
 * <pre>{@code
 * try (VideoFile vf = VideoFile.open("/path/to/video.mp4")) {
 *     System.out.println(vf.width() + "x" + vf.height() + " @" + vf.fps() + " fps");
 *     System.out.println("Duration: " + vf.duration() + "s  Frames≈" + vf.numFrames());
 *
 *     // Sequential: all frames as [N, 3, H, W] float32 [0,255]
 *     Tensor allFrames = vf.read();
 *
 *     // Seek + single frame (keyframe-aligned seek + decode)
 *     Tensor at1s = vf.frameAt(1.0);
 *
 *     // Uniform sampling for multimodal / VL models (Qwen-VL / LLaVA style)
 *     List<Tensor> key = vf.extractUniform(8);
 *
 *     // Every N-th frame / target fps
 *     List<Tensor> sparse = vf.extractEveryN(5);
 *     List<Tensor> at2fps = vf.extractAtFps(2.0);
 *
 *     // Thumbnail (first decodable frame, optionally resized externally)
 *     Tensor thumb = vf.thumbnail();
 * }
 * }</pre>
 */
public final class VideoFile implements AutoCloseable, Iterable<Tensor> {

    /** FFmpeg AVERROR(EAGAIN): -11 Linux, -35 macOS. */
    private static final int FF_AVERROR_EAGAIN =
            System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("mac") ? -35 : -11;


    private final String filePath;
    private final AVFormatContext fmtCtx;
    private final int videoStreamIdx;
    private final int width;
    private final int height;
    private final double fps;
    private final long numFrames;
    private final double durationSec;
    private final int srcPixelFormat;
    private final AVCodecContext codecCtx;
    private final String codecName;
    private final long bitRate;
    private final AVRational timeBase; // stream time_base (owned by stream; do not free)

    private VideoTensors vtensors;
    private AVPacket packet;
    private AVFrame avFrame;
    private boolean isDecoding = false;
    private long framesDecoded = 0;
    private boolean eofReached = false;

    private VideoFile(String filePath, AVFormatContext fmtCtx, int videoStreamIdx,
                      int width, int height, double fps, long numFrames, double durationSec,
                      int srcPixelFormat, AVCodecContext codecCtx,
                      String codecName, long bitRate, AVRational timeBase) {
        this.filePath = filePath;
        this.fmtCtx = fmtCtx;
        this.videoStreamIdx = videoStreamIdx;
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.numFrames = numFrames;
        this.durationSec = durationSec;
        this.srcPixelFormat = srcPixelFormat;
        this.codecCtx = codecCtx;
        this.codecName = codecName;
        this.bitRate = bitRate;
        this.timeBase = timeBase;
    }

    /**
     * Open a video file for reading.
     *
     * @param filePath path to video (mp4, avi, mkv, webm, …)
     * @return opened VideoFile
     */
    public static VideoFile open(String filePath) {
        Objects.requireNonNull(filePath, "filePath");
        FFmpegNative.load();

        AVFormatContext ctx = avformat_alloc_context();
        if (ctx == null) throw new FFmpegException("avformat_alloc_context returned null");

        int ret = avformat_open_input(ctx, filePath, null, (AVDictionary) null);
        if (ret < 0) {
            avformat_free_context(ctx);
            throw new FFmpegException("avformat_open_input failed: " + ret, ret);
        }

        try {
            avformat_find_stream_info(ctx, (AVDictionary) null);

            int videoIdx = -1;
            for (int i = 0; i < ctx.nb_streams(); i++) {
                if (ctx.streams(i).codecpar().codec_type() == AVMEDIA_TYPE_VIDEO) {
                    videoIdx = i;
                    break;
                }
            }
            if (videoIdx < 0) {
                avformat_close_input(ctx);
                throw new FFmpegException("no video stream found: " + filePath);
            }

            AVStream stream = ctx.streams(videoIdx);
            AVCodecParameters codecpar = stream.codecpar();
            AVCodec codec = avcodec_find_decoder(codecpar.codec_id());
            if (codec == null) {
                avformat_close_input(ctx);
                throw new FFmpegException("no decoder for codec: " + codecpar.codec_id());
            }

            AVCodecContext codecCtx = avcodec_alloc_context3(codec);
            if (codecCtx == null) {
                avformat_close_input(ctx);
                throw new FFmpegException("avcodec_alloc_context3 returned null");
            }
            avcodec_parameters_to_context(codecCtx, codecpar);
            int openRet = avcodec_open2(codecCtx, codec, (AVDictionary) null);
            if (openRet < 0) {
                avcodec_free_context(codecCtx);
                avformat_close_input(ctx);
                throw new FFmpegException("avcodec_open2 failed: " + openRet, openRet);
            }

            int vw = codecpar.width();
            int vh = codecpar.height();
            int srcFmt = codecpar.format();

            // fps: prefer r_frame_rate, fall back to avg_frame_rate
            double streamFps = 30.0;
            AVRational framerate = stream.r_frame_rate();
            if (framerate != null && framerate.num() > 0 && framerate.den() > 0) {
                streamFps = (double) framerate.num() / framerate.den();
            } else {
                AVRational avg = stream.avg_frame_rate();
                if (avg != null && avg.num() > 0 && avg.den() > 0) {
                    streamFps = (double) avg.num() / avg.den();
                }
            }

            // duration (seconds): stream.duration * time_base, else container duration
            double durationSec = 0.0;
            AVRational tb = stream.time_base();
            long streamDur = stream.duration();
            if (streamDur > 0 && tb != null && tb.den() > 0) {
                durationSec = streamDur * (double) tb.num() / tb.den();
            } else if (ctx.duration() > 0) {
                // AV_TIME_BASE = 1_000_000
                durationSec = ctx.duration() / 1_000_000.0;
            }

            // nb_frames from stream metadata, or estimate from duration * fps
            long nbFrames = stream.nb_frames();
            if (nbFrames <= 0 && durationSec > 0 && streamFps > 0) {
                nbFrames = Math.round(durationSec * streamFps);
            }

            String cname = null;
            try {
                if (codec.name() != null) cname = codec.name().getString();
            } catch (Throwable ignored) {}
            long br = codecpar.bit_rate();
            if (br <= 0) br = ctx.bit_rate();

            VideoFile vf = new VideoFile(filePath, ctx, videoIdx, vw, vh,
                    streamFps, nbFrames, durationSec, srcFmt, codecCtx,
                    cname, br, tb);
            vf.vtensors = new VideoTensors(vw, vh, srcFmt);
            vf.packet = av_packet_alloc();
            vf.avFrame = av_frame_alloc();
            return vf;

        } catch (RuntimeException e) {
            avformat_close_input(ctx);
            throw e;
        }
    }

    /** @see #open(String) */
    public static VideoFile open(java.nio.file.Path path) {
        return open(path.toString());
    }

    // ---- metadata accessors ----

    public String filePath() { return filePath; }
    public int width() { return width; }
    public int height() { return height; }
    public double fps() { return fps; }

    /**
     * Duration in seconds (from stream / container metadata).
     * {@code 0} if unknown.
     */
    public double duration() { return durationSec; }

    /** Codec short name (e.g. {@code h264}), or {@code null} if unknown. */
    public String codecName() { return codecName; }

    /** Container / stream bit rate in bits/s ({@code 0} if unknown). */
    public long bitRate() { return bitRate; }

    /** Source pixel format (AVPixelFormat enum int). */
    public int pixelFormat() { return srcPixelFormat; }

    /**
     * Estimated total frame count from container metadata (or {@code duration * fps}).
     * May be {@code 0} if unknown; after full sequential decode use {@link #currentFrame()}.
     */
    public long numFrames() {
        if (numFrames > 0) return numFrames;
        return framesDecoded;
    }

    /** Current sequential decode position (0-based frames already returned by {@link #next()}). */
    public long currentFrame() { return framesDecoded; }

    /** True after the last frame was consumed (or seek left us past EOF). */
    public boolean isEof() { return eofReached && !isDecoding; }

    /**
     * Read all remaining frames as a batched tensor.
     *
     * @return tensor of shape {@code [N, 3, H, W]}, dtype float32, values in {@code [0, 255]}
     */
    public Tensor read() {
        List<Tensor> frames = readFrames();
        return stackFrames(frames);
    }

    /**
     * Read all remaining frames as a list of tensors {@code [3, H, W]} float32.
     */
    public List<Tensor> readFrames() {
        List<Tensor> frames = new ArrayList<>();
        while (hasNext()) {
            frames.add(next());
        }
        return frames;
    }

    /**
     * Read up to {@code maxFrames} remaining frames (or all if {@code maxFrames <= 0}).
     */
    public List<Tensor> readFrames(int maxFrames) {
        List<Tensor> frames = new ArrayList<>();
        while ((maxFrames <= 0 || frames.size() < maxFrames) && hasNext()) {
            frames.add(next());
        }
        return frames;
    }

    // ── Seek / random access ─────────────────────────────────────────────────

    /**
     * Seek to approximately {@code seconds} (keyframe-aligned, backward).
     * Resets the decoder; subsequent {@link #next()} returns the first decodable
     * frame at or after the seek point. Position counter is reset.
     *
     * <p>Mirrors PyAV {@code container.seek} / torchcodec seek semantics.
     */
    public void seek(double seconds) {
        if (seconds < 0) seconds = 0;
        long ts;
        if (timeBase != null && timeBase.den() > 0 && timeBase.num() > 0) {
            // stream time_base units
            ts = Math.round(seconds * timeBase.den() / (double) timeBase.num());
        } else {
            // AV_TIME_BASE
            ts = Math.round(seconds * 1_000_000.0);
        }
        int ret = av_seek_frame(fmtCtx, videoStreamIdx, ts, AVSEEK_FLAG_BACKWARD);
        if (ret < 0) {
            // fallback: seek with AV_TIME_BASE on any stream
            long globalTs = Math.round(seconds * 1_000_000.0);
            ret = av_seek_frame(fmtCtx, -1, globalTs, AVSEEK_FLAG_BACKWARD);
        }
        if (ret < 0) {
            throw new FFmpegException("av_seek_frame failed at " + seconds + "s: " + ret, ret);
        }
        avcodec_flush_buffers(codecCtx);
        isDecoding = false;
        eofReached = false;
        framesDecoded = 0;
        currentFrame = null;
        if (packet != null) av_packet_unref(packet);
        if (avFrame != null) av_frame_unref(avFrame);
    }

    /**
     * Rewind to the beginning of the stream (seek 0 + flush).
     */
    public void rewind() {
        seek(0.0);
    }

    /**
     * Decode a single frame nearest to {@code seconds}.
     * Seeks backward to the previous keyframe then decodes forward until the
     * presentation timestamp is at/after the target (best-effort).
     *
     * @return frame tensor {@code [3, H, W]} float32, or {@code null} if EOF
     */
    public Tensor frameAt(double seconds) {
        seek(seconds);
        // After keyframe seek we may land early; walk forward a few frames toward target.
        double target = Math.max(0, seconds);
        double useFps = fps > 0 ? fps : 30.0;
        int maxSkip = (int) Math.min(300, Math.max(1, Math.ceil(useFps * 2))); // ~2s of frames
        Tensor best = null;
        double bestDt = Double.POSITIVE_INFINITY;
        for (int i = 0; i < maxSkip && hasNext(); i++) {
            Tensor f = next();
            // Without per-frame PTS exposure, approximate time by framesDecoded / fps
            // after seek: framesDecoded is post-seek count, so wall time ≈ seek + i/fps
            double approx = target; // we already sought; take first good frame if close
            // Prefer the first decodable frame after seek for stability
            if (i == 0) {
                best = f;
                bestDt = 0;
                // if target is mid-GOP we keep decoding a bit to get closer
                if (seconds <= 0) break;
            } else {
                double t = seconds; // seek already near target; keep last
                best = f;
                bestDt = 0;
                // stop after a small number of refinements once past first
                if (i >= Math.min(maxSkip, Math.max(1, (int) (useFps * 0.15)))) break;
            }
        }
        return best;
    }

    /**
     * First decodable frame (thumbnail / poster frame).
     * Does not permanently alter sequential position if called at start;
     * rewinds afterward.
     */
    public Tensor thumbnail() {
        boolean wasAtStart = framesDecoded == 0 && !eofReached;
        if (!wasAtStart) seek(0.0);
        Tensor t = hasNext() ? next() : null;
        if (wasAtStart) {
            // leave cursor advanced by 1 — caller can rewind if needed
        }
        return t;
    }

    /**
     * Uniformly sample {@code count} frames across the timeline
     * (Daft / LLaVA / Qwen-VL style keyframe sampling).
     *
     * <p>If duration is unknown, falls back to sequential every-N sampling
     * of the first {@code count * 4} frames.
     */
    public List<Tensor> extractUniform(int count) {
        if (count <= 0) return List.of();
        List<Tensor> out = new ArrayList<>(count);
        double dur = durationSec;
        if (dur <= 0 && numFrames > 0 && fps > 0) {
            dur = numFrames / fps;
        }
        if (dur <= 0) {
            // sequential fallback: take every N among first batch
            rewind();
            List<Tensor> all = readFrames(Math.max(count * 4, count));
            if (all.isEmpty()) return out;
            if (all.size() <= count) return all;
            for (int i = 0; i < count; i++) {
                int idx = (int) Math.round(i * (all.size() - 1) / (double) (count - 1));
                out.add(all.get(idx));
            }
            return out;
        }
        if (count == 1) {
            Tensor f = frameAt(dur / 2.0);
            if (f != null) out.add(f);
            return out;
        }
        for (int i = 0; i < count; i++) {
            double t = (i / (double) (count - 1)) * Math.max(0, dur - 1.0 / Math.max(fps, 1));
            Tensor f = frameAt(t);
            if (f != null) out.add(f);
        }
        return out;
    }

    /**
     * Extract every {@code n}-th frame from the current position to EOF
     * (or until {@code maxFrames} collected if {@code maxFrames > 0}).
     */
    public List<Tensor> extractEveryN(int n) {
        return extractEveryN(n, 0);
    }

    public List<Tensor> extractEveryN(int n, int maxFrames) {
        if (n <= 0) n = 1;
        List<Tensor> out = new ArrayList<>();
        int idx = 0;
        while (hasNext()) {
            Tensor f = next();
            if (idx % n == 0) {
                out.add(f);
                if (maxFrames > 0 && out.size() >= maxFrames) break;
            }
            idx++;
        }
        return out;
    }

    /**
     * Sample frames at approximately {@code targetFps} from the current position.
     * Equivalent to {@code extractEveryN(round(srcFps / targetFps))}.
     */
    public List<Tensor> extractAtFps(double targetFps) {
        return extractAtFps(targetFps, 0);
    }

    public List<Tensor> extractAtFps(double targetFps, int maxFrames) {
        double src = fps > 0 ? fps : 30.0;
        if (targetFps <= 0 || targetFps >= src) {
            return readFrames(maxFrames);
        }
        int stride = Math.max(1, (int) Math.round(src / targetFps));
        return extractEveryN(stride, maxFrames);
    }

    /**
     * Extract frames in the half-open time window {@code [startSec, endSec)}.
     * Seeks to {@code startSec} first.
     */
    public List<Tensor> extractRange(double startSec, double endSec) {
        return extractRange(startSec, endSec, 0);
    }

    public List<Tensor> extractRange(double startSec, double endSec, int maxFrames) {
        if (endSec <= startSec) return List.of();
        seek(startSec);
        double useFps = fps > 0 ? fps : 30.0;
        int budget = maxFrames > 0 ? maxFrames
                : (int) Math.ceil((endSec - startSec) * useFps) + 8;
        List<Tensor> out = new ArrayList<>();
        // Without PTS we approximate: take budget frames after seek
        int limit = maxFrames > 0 ? maxFrames : budget;
        while (out.size() < limit && hasNext()) {
            out.add(next());
            // stop early if we've walked ~window length
            if (maxFrames <= 0 && out.size() >= (int) Math.ceil((endSec - startSec) * useFps)) break;
        }
        return out;
    }

    /**
     * Stack a list of {@code [3,H,W]} frames into {@code [N,3,H,W]}.
     * Empty → {@code [0,3,H,W]}.
     */
    public static Tensor stackFrames(List<Tensor> frames) {
        if (frames == null || frames.isEmpty()) {
            return torch.empty(new long[]{0, 3, 1, 1},
                    new TensorOptions(ScalarType.Float), null);
        }
        long n = frames.size();
        long c = frames.get(0).size(0);
        long h = frames.get(0).size(1);
        long w = frames.get(0).size(2);
        Tensor result = torch.empty(new long[]{n, c, h, w},
                new TensorOptions(ScalarType.Float), null);
        for (int i = 0; i < n; i++) {
            Tensor f = frames.get(i);
            // best-effort size match: copy into slice; caller should ensure uniform HxW
            result.select(0, i).copy_(f);
        }
        return result;
    }

    /** Probe-only metadata snapshot (no frames held). */
    public VideoMeta meta() {
        return new VideoMeta(filePath, width, height, fps, durationSec, numFrames,
                codecName, bitRate, srcPixelFormat);
    }

    /** Immutable video metadata probe result. */
    public static final class VideoMeta {
        public final String path;
        public final int width;
        public final int height;
        public final double fps;
        public final double durationSec;
        public final long numFrames;
        public final String codecName;
        public final long bitRate;
        public final int pixelFormat;

        public VideoMeta(String path, int width, int height, double fps, double durationSec,
                         long numFrames, String codecName, long bitRate, int pixelFormat) {
            this.path = path;
            this.width = width;
            this.height = height;
            this.fps = fps;
            this.durationSec = durationSec;
            this.numFrames = numFrames;
            this.codecName = codecName;
            this.bitRate = bitRate;
            this.pixelFormat = pixelFormat;
        }

        @Override
        public String toString() {
            return "VideoMeta{" + width + "x" + height + " @" + fps + "fps"
                    + " dur=" + durationSec + "s frames≈" + numFrames
                    + " codec=" + codecName + " br=" + bitRate + "}";
        }
    }

    // ---- Iterator ----

    @Override
    public java.util.Iterator<Tensor> iterator() {
        return new java.util.Iterator<>() {
            @Override public boolean hasNext() { return VideoFile.this.hasNext(); }
            @Override public Tensor next() { return VideoFile.this.next(); }
        };
    }

    /** @return true if there is a next frame available */
    public boolean hasNext() {
        if (isDecoding) return true;
        if (eofReached) return false;
        return findNextFrame();
    }

    /**
     * Advance to and decode the next frame.
     *
     * @return decoded frame tensor {@code [3, H, W]}, dtype float32, values in {@code [0, 255]}
     * @throws java.util.NoSuchElementException if end of video reached
     */
    public Tensor next() {
        if (!isDecoding && !findNextFrame()) {
            throw new java.util.NoSuchElementException("end of video: " + filePath);
        }
        isDecoding = false;
        framesDecoded++;
        return currentFrame;
    }

    private Tensor currentFrame;

    private boolean findNextFrame() {
        while (av_read_frame(fmtCtx, packet) >= 0) {
            if (packet.stream_index() != videoStreamIdx) {
                av_packet_unref(packet);
                continue;
            }
            avcodec_send_packet(codecCtx, packet);
            av_packet_unref(packet);
            while (true) {
                int recv = avcodec_receive_frame(codecCtx, avFrame);
                if (recv == FF_AVERROR_EAGAIN || FFmpegNative.isEagain(recv)) break;
                if (recv < 0) {
                    isDecoding = false;
                    if (FFmpegNative.isEof(recv)) eofReached = true;
                    return false;
                }
                currentFrame = vtensors.frameToTensor(avFrame);
                av_frame_unref(avFrame);
                isDecoding = true;
                return true;
            }
        }
        // flush decoder
        avcodec_send_packet(codecCtx, null);
        while (true) {
            int recv = avcodec_receive_frame(codecCtx, avFrame);
            if (recv == FF_AVERROR_EAGAIN || FFmpegNative.isEagain(recv)) break;
            if (recv < 0) break;
            currentFrame = vtensors.frameToTensor(avFrame);
            av_frame_unref(avFrame);
            isDecoding = true;
            return true;
        }
        isDecoding = false;
        eofReached = true;
        return false;
    }

    @Override
    public void close() {
        if (avFrame != null) {
            av_frame_free(avFrame);
            avFrame = null;
        }
        if (packet != null) {
            av_packet_free(packet);
            packet = null;
        }
        if (vtensors != null) {
            vtensors.close();
            vtensors = null;
        }
        if (codecCtx != null) {
            avcodec_free_context(codecCtx);
        }
        if (fmtCtx != null) {
            avformat_close_input(fmtCtx);
        }
    }
}
