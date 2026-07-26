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
 * High-level FFmpeg video file reader — mirrors {@code torchvision.io.VideoReader}.
 *
 * <pre>{@code
 * try (VideoFile vf = VideoFile.open("/path/to/video.mp4")) {
 *     System.out.println(vf.width() + "x" + vf.height() + " @" + vf.fps() + " fps");
 *     System.out.println("Frames: " + vf.numFrames());
 *
 *     // Read all frames as tensors [N, 3, H, W] float32 [0,255]
 *     Tensor allFrames = vf.read();
 *
 *     // Or iterate per-frame
 *     for (Tensor frame : vf) {
 *         // frame: [3, H, W] float32
 *     }
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
    private final int srcPixelFormat;
    private final AVCodecContext codecCtx;

    private VideoTensors vtensors;
    private AVPacket packet;
    private AVFrame avFrame;
    private boolean isDecoding = false;
    private long framesDecoded = 0;

    private VideoFile(String filePath, AVFormatContext fmtCtx, int videoStreamIdx,
                      int width, int height, double fps, long numFrames,
                      int srcPixelFormat, AVCodecContext codecCtx) {
        this.filePath = filePath;
        this.fmtCtx = fmtCtx;
        this.videoStreamIdx = videoStreamIdx;
        this.width = width;
        this.height = height;
        this.fps = fps;
        this.numFrames = numFrames;
        this.srcPixelFormat = srcPixelFormat;
        this.codecCtx = codecCtx;
    }

    /**
     * Open a video file for reading.
     *
     * @param filePath path to video (mp4, avi, mkv, webm, …)
     * @return opened VideoFile
     */
    public static VideoFile open(String filePath) {
        Objects.requireNonNull(filePath, "filePath");

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
            double streamFps = 30.0;
            long nbFrames = 0;
            long duration = stream.duration();
            if (duration > 0 && stream.time_base().den() > 0) {
                AVRational tb = stream.time_base();
                // fps estimate from duration
            }
            // fps from r_frame_rate
            AVRational framerate = stream.r_frame_rate();
            if (framerate.num() > 0 && framerate.den() > 0) {
                streamFps = (double) framerate.num() / framerate.den();
            }

            VideoFile vf = new VideoFile(filePath, ctx, videoIdx, vw, vh,
                    streamFps, nbFrames, srcFmt, codecCtx);
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
     * Estimated total frame count (may be -1 if unknown).
     * Accurate only after full decode.
     */
    public long numFrames() {
        if (numFrames > 0) return numFrames;
        return framesDecoded;
    }

    /** Current frame index (0-based). */
    public long currentFrame() { return framesDecoded; }

    /**
     * Read all remaining frames as a batched tensor.
     *
     * @return tensor of shape {@code [N, 3, H, W]}, dtype float32
     */
    public Tensor read() {
        List<Tensor> frames = new ArrayList<>();
        while (hasNext()) {
            frames.add(next());
        }
        if (frames.isEmpty()) {
            return torch.empty(new long[]{0, 3, height, width},
                    new TensorOptions(ScalarType.Float), null);
        }
        long n = frames.size();
        Tensor result = torch.empty(new long[]{n, 3, height, width},
                new TensorOptions(ScalarType.Float), null);
        for (int i = 0; i < n; i++) {
            // Copy frame i into result[i]
            Tensor dstSlice = result.select(0, i);
            dstSlice.copy_(frames.get(i));
        }
        return result;
    }

    /**
     * Read all remaining frames as a list of tensors.
     */
    public List<Tensor> readFrames() {
        List<Tensor> frames = new ArrayList<>();
        while (hasNext()) {
            frames.add(next());
        }
        return frames;
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
        return findNextFrame();
    }

    /**
     * Advance to and decode the next frame.
     *
     * @return decoded frame tensor {@code [3, H, W]}, dtype float32
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
            while (true) {
                int recv = avcodec_receive_frame(codecCtx, avFrame);
                if (recv == FF_AVERROR_EAGAIN) break;
                if (recv < 0) {
                    av_packet_unref(packet);
                    isDecoding = false;
                    return false;
                }
                currentFrame = vtensors.frameToTensor(avFrame);
                av_frame_unref(avFrame);
                av_packet_unref(packet);
                isDecoding = true;
                return true;
            }
        }
        // flush
        avcodec_send_packet(codecCtx, null);
        while (true) {
            int recv = avcodec_receive_frame(codecCtx, avFrame);
            if (recv == FF_AVERROR_EAGAIN) break;
            if (recv < 0) break;
            currentFrame = vtensors.frameToTensor(avFrame);
            av_frame_unref(avFrame);
            isDecoding = true;
            return true;
        }
        isDecoding = false;
        return false;
    }

    @Override
    public void close() {
        if (avFrame != null) av_frame_free(avFrame);
        if (packet != null) av_packet_free(packet);
        if (vtensors != null) vtensors.close();
        if (codecCtx != null) avcodec_free_context(codecCtx);
        if (fmtCtx != null) avformat_close_input(fmtCtx);
    }
}
