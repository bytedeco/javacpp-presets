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
package org.bytedeco.pytorch.vision.ffmpeg;

import org.bytedeco.ffmpeg.avformat.AVFormatContext;

import org.bytedeco.ffmpeg.avutil.AVDictionary;

import org.bytedeco.ffmpeg.swscale.SwsContext;
import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Pointer;
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
import static org.bytedeco.ffmpeg.global.swscale.*;

/**
 * Converts decoded FFmpeg video frames to PyTorch tensors.
 *
 * <p>Output tensor shape: {@code [3, H, W]}, dtype float32, values in range {@code [0, 255]}.
 *
 * <p>Convenience method — decode all frames from a file:
 * <pre>{@code
 * List<Tensor> frames = VideoTensors.decodeAllFrames("/path/to/video.mp4");
 * }</pre>
 */
public class VideoTensors implements AutoCloseable {

    private static final int FF_AVERROR_EAGAIN =
            System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("mac") ? -35 : -11;


    /** Output pixel format: AV_PIX_FMT_RGB24 = 3 */
    public static final int TARGET_PIX_FMT = 3;

    private final int width;
    private final int height;
    private final Pointer rgbBuffer;  // owned; freed on close
    private final SwsContext swsCtx;

    /**
     * @param width           frame width in pixels
     * @param height          frame height in pixels
     * @param srcPixelFormat  native source pixel format (e.g. {@code AV_PIX_FMT_YUV420P});
     *                        pass {@code -1} to default to YUV420P
     */
    public VideoTensors(int width, int height, int srcPixelFormat) {
        this.width = width;
        this.height = height;

        // Allocate RGB24 buffer: 3 bytes/pixel
        long frameBytes = (long) width * height * 3;
        this.rgbBuffer = av_malloc(frameBytes);
        if (rgbBuffer == null) {
            throw new FFmpegException("av_malloc(" + frameBytes + ") returned null");
        }

        // Build swscale context: src fmt → RGB24
        int actualSrc = srcPixelFormat >= 0 ? srcPixelFormat : AV_PIX_FMT_YUV420P;
        this.swsCtx = sws_getContext(
                width, height, actualSrc,
                width, height, TARGET_PIX_FMT,
                SWS_BILINEAR, null, null, (org.bytedeco.javacpp.DoublePointer) null);
        if (swsCtx == null) {
            av_free(rgbBuffer);
            throw new FFmpegException("sws_getContext returned null: "
                    + width + "x" + height + " srcFmt=" + actualSrc + " dstFmt=RGB24");
        }
    }

    /**
     * Convert one decoded FFmpeg frame to a float32 RGB tensor.
     *
     * @param frame native AVFrame from {@code avcodec_receive_frame()}
     * @return new Tensor with shape {@code [3, H, W]}, dtype float32, values {@code [0, 255]}
     */
    public Tensor frameToTensor(org.bytedeco.ffmpeg.avutil.AVFrame frame) {
        Objects.requireNonNull(frame, "frame");

        // src/dst via PointerPointer + IntPointer (javacpp overloads)
        org.bytedeco.javacpp.PointerPointer<BytePointer> srcPP =
                new org.bytedeco.javacpp.PointerPointer<>(8);
        org.bytedeco.javacpp.IntPointer srcLS = new org.bytedeco.javacpp.IntPointer(8);
        for (int i = 0; i < 8; i++) {
            srcPP.put(i, frame.data(i));
            srcLS.put(i, frame.linesize(i));
        }
        org.bytedeco.javacpp.PointerPointer<BytePointer> dstPP =
                new org.bytedeco.javacpp.PointerPointer<>(1);
        org.bytedeco.javacpp.IntPointer dstLS = new org.bytedeco.javacpp.IntPointer(1);
        BytePointer dstBuf = new BytePointer(rgbBuffer).capacity((long) width * height * 3);
        dstPP.put(0, dstBuf);
        dstLS.put(0, width * 3);

        int ret = sws_scale(swsCtx, srcPP, srcLS, 0, height, dstPP, dstLS);
        if (ret < 0) {
            throw new FFmpegException("sws_scale failed: " + ret, ret);
        }

        // Read RGB bytes, convert to CHW float32
        Tensor t = torch.empty(new long[]{3, height, width},
                new TensorOptions(ScalarType.Float), null);

        BytePointer bp = new BytePointer(rgbBuffer).capacity((long) width * height * 3);
        long n = (long) 3 * height * width;
        float[] flat = new float[(int) n];
        int idx = 0;
        for (int y = 0; y < height; y++) {
            int rowBase = y * width * 3;
            for (int x = 0; x < width; x++) {
                flat[idx++] = (bp.get(rowBase + x * 3    )) & 0xFF; // R
                flat[idx++] = (bp.get(rowBase + x * 3 + 1)) & 0xFF; // G
                flat[idx++] = (bp.get(rowBase + x * 3 + 2)) & 0xFF; // B
            }
        }

        // Copy into tensor via from_blob (no extra copy)
        Tensor src = torch.tensor(flat).reshape(new long[]{3, height, width});
        t.copy_(src);
        return t;
    }

    /**
     * Decode all video frames from a file.
     *
     * @param filePath path to a video file (mp4, avi, mkv, webm, …)
     * @return list of RGB tensors, one per decoded frame, in temporal order
     */
    public static List<Tensor> decodeAllFrames(String filePath) {
        List<Tensor> frames = new ArrayList<>();

        AVFormatContext fmtCtx = avformat_alloc_context();
        if (fmtCtx == null) throw new FFmpegException("avformat_alloc_context returned null");

        try {
            int ret = avformat_open_input(fmtCtx, filePath, null, (AVDictionary) null);
            if (ret < 0) throw new FFmpegException("avformat_open_input failed: " + ret, ret);

            try {
                avformat_find_stream_info(fmtCtx, (AVDictionary) null);

                int videoIdx = -1;
                for (int i = 0; i < fmtCtx.nb_streams(); i++) {
                    if (fmtCtx.streams(i).codecpar().codec_type() == AVMEDIA_TYPE_VIDEO) {
                        videoIdx = i;
                        break;
                    }
                }
                if (videoIdx < 0) throw new FFmpegException("no video stream found: " + filePath);

                org.bytedeco.ffmpeg.avformat.AVStream stream = fmtCtx.streams(videoIdx);
                org.bytedeco.ffmpeg.avcodec.AVCodecParameters codecpar = stream.codecpar();
                org.bytedeco.ffmpeg.avcodec.AVCodec codec = avcodec_find_decoder(codecpar.codec_id());
                if (codec == null) throw new FFmpegException("no decoder for codec: " + codecpar.codec_id());

                org.bytedeco.ffmpeg.avcodec.AVCodecContext codecCtx = avcodec_alloc_context3(codec);
                if (codecCtx == null) throw new FFmpegException("avcodec_alloc_context3 returned null");
                avcodec_parameters_to_context(codecCtx, codecpar);
                int openRet = avcodec_open2(codecCtx, codec, (AVDictionary) null);
                if (openRet < 0) throw new FFmpegException("avcodec_open2 failed: " + openRet, openRet);

                int srcFmt = codecpar.format();
                int vw = codecpar.width();
                int vh = codecpar.height();

                try (VideoTensors vt = new VideoTensors(vw, vh, srcFmt)) {
                    org.bytedeco.ffmpeg.avcodec.AVPacket packet = av_packet_alloc();
                    org.bytedeco.ffmpeg.avutil.AVFrame frame = av_frame_alloc();

                    try {
                        while (av_read_frame(fmtCtx, packet) >= 0) {
                            if (packet.stream_index() == videoIdx) {
                                avcodec_send_packet(codecCtx, packet);
                                while (true) {
                                    int recv = avcodec_receive_frame(codecCtx, frame);
                                    if (recv == FF_AVERROR_EAGAIN) break;
                                    if (recv < 0) {
                                        av_packet_unref(packet);
                                        throw new FFmpegException("avcodec_receive_frame: " + recv, recv);
                                    }
                                    frames.add(vt.frameToTensor(frame));
                                    av_frame_unref(frame);
                                }
                            }
                            av_packet_unref(packet);
                        }
                        // flush
                        avcodec_send_packet(codecCtx, null);
                        while (true) {
                            int recv = avcodec_receive_frame(codecCtx, frame);
                            if (recv == FF_AVERROR_EAGAIN) break;
                            if (recv < 0) break;
                            frames.add(vt.frameToTensor(frame));
                            av_frame_unref(frame);
                        }
                    } finally {
                        av_frame_free(frame);
                        av_packet_free(packet);
                    }
                }
                avcodec_free_context(codecCtx);
            } finally {
                avformat_close_input(fmtCtx);
            }
        } finally {
            avformat_free_context(fmtCtx);
        }
        return frames;
    }

    @Override
    public void close() {
        if (swsCtx != null) sws_freeContext(swsCtx);
        if (rgbBuffer != null) av_free(rgbBuffer);
    }
}
