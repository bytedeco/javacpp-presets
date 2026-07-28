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

import org.bytedeco.ffmpeg.avformat.AVFormatContext;

import org.bytedeco.ffmpeg.avutil.AVRational;

import org.bytedeco.ffmpeg.avutil.AVDictionary;

import org.bytedeco.ffmpeg.avformat.AVStream;

import org.bytedeco.ffmpeg.avcodec.AVCodecParameters;

import org.bytedeco.ffmpeg.avcodec.AVCodec;

import org.bytedeco.ffmpeg.swresample.SwrContext;
import org.bytedeco.javacpp.FloatPointer;
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
import static org.bytedeco.ffmpeg.global.swresample.*;

/**
 * Decodes audio from FFmpeg into PyTorch waveform tensors.
 *
 * <p>Output tensor shape: {@code [channels, time]}, dtype float32.
 *
 * <pre>{@code
 * // All samples from a file:
 * Tensor wave = AudioTensorsFFmpeg.decodeAllSamples("/path/to/audio.wav");
 *
 * // Low-level:
 * try (AudioTensorsFFmpeg dec = new AudioTensorsFFmpeg(codecCtx, sampleRate, channels)) {
 *     while (av_read_frame(fmtCtx, packet) >= 0) {
 *         if (packet.stream_index() == audioIdx) {
 *             avcodec_send_packet(codecCtx, packet);
 *             while (avcodec_receive_frame(codecCtx, frame) >= 0) {
 *                 Tensor chunk = dec.frameToTensor(frame);
 *                 // accumulate
 *                 av_frame_unref(frame);
 *             }
 *         }
 *         av_packet_unref(packet);
 *     }
 * }
 * }</pre>
 */
public class AudioTensorsFFmpeg implements AutoCloseable {

    private static final int FF_AVERROR_EAGAIN =
            System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("mac") ? -35 : -11;


    private final int sampleRate;
    private final int channels;
    private final SwrContext swrCtx;
    private final long maxSamples;  // preallocated output size per frame

    /**
     * @param sampleRate target output sample rate (Hz)
     * @param channels   number of output channels
     */
    public AudioTensorsFFmpeg(int sampleRate, int channels) {
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.maxSamples = 8192; // default per-frame capacity hint
        // Allocate swr context: input fmt → same fmt, planar float
        this.swrCtx = swr_alloc();
        if (swrCtx == null) throw new FFmpegException("swr_alloc returned null");

        // Set output: planar float, same channels/sampleRate
        swr_set_channel_mapping(swrCtx, (org.bytedeco.javacpp.IntPointer) null);
        // swr is configured per-frame based on the input context
    }

    /**
     * Configure swr context from the actual decoded frame layout.
     * Call this once after the first frame is decoded.
     */
    public void configureFromFrame(org.bytedeco.ffmpeg.avutil.AVFrame frame) {
        if (swrCtx == null) return;
        // Set options for conversion: output to planar float
        // The simplest case: just resample to target rate
        swr_init(swrCtx);
    }

    /**
     * Convert one decoded audio frame to a float32 waveform tensor.
     *
     * <p>Handles common sample formats (FLTP/FLT/S16/S16P/S32/S32P/U8/U8P). Values are
     * normalized to roughly {@code [-1, 1]} for integer source formats.
     *
     * @param frame native AVFrame (e.g. from avcodec_receive_frame)
     * @return Tensor {@code [channels, time]}, dtype float32
     */
    public Tensor frameToTensor(org.bytedeco.ffmpeg.avutil.AVFrame frame) {
        Objects.requireNonNull(frame, "frame");
        int nSamples = frame.nb_samples();
        if (nSamples <= 0) {
            return torch.empty(new long[]{Math.max(1, channels), 0},
                    new TensorOptions(ScalarType.Float), null);
        }
        int frameCh = 0;
        try {
            frameCh = frame.ch_layout().nb_channels();
        } catch (Throwable ignored) {
            // older layouts may not expose ch_layout cleanly
        }
        if (frameCh <= 0) frameCh = Math.max(1, channels);

        int fmt = frame.format();
        float[] planar = decodeFrameToPlanarFloat(frame, frameCh, nSamples, fmt);
        Tensor t = torch.empty(new long[]{frameCh, nSamples},
                new TensorOptions(ScalarType.Float), null);
        t.copy_(torch.tensor(planar).reshape(frameCh, nSamples));
        return t;
    }

    /**
     * Decode AVFrame samples into channel-first planar float {@code [C*T]} layout.
     * Supports planar and interleaved integer/float formats commonly produced by decoders.
     */
    private static float[] decodeFrameToPlanarFloat(org.bytedeco.ffmpeg.avutil.AVFrame frame,
                                                    int ch, int nSamples, int fmt) {
        float[] planar = new float[ch * nSamples];
        // FFmpeg sample format constants (avutil)
        final int AV_SAMPLE_FMT_U8 = 0;
        final int AV_SAMPLE_FMT_S16 = 1;
        final int AV_SAMPLE_FMT_S32 = 2;
        final int AV_SAMPLE_FMT_FLT = 3;
        final int AV_SAMPLE_FMT_DBL = 4;
        final int AV_SAMPLE_FMT_U8P = 5;
        final int AV_SAMPLE_FMT_S16P = 6;
        final int AV_SAMPLE_FMT_S32P = 7;
        final int AV_SAMPLE_FMT_FLTP = 8;
        final int AV_SAMPLE_FMT_DBLP = 9;

        switch (fmt) {
            case AV_SAMPLE_FMT_FLTP: {
                for (int c = 0; c < ch; c++) {
                    FloatPointer plane = new FloatPointer(frame.data(c));
                    for (int s = 0; s < nSamples; s++) {
                        planar[c * nSamples + s] = plane.get(s);
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_FLT: {
                FloatPointer interleaved = new FloatPointer(frame.data(0));
                for (int s = 0; s < nSamples; s++) {
                    for (int c = 0; c < ch; c++) {
                        planar[c * nSamples + s] = interleaved.get((long) s * ch + c);
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_S16P: {
                for (int c = 0; c < ch; c++) {
                    org.bytedeco.javacpp.ShortPointer plane =
                            new org.bytedeco.javacpp.ShortPointer(frame.data(c));
                    for (int s = 0; s < nSamples; s++) {
                        planar[c * nSamples + s] = plane.get(s) / 32768.0f;
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_S16: {
                org.bytedeco.javacpp.ShortPointer interleaved =
                        new org.bytedeco.javacpp.ShortPointer(frame.data(0));
                for (int s = 0; s < nSamples; s++) {
                    for (int c = 0; c < ch; c++) {
                        planar[c * nSamples + s] = interleaved.get((long) s * ch + c) / 32768.0f;
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_S32P: {
                for (int c = 0; c < ch; c++) {
                    org.bytedeco.javacpp.IntPointer plane =
                            new org.bytedeco.javacpp.IntPointer(frame.data(c));
                    for (int s = 0; s < nSamples; s++) {
                        planar[c * nSamples + s] = plane.get(s) / 2147483648.0f;
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_S32: {
                org.bytedeco.javacpp.IntPointer interleaved =
                        new org.bytedeco.javacpp.IntPointer(frame.data(0));
                for (int s = 0; s < nSamples; s++) {
                    for (int c = 0; c < ch; c++) {
                        planar[c * nSamples + s] = interleaved.get((long) s * ch + c) / 2147483648.0f;
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_U8P: {
                for (int c = 0; c < ch; c++) {
                    org.bytedeco.javacpp.BytePointer plane =
                            new org.bytedeco.javacpp.BytePointer(frame.data(c));
                    for (int s = 0; s < nSamples; s++) {
                        planar[c * nSamples + s] = ((plane.get(s) & 0xFF) - 128) / 128.0f;
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_U8: {
                org.bytedeco.javacpp.BytePointer interleaved =
                        new org.bytedeco.javacpp.BytePointer(frame.data(0));
                for (int s = 0; s < nSamples; s++) {
                    for (int c = 0; c < ch; c++) {
                        planar[c * nSamples + s] =
                                ((interleaved.get((long) s * ch + c) & 0xFF) - 128) / 128.0f;
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_DBLP: {
                for (int c = 0; c < ch; c++) {
                    org.bytedeco.javacpp.DoublePointer plane =
                            new org.bytedeco.javacpp.DoublePointer(frame.data(c));
                    for (int s = 0; s < nSamples; s++) {
                        planar[c * nSamples + s] = (float) plane.get(s);
                    }
                }
                break;
            }
            case AV_SAMPLE_FMT_DBL: {
                org.bytedeco.javacpp.DoublePointer interleaved =
                        new org.bytedeco.javacpp.DoublePointer(frame.data(0));
                for (int s = 0; s < nSamples; s++) {
                    for (int c = 0; c < ch; c++) {
                        planar[c * nSamples + s] = (float) interleaved.get((long) s * ch + c);
                    }
                }
                break;
            }
            default: {
                // Best-effort: try planar float, then interleaved s16
                try {
                    for (int c = 0; c < ch; c++) {
                        if (frame.data(c) == null || frame.data(c).isNull()) {
                            throw new IllegalStateException("null plane");
                        }
                        FloatPointer plane = new FloatPointer(frame.data(c));
                        for (int s = 0; s < nSamples; s++) {
                            float v = plane.get(s);
                            if (!Float.isFinite(v)) throw new IllegalStateException("non-finite");
                            planar[c * nSamples + s] = v;
                        }
                    }
                } catch (Throwable t) {
                    org.bytedeco.javacpp.ShortPointer interleaved =
                            new org.bytedeco.javacpp.ShortPointer(frame.data(0));
                    for (int s = 0; s < nSamples; s++) {
                        for (int c = 0; c < ch; c++) {
                            planar[c * nSamples + s] =
                                    interleaved.get((long) s * ch + c) / 32768.0f;
                        }
                    }
                }
                break;
            }
        }
        return planar;
    }

    /**
     * Decode all audio samples from a file into one waveform tensor.
     *
     * @param filePath path to an audio/video file
     * @return Tensor {@code [channels, totalSamples]}, dtype float32
     */
    public static Tensor decodeAllSamples(String filePath) {
        List<float[]> chunks = new ArrayList<>();
        int[] sr = new int[1];
        int[] ch = new int[1];

        AVFormatContext fmtCtx = avformat_alloc_context();
        if (fmtCtx == null) throw new FFmpegException("avformat_alloc_context returned null");

        try {
            int ret = avformat_open_input(fmtCtx, filePath, null, (AVDictionary) null);
            if (ret < 0) throw new FFmpegException("avformat_open_input failed: " + ret, ret);

            try {
                avformat_find_stream_info(fmtCtx, (AVDictionary) null);

                int audioIdx = -1;
                for (int i = 0; i < fmtCtx.nb_streams(); i++) {
                    if (fmtCtx.streams(i).codecpar().codec_type() == AVMEDIA_TYPE_AUDIO) {
                        audioIdx = i;
                        break;
                    }
                }
                if (audioIdx < 0) throw new FFmpegException("no audio stream found: " + filePath);

                org.bytedeco.ffmpeg.avformat.AVStream stream = fmtCtx.streams(audioIdx);
                org.bytedeco.ffmpeg.avcodec.AVCodecParameters codecpar = stream.codecpar();
                org.bytedeco.ffmpeg.avcodec.AVCodec codec = avcodec_find_decoder(codecpar.codec_id());
                if (codec == null) throw new FFmpegException("no decoder for codec: " + codecpar.codec_id());

                org.bytedeco.ffmpeg.avcodec.AVCodecContext codecCtx = avcodec_alloc_context3(codec);
                if (codecCtx == null) throw new FFmpegException("avcodec_alloc_context3 returned null");
                avcodec_parameters_to_context(codecCtx, codecpar);
                int openRet = avcodec_open2(codecCtx, codec, (AVDictionary) null);
                if (openRet < 0) throw new FFmpegException("avcodec_open2 failed: " + openRet, openRet);

                int inSr = codecpar.sample_rate();
                int inCh = codecCtx.ch_layout().nb_channels();

                try (AudioTensorsFFmpeg dec = new AudioTensorsFFmpeg(inSr, inCh)) {
                    org.bytedeco.ffmpeg.avcodec.AVPacket packet = av_packet_alloc();
                    org.bytedeco.ffmpeg.avutil.AVFrame frame = av_frame_alloc();

                    try {
                        while (av_read_frame(fmtCtx, packet) >= 0) {
                            if (packet.stream_index() == audioIdx) {
                                avcodec_send_packet(codecCtx, packet);
                                while (true) {
                                    int recv = avcodec_receive_frame(codecCtx, frame);
                                    if (recv == FF_AVERROR_EAGAIN) break;
                                    if (recv < 0) {
                                        av_packet_unref(packet);
                                        break;
                                    }
                                    Tensor chunk = dec.frameToTensor(frame);
                                    float[] flat = dec.tensorToFloatArray(chunk);
                                    chunks.add(flat);
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
                            Tensor chunk = dec.frameToTensor(frame);
                            float[] flat = dec.tensorToFloatArray(chunk);
                            chunks.add(flat);
                            av_frame_unref(frame);
                        }
                    } finally {
                        av_frame_free(frame);
                        av_packet_free(packet);
                    }
                }
                avcodec_free_context(codecCtx);

                // Concatenate all chunks
                long totalSamples = chunks.stream().mapToLong(a -> a.length / inCh).sum();
                float[] all = new float[(int) (totalSamples * inCh)];
                long pos = 0;
                for (float[] chunk : chunks) {
                    System.arraycopy(chunk, 0, all, (int) pos, chunk.length);
                    pos += chunk.length;
                }
                Tensor result = torch.empty(new long[]{inCh, totalSamples},
                        new TensorOptions(ScalarType.Float), null);
                result.copy_(torch.tensor(all).reshape(inCh, totalSamples));
                return result;

            } finally {
                avformat_close_input(fmtCtx);
            }
        } finally {
            avformat_free_context(fmtCtx);
        }
    }

    private float[] tensorToFloatArray(Tensor t) {
        Tensor cpu = t.contiguous().cpu();
        FloatPointer fp = cpu.data_ptr_float();
        long n = cpu.numel();
        float[] out = new float[(int) n];
        for (int i = 0; i < n; i++) out[i] = fp.get(i);
        return out;
    }

    public int sampleRate() { return sampleRate; }
    public int channels() { return channels; }

    @Override
    public void close() {
        if (swrCtx != null) swr_free(swrCtx);
    }
}
