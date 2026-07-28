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

import org.bytedeco.ffmpeg.avutil.AVRational;

import org.bytedeco.ffmpeg.avutil.AVDictionary;

import org.bytedeco.javacpp.Pointer;

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
 * High-level FFmpeg audio file reader — mirrors {@code torchaudio.load}.
 *
 * <p>Output waveform tensor shape: {@code [channels, time]}, dtype float32.
 *
 * <pre>{@code
 * try (AudioFile af = AudioFile.open("/path/to/audio.mp3")) {
 *     System.out.println(af.sampleRate() + " Hz, " + af.channels() + " ch");
 *     System.out.println("Duration: " + af.durationSec() + " s");
 *
 *     Tensor wave = af.read();        // all samples [C, T]
 *     Tensor chunk = af.read(0, 16000); // first 1 second at 16 kHz
 * }
 * }</pre>
 */
public final class AudioFile implements AutoCloseable {

    /** FFmpeg AVERROR(EAGAIN): -11 Linux, -35 macOS. */
    private static final int FF_AVERROR_EAGAIN =
            System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("mac") ? -35 : -11;


    private final String filePath;
    private final AVFormatContext fmtCtx;
    private final int audioStreamIdx;
    private final int sampleRate;
    private final int channels;
    private final long numSamples;
    private final double durationSec;
    private final AVCodecContext codecCtx;
    private final AudioTensorsFFmpeg decoder;

    private AVPacket packet;
    private AVFrame frame;
    private List<Float> sampleBuffer;     // accumulated samples
    private boolean flushed = false;

    private AudioFile(String filePath, AVFormatContext fmtCtx, int audioStreamIdx,
                      int sampleRate, int channels, long numSamples, double durationSec,
                      AVCodecContext codecCtx, AudioTensorsFFmpeg decoder) {
        this.filePath = filePath;
        this.fmtCtx = fmtCtx;
        this.audioStreamIdx = audioStreamIdx;
        this.sampleRate = sampleRate;
        this.channels = channels;
        this.numSamples = numSamples;
        this.durationSec = durationSec;
        this.codecCtx = codecCtx;
        this.decoder = decoder;
        this.sampleBuffer = new ArrayList<>();
    }

    /**
     * Open an audio file for reading.
     *
     * @param filePath path to audio/video file (mp3, wav, flac, aac, …)
     * @return opened AudioFile
     */
    public static AudioFile open(String filePath) {
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

            int audioIdx = -1;
            for (int i = 0; i < ctx.nb_streams(); i++) {
                if (ctx.streams(i).codecpar().codec_type() == AVMEDIA_TYPE_AUDIO) {
                    audioIdx = i;
                    break;
                }
            }
            if (audioIdx < 0) {
                throw new FFmpegException("no audio stream found: " + filePath);
            }

            AVStream stream = ctx.streams(audioIdx);
            AVCodecParameters codecpar = stream.codecpar();
            AVCodec codec = avcodec_find_decoder(codecpar.codec_id());
            if (codec == null) throw new FFmpegException("no decoder for codec: " + codecpar.codec_id());

            AVCodecContext codecCtx = avcodec_alloc_context3(codec);
            if (codecCtx == null) throw new FFmpegException("avcodec_alloc_context3 returned null");
            avcodec_parameters_to_context(codecCtx, codecpar);
            int openRet = avcodec_open2(codecCtx, codec, (AVDictionary) null);
            if (openRet < 0) throw new FFmpegException("avcodec_open2 failed: " + openRet, openRet);

            int sr = codecpar.sample_rate();
            if (sr <= 0) sr = codecCtx.sample_rate();
            if (sr <= 0) sr = 44100;
            int ch = codecCtx.ch_layout().nb_channels();
            if (ch <= 0) ch = 2;

            long nbSamples = 0;
            double durSec = 0.0;
            if (stream.duration() > 0 && stream.time_base().den() > 0) {
                long dur = stream.duration();
                int tbNum = stream.time_base().num();
                int tbDen = stream.time_base().den();
                durSec = (double) dur * tbNum / tbDen;
                nbSamples = (long) (durSec * sr);
            }

            AudioTensorsFFmpeg dec = new AudioTensorsFFmpeg(sr, ch);
            AudioFile af = new AudioFile(filePath, ctx, audioIdx, sr, ch,
                    nbSamples, durSec, codecCtx, dec);
            af.packet = av_packet_alloc();
            af.frame = av_frame_alloc();
            return af;

        } catch (RuntimeException e) {
            avformat_close_input(ctx);
            throw e;
        }
    }

    /** @see #open(String) */
    public static AudioFile open(java.nio.file.Path path) {
        return open(path.toString());
    }

    // ---- metadata ----

    public String filePath() { return filePath; }
    public int sampleRate() { return sampleRate; }
    public int channels() { return channels; }
    public long numSamples() { return numSamples; }
    public double durationSec() {
        if (durationSec > 0) return durationSec;
        return (double) sampleBuffer.size() / channels / sampleRate;
    }

    /**
     * Read all remaining samples as a waveform tensor.
     *
     * @return tensor {@code [channels, time]}, dtype float32
     */
    public Tensor read() {
        drainAll();
        return bufferToTensor(readPos, sampleBuffer.size());
    }

    /**
     * Read a slice of samples (sample index is per multi-channel frame, i.e. time steps).
     *
     * @param offsetSamples number of time-steps to skip from start
     * @param countSamples  max number of time-steps to return (-1 = all remaining)
     * @return tensor {@code [channels, min(count, remaining)]}, dtype float32
     */
    public Tensor read(long offsetSamples, long countSamples) {
        if (offsetSamples < 0) offsetSamples = 0;
        if (countSamples < 0) countSamples = Long.MAX_VALUE;

        long needFloats = (offsetSamples + countSamples) * channels;
        while (!flushed && sampleBuffer.size() < needFloats) {
            if (fillBuffer() == 0 && flushed) break;
        }

        int start = (int) Math.min(offsetSamples * channels, sampleBuffer.size());
        int end;
        if (countSamples == Long.MAX_VALUE) {
            end = sampleBuffer.size();
        } else {
            end = (int) Math.min(sampleBuffer.size(),
                    offsetSamples * channels + countSamples * channels);
        }
        // align to channel boundary
        start = (start / channels) * channels;
        end = (end / channels) * channels;
        return bufferToTensor(start, end);
    }

    /** Decode the entire stream into {@link #sampleBuffer} (idempotent). */
    private void drainAll() {
        while (!flushed) {
            if (fillBuffer() == 0 && flushed) break;
        }
    }

    /**
     * Build {@code [C, T]} tensor from planar-ish buffer range {@code [start, end)}.
     * Buffer layout from {@link #appendChunk} is channel-first planar concatenated per chunk;
     * across chunks we store each chunk as C*T floats in channel-first order, so the global
     * buffer is a sequence of planar chunks. For simplicity and correctness we re-pack by
     * treating the whole buffer as planar only when a single chunk was written; otherwise
     * we store interleaved floats in appendChunk (see below).
     */
    private Tensor bufferToTensor(int start, int end) {
        int nFloats = Math.max(0, end - start);
        int n = channels <= 0 ? 0 : nFloats / channels;
        nFloats = n * channels;
        float[] flat = new float[nFloats];
        for (int i = 0; i < nFloats; i++) {
            flat[i] = sampleBuffer.get(start + i);
        }
        // flat is interleaved LRLR… from appendChunk; de-interleave to [C,T]
        float[] planar = new float[nFloats];
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < channels; c++) {
                planar[c * n + t] = flat[t * channels + c];
            }
        }
        Tensor t = torch.empty(new long[]{channels, n},
                new TensorOptions(ScalarType.Float), null);
        if (nFloats > 0) {
            t.copy_(torch.tensor(planar).reshape(channels, n));
        }
        return t;
    }

    private int readPos = 0; // index into sampleBuffer for streaming next()

    /** @return true if there are more interleaved samples available via {@link #next()} */
    public boolean hasNext() {
        if (readPos < sampleBuffer.size()) return true;
        if (flushed) return false;
        return fillBuffer() > 0;
    }

    /**
     * Advance and return the next interleaved sample value (channel-major over time:
     * L0,R0,L1,R1,…). Prefer {@link #read()} for batch access.
     */
    public float next() {
        if (!hasNext()) {
            throw new java.util.NoSuchElementException("end of audio: " + filePath);
        }
        return sampleBuffer.get(readPos++);
    }

    /**
     * Pull more decoded samples into {@link #sampleBuffer}.
     *
     * @return number of float samples newly appended (0 if nothing new / EOF)
     */
    private int fillBuffer() {
        if (flushed) return 0;
        int before = sampleBuffer.size();

        while (av_read_frame(fmtCtx, packet) >= 0) {
            if (packet.stream_index() != audioStreamIdx) {
                av_packet_unref(packet);
                continue;
            }
            avcodec_send_packet(codecCtx, packet);
            while (true) {
                int recv = avcodec_receive_frame(codecCtx, frame);
                if (recv == FF_AVERROR_EAGAIN) break;
                if (recv < 0) {
                    av_packet_unref(packet);
                    // hard error — treat as end
                    flushed = true;
                    return sampleBuffer.size() - before;
                }
                Tensor chunk = decoder.frameToTensor(frame);
                appendChunk(chunk);
                av_frame_unref(frame);
            }
            av_packet_unref(packet);
            int added = sampleBuffer.size() - before;
            if (added > 0) return added; // return promptly so streaming can consume
        }

        // Flush decoder
        avcodec_send_packet(codecCtx, null);
        while (true) {
            int recv = avcodec_receive_frame(codecCtx, frame);
            if (recv == FF_AVERROR_EAGAIN || recv < 0) break;
            Tensor chunk = decoder.frameToTensor(frame);
            appendChunk(chunk);
            av_frame_unref(frame);
        }
        flushed = true;
        return sampleBuffer.size() - before;
    }

    /** Append frame tensor {@code [C,T]} as interleaved floats into {@link #sampleBuffer}. */
    private void appendChunk(Tensor chunk) {
        Tensor cpu = chunk.contiguous().cpu().to(ScalarType.Float);
        long[] shape = new long[(int) cpu.dim()];
        for (int i = 0; i < shape.length; i++) shape[i] = cpu.size(i);
        org.bytedeco.javacpp.FloatPointer fp = cpu.data_ptr_float();
        if (shape.length == 2) {
            int c = (int) shape[0];
            int t = (int) shape[1];
            // de-planar → interleaved
            for (int i = 0; i < t; i++) {
                for (int ch = 0; ch < c; ch++) {
                    sampleBuffer.add(fp.get((long) ch * t + i));
                }
            }
        } else {
            long n = cpu.numel();
            for (int i = 0; i < n; i++) {
                sampleBuffer.add(fp.get(i));
            }
        }
    }

    @Override
    public void close() {
        if (frame != null) av_frame_free(frame);
        if (packet != null) av_packet_free(packet);
        if (decoder != null) decoder.close();
        if (codecCtx != null) avcodec_free_context(codecCtx);
        if (fmtCtx != null) avformat_close_input(fmtCtx);
    }
}
