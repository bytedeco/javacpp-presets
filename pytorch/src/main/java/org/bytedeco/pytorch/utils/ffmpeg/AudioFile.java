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
        while (hasNext()) {
            next();
        }
        return bufferToTensor();
    }

    /**
     * Read a slice of samples.
     *
     * @param offsetSamples number of samples to skip from start
     * @param countSamples  max number of samples to return (-1 = all remaining)
     * @return tensor {@code [channels, min(count, remaining)]}, dtype float32
     */
    public Tensor read(long offsetSamples, long countSamples) {
        if (offsetSamples < 0) offsetSamples = 0;
        if (countSamples < 0) countSamples = Long.MAX_VALUE;

        while (hasNext() && (long) sampleBuffer.size() < offsetSamples + countSamples) {
            next();
        }

        int start = (int) Math.min(offsetSamples, sampleBuffer.size());
        int end = (int) Math.min(sampleBuffer.size(), offsetSamples + countSamples);
        int n = Math.max(0, end - start);

        float[] flat = new float[n * channels];
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < channels; c++) {
                flat[i * channels + c] = sampleBuffer.get(start + i);
            }
        }
        Tensor t = torch.empty(new long[]{channels, n},
                new TensorOptions(ScalarType.Float), null);
        t.copy_(torch.tensor(flat).reshape(channels, n));
        return t;
    }

    private Tensor bufferToTensor() {
        int n = sampleBuffer.size() / channels;
        float[] flat = new float[sampleBuffer.size()];
        for (int i = 0; i < sampleBuffer.size(); i++) {
            flat[i] = sampleBuffer.get(i);
        }
        Tensor t = torch.empty(new long[]{channels, n},
                new TensorOptions(ScalarType.Float), null);
        t.copy_(torch.tensor(flat).reshape(channels, n));
        return t;
    }

    private Float currentSample;

    /** @return true if there are more samples available */
    public boolean hasNext() {
        return flushed || currentSample != null || fillBuffer() > 0;
    }

    /**
     * Advance and return the next sample value (interleaved, all channels).
     * Prefer {@link #read()} for batch access.
     */
    public float next() {
        if (currentSample != null) {
            float s = currentSample;
            currentSample = null;
            return s;
        }
        if (fillBuffer() == 0) throw new java.util.NoSuchElementException("end of audio: " + filePath);
        return next();
    }

    private int fillBuffer() {
        if (flushed) return 0;

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
                    return 0;
                }
                Tensor chunk = decoder.frameToTensor(frame);
                appendChunk(chunk);
                av_frame_unref(frame);
            }
            av_packet_unref(packet);
        }

        // Flush
        avcodec_send_packet(codecCtx, null);
        while (true) {
            int recv = avcodec_receive_frame(codecCtx, frame);
            if (recv == FF_AVERROR_EAGAIN) break;
            if (recv < 0) break;
            Tensor chunk = decoder.frameToTensor(frame);
            appendChunk(chunk);
            av_frame_unref(frame);
        }
        flushed = true;
        return 0;
    }

    private void appendChunk(Tensor chunk) {
        Tensor cpu = chunk.contiguous().cpu();
        org.bytedeco.javacpp.FloatPointer fp = cpu.data_ptr_float();
        long n = cpu.numel();
        for (int i = 0; i < n; i++) {
            sampleBuffer.add(fp.get(i));
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
