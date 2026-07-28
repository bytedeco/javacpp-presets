/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
package org.bytedeco.pytorch.utils.ffmpeg;

import org.bytedeco.ffmpeg.avcodec.AVCodec;
import org.bytedeco.ffmpeg.avcodec.AVCodecContext;
import org.bytedeco.ffmpeg.avcodec.AVCodecParameters;
import org.bytedeco.ffmpeg.avcodec.AVPacket;
import org.bytedeco.ffmpeg.avformat.AVStream;
import org.bytedeco.ffmpeg.avutil.AVDictionary;
import org.bytedeco.ffmpeg.avutil.AVFrame;
import org.bytedeco.ffmpeg.avutil.AVRational;
import org.bytedeco.javacpp.PointerPointer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.ffmpeg.global.avcodec.*;
import static org.bytedeco.ffmpeg.global.avutil.*;

/**
 * Base media stream — PyAV {@code av.stream.Stream}.
 *
 * <p>Thin wrapper over {@link AVStream} + optional encode/decode {@link AVCodecContext}.
 */
public class Stream {

    protected final Container container;
    protected final AVStream avStream;
    protected final int index;
    protected final int type; // AVMEDIA_TYPE_*
    protected Codec codec;
    protected AVCodecContext codecCtx;
    protected boolean codecOpened;
    protected HardwareContext hwContext;

    Stream(Container container, AVStream avStream, int index) {
        this.container = container;
        this.avStream = avStream;
        this.index = index;
        AVCodecParameters par = avStream.codecpar();
        this.type = par != null ? par.codec_type() : AVMEDIA_TYPE_UNKNOWN;
        this.codec = Codec.fromParameters(par, container != null && container.isWritable());
    }

    public Container container() { return container; }
    public int index() { return index; }
    public int type() { return type; }

    public String typeName() {
        switch (type) {
            case AVMEDIA_TYPE_VIDEO: return "video";
            case AVMEDIA_TYPE_AUDIO: return "audio";
            case AVMEDIA_TYPE_SUBTITLE: return "subtitle";
            default: return "unknown";
        }
    }

    public boolean isVideo() { return type == AVMEDIA_TYPE_VIDEO; }
    public boolean isAudio() { return type == AVMEDIA_TYPE_AUDIO; }

    public Codec codec() { return codec; }

    public Rational timeBase() {
        return Rational.of(avStream.time_base());
    }

    public void timeBase(Rational tb) {
        avStream.time_base().num(tb.num);
        avStream.time_base().den(tb.den);
    }

    public long duration() {
        return avStream.duration();
    }

    /** Duration in seconds (duration * time_base), or NaN. */
    public double durationSeconds() {
        long d = duration();
        if (d <= 0 || d == 0x8000000000000000L) return Double.NaN;
        return timeBase().mul(d);
    }

    public AVStream nativeStream() { return avStream; }
    public AVCodecParameters codecpar() { return avStream.codecpar(); }

    /**
     * PyAV {@code stream.thread_type = "AUTO"}.
     * Call before first decode. Values: {@code "NONE"}, {@code "FRAME"}, {@code "SLICE"}, {@code "AUTO"}.
     */
    public void threadType(String type) {
        ensureCodecContext(false);
        if (type == null || "NONE".equalsIgnoreCase(type)) {
            codecCtx.thread_type(0);
            codecCtx.thread_count(1);
            return;
        }
        if ("AUTO".equalsIgnoreCase(type)) {
            codecCtx.thread_type(AVCodecContext.FF_THREAD_FRAME | AVCodecContext.FF_THREAD_SLICE);
            codecCtx.thread_count(0); // auto
        } else if ("FRAME".equalsIgnoreCase(type)) {
            codecCtx.thread_type(AVCodecContext.FF_THREAD_FRAME);
        } else if ("SLICE".equalsIgnoreCase(type)) {
            codecCtx.thread_type(AVCodecContext.FF_THREAD_SLICE);
        }
    }

    /** Attach hardware device context (decode). Best-effort; may no-op. */
    public void hwaccel(HardwareContext hw) {
        this.hwContext = hw;
    }

    public HardwareContext hwaccel() {
        return hwContext;
    }

    // ---- encode path (output containers) ------------------------------------

    /**
     * Encode one frame → zero or more packets.
     * Pass {@code null} to flush. PyAV: {@code stream.encode(frame)}.
     */
    public List<Packet> encode(Frame frame) {
        ensureCodecContext(true);
        openCodecIfNeeded(true);
        List<Packet> out = new ArrayList<>();
        AVFrame raw = frame == null ? null : frame.nativeFrame();
        int ret = avcodec_send_frame(codecCtx, raw);
        if (ret < 0 && !FFmpegNative.isEof(ret)) {
            throw new FFmpegException("avcodec_send_frame", ret);
        }
        while (true) {
            AVPacket pkt = av_packet_alloc();
            ret = avcodec_receive_packet(codecCtx, pkt);
            if (FFmpegNative.isEagain(ret) || FFmpegNative.isEof(ret)) {
                av_packet_free(pkt);
                break;
            }
            if (ret < 0) {
                av_packet_free(pkt);
                throw new FFmpegException("avcodec_receive_packet", ret);
            }
            pkt.stream_index(index);
            // rescale from codec time_base to stream time_base
            av_packet_rescale_ts(pkt, codecCtx.time_base(), avStream.time_base());
            Packet p = new Packet(pkt);
            p.setStream(this);
            out.add(p);
        }
        return out;
    }

    /** Convenience: encode and return first packet or null. */
    public Packet encodeOne(Frame frame) {
        List<Packet> ps = encode(frame);
        if (ps.isEmpty()) return null;
        for (int i = 1; i < ps.size(); i++) ps.get(i).close();
        return ps.get(0);
    }

    // ---- decode helpers used by Container -----------------------------------

    void ensureCodecContext(boolean encoder) {
        if (codecCtx != null && !codecCtx.isNull()) return;
        FFmpegNative.load();
        AVCodecParameters par = avStream.codecpar();
        AVCodec c;
        if (encoder) {
            c = avcodec_find_encoder(par.codec_id());
            if (c == null || c.isNull()) {
                // try by name from our Codec wrapper
                if (codec != null && codec.nativeCodec() != null) c = codec.nativeCodec();
            }
            if (c == null || c.isNull()) throw new FFmpegException("no encoder for codec id " + par.codec_id());
            codec = new Codec(codec != null ? codec.name() : "enc", par.codec_id(), true, c);
        } else {
            c = avcodec_find_decoder(par.codec_id());
            if (c == null || c.isNull()) throw new FFmpegException("no decoder for codec id " + par.codec_id());
            codec = new Codec(FFmpegNative.ptrToString(c.name()), par.codec_id(), false, c);
        }
        codecCtx = avcodec_alloc_context3(c);
        if (codecCtx == null || codecCtx.isNull()) throw new FFmpegException("avcodec_alloc_context3 failed");
        if (!encoder) {
            FFmpegNative.check(avcodec_parameters_to_context(codecCtx, par), "avcodec_parameters_to_context");
        } else {
            // encoder: copy params set on stream into context where relevant
            FFmpegNative.check(avcodec_parameters_to_context(codecCtx, par), "avcodec_parameters_to_context");
            // time_base for encoders
            if (codecCtx.time_base().num() == 0) {
                codecCtx.time_base(avStream.time_base());
            }
        }
    }

    void openCodecIfNeeded(boolean encoder) {
        if (codecOpened) return;
        ensureCodecContext(encoder);
        if (hwContext != null && !encoder) {
            hwContext.attachTo(codecCtx);
        }
        int ret = avcodec_open2(codecCtx, codec.nativeCodec(), (AVDictionary) null);
        FFmpegNative.check(ret, "avcodec_open2");
        codecOpened = true;
        if (encoder) {
            // write back extradata etc.
            FFmpegNative.check(avcodec_parameters_from_context(avStream.codecpar(), codecCtx),
                    "avcodec_parameters_from_context");
        }
    }

    List<Frame> decodePacket(Packet packet) {
        ensureCodecContext(false);
        openCodecIfNeeded(false);
        List<Frame> frames = new ArrayList<>();
        AVPacket raw = packet == null ? null : packet.nativePacket();
        int ret = avcodec_send_packet(codecCtx, raw);
        if (ret < 0 && !FFmpegNative.isEof(ret)) {
            // some codecs return EAGAIN here — ignore
            if (!FFmpegNative.isEagain(ret)) {
                throw new FFmpegException("avcodec_send_packet", ret);
            }
        }
        while (true) {
            AVFrame fr = av_frame_alloc();
            ret = avcodec_receive_frame(codecCtx, fr);
            if (FFmpegNative.isEagain(ret) || FFmpegNative.isEof(ret)) {
                av_frame_free(fr);
                break;
            }
            if (ret < 0) {
                av_frame_free(fr);
                throw new FFmpegException("avcodec_receive_frame", ret);
            }
            Frame f = Frame.wrap(fr, type, timeBase(), this);
            frames.add(f);
        }
        return frames;
    }

    void closeCodec() {
        if (codecCtx != null && !codecCtx.isNull()) {
            avcodec_free_context(codecCtx);
            codecCtx = null;
        }
        codecOpened = false;
    }

    public AVCodecContext codecContext() {
        return codecCtx;
    }

    @Override
    public String toString() {
        return "Stream#" + index + "(" + typeName() + ", codec=" + (codec != null ? codec.name() : "?") + ")";
    }
}
