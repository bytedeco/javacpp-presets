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
package org.bytedeco.pytorch.vision.ffmpeg;

import org.bytedeco.ffmpeg.avcodec.AVCodec;
import org.bytedeco.ffmpeg.avcodec.AVCodecParameters;
import org.bytedeco.ffmpeg.avcodec.AVPacket;
import org.bytedeco.ffmpeg.avformat.AVFormatContext;
import org.bytedeco.ffmpeg.avformat.AVIOContext;
import org.bytedeco.ffmpeg.avformat.AVOutputFormat;
import org.bytedeco.ffmpeg.avformat.AVStream;
import org.bytedeco.ffmpeg.avutil.AVDictionary;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;

import static org.bytedeco.ffmpeg.global.avcodec.avcodec_find_encoder_by_name;
import static org.bytedeco.ffmpeg.global.avcodec.avcodec_parameters_copy;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_alloc;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_free;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_unref;
import static org.bytedeco.ffmpeg.global.avformat.*;
import static org.bytedeco.ffmpeg.global.avutil.AVMEDIA_TYPE_AUDIO;
import static org.bytedeco.ffmpeg.global.avutil.AVMEDIA_TYPE_VIDEO;

/**
 * Media container — PyAV {@code av.container.Container} / {@code av.open(...)}.
 *
 * <p>Thin AutoCloseable glue over {@link AVFormatContext}: open input/output,
 * demux/decode/mux/encode, seek. Does not reimplement FFmpeg — only ergonomics.
 *
 * <pre>{@code
 * try (Container c = Av.open("in.mp4")) {
 *     VideoStream v = c.streams().video(0);
 *     v.threadType("AUTO");
 *     for (Frame f : c.decode(v)) {
 *         VideoFrame vf = (VideoFrame) f;
 *         var rgb = vf.toNdarray("rgb24");
 *     }
 * }
 * try (Container out = Av.open("out.mp4", "w")) {
 *     VideoStream vs = out.addStream("libx264", 24);
 *     vs.width(640); vs.height(480); vs.pixFmt("yuv420p");
 *     out.writeHeader();
 *     // encode + mux ...
 * }
 * }</pre>
 */
public final class Container implements AutoCloseable {

    private final String name;
    private final boolean writable;
    private AVFormatContext fmtCtx;
    private StreamContainer streams;
    private final List<Stream> streamList = new ArrayList<>();
    private boolean headerWritten;
    private boolean trailerWritten;
    private boolean closed;
    private boolean ioOpened; // avio_open for output

    private Container(String name, AVFormatContext fmtCtx, boolean writable) {
        this.name = name;
        this.fmtCtx = fmtCtx;
        this.writable = writable;
        rebuildStreams();
    }

    // ── open ──────────────────────────────────────────────────────────────

    static Container openInput(String url, Map<String, String> options) {
        Objects.requireNonNull(url, "url");
        FFmpegNative.load();
        // Same pattern as VideoFile: alloc context, then open_input.
        AVFormatContext ctx = avformat_alloc_context();
        if (ctx == null || ctx.isNull()) throw new FFmpegException("avformat_alloc_context returned null");

        AVDictionary dict = null;
        try {
            if (options != null && !options.isEmpty()) {
                dict = Dictionary.of(options).toNative();
            }
            int ret = avformat_open_input(ctx, url, null, dict != null ? dict : (AVDictionary) null);
            if (ret < 0) {
                avformat_free_context(ctx);
                throw new FFmpegException("avformat_open_input(" + url + "): "
                        + FFmpegNative.errorString(ret), ret);
            }
            // open_input may consume/replace dict entries; free leftovers
            Dictionary.free(dict);
            dict = null;

            ret = avformat_find_stream_info(ctx, (AVDictionary) null);
            if (ret < 0) {
                avformat_close_input(ctx);
                throw new FFmpegException("avformat_find_stream_info: " + FFmpegNative.errorString(ret), ret);
            }
            return new Container(url, ctx, false);
        } catch (RuntimeException e) {
            Dictionary.free(dict);
            throw e;
        }
    }

    static Container openOutput(String url, String formatName, Map<String, String> options) {
        Objects.requireNonNull(url, "url");
        FFmpegNative.load();
        // @ByPtrPtr AVFormatContext holder
        AVFormatContext ctx = new AVFormatContext(null);
        int ret = avformat_alloc_output_context2(ctx, (AVOutputFormat) null, formatName, url);
        if (ret < 0 || ctx.isNull()) {
            throw new FFmpegException("avformat_alloc_output_context2(" + url + "): "
                    + FFmpegNative.errorString(ret), ret);
        }

        // open IO unless NOFILE
        boolean nofile = false;
        try {
            AVOutputFormat of = ctx.oformat();
            if (of != null && !of.isNull()) {
                nofile = (of.flags() & AVFMT_NOFILE) != 0;
            }
        } catch (Throwable ignored) {}

        if (!nofile) {
            AVIOContext pb = new AVIOContext(null);
            ret = avio_open(pb, url, AVIO_FLAG_WRITE);
            if (ret < 0) {
                avformat_free_context(ctx);
                throw new FFmpegException("avio_open(" + url + "): " + FFmpegNative.errorString(ret), ret);
            }
            ctx.pb(pb);
        }

        Container c = new Container(url, ctx, true);
        c.ioOpened = !nofile;
        if (options != null && !options.isEmpty()) {
            c.pendingOptions = Dictionary.of(options);
        }
        return c;
    }

    private Dictionary pendingOptions;

    // ── metadata ──────────────────────────────────────────────────────────

    public String name() { return name; }
    public boolean isWritable() { return writable; }
    public boolean isReadable() { return !writable; }

    public StreamContainer streams() {
        ensureOpen();
        return streams;
    }

    public AVFormatContext nativeContext() {
        ensureOpen();
        return fmtCtx;
    }

    public double duration() {
        ensureOpen();
        long d = fmtCtx.duration();
        if (d <= 0 || d == 0x8000000000000000L) return Double.NaN;
        // AV_TIME_BASE = 1000000
        return d / 1_000_000.0;
    }

    public long bitRate() {
        ensureOpen();
        return fmtCtx.bit_rate();
    }

    // ── output: add_stream ────────────────────────────────────────────────

    /**
     * PyAV {@code container.add_stream("libx264", rate=24)}.
     *
     * @param codecName encoder name (libx264, aac, …) or alias (h264)
     * @param rate      video fps or audio sample-rate hint; use 0 if unknown
     */
    /**
     * Resolve an encoder by name with practical aliases.
     * {@code libx264}/{@code h264} → try libx264, libx264rgb, openh264, h264_videotoolbox, h264, mpeg4.
     */
    static AVCodec resolveEncoder(String codecName) {
        if (codecName == null || codecName.isEmpty()) return null;
        String n = codecName.toLowerCase();
        List<String> candidates = new ArrayList<>();
        candidates.add(codecName);
        if (!candidates.contains(n)) candidates.add(n);
        if ("h264".equals(n) || "libx264".equals(n) || "avc".equals(n)) {
            candidates.addAll(Arrays.asList("libx264", "libx264rgb", "openh264", "h264_videotoolbox", "h264", "mpeg4"));
        } else if ("hevc".equals(n) || "h265".equals(n) || "libx265".equals(n)) {
            candidates.addAll(Arrays.asList("libx265", "hevc_videotoolbox", "hevc", "h265"));
        } else if ("mp3".equals(n)) {
            candidates.addAll(Arrays.asList("libmp3lame", "mp3"));
        } else if ("aac".equals(n)) {
            candidates.add("aac");
        }
        // dedupe while preserving order
        LinkedHashSet<String> seen = new LinkedHashSet<>(candidates);
        for (String c : seen) {
            AVCodec enc = avcodec_find_encoder_by_name(c);
            if (enc != null && !enc.isNull()) return enc;
        }
        // last resort: by codec id for h264
        if ("h264".equals(n) || "libx264".equals(n) || "avc".equals(n)) {
            try {
                AVCodec enc = org.bytedeco.ffmpeg.global.avcodec.avcodec_find_encoder(
                        org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_H264);
                if (enc != null && !enc.isNull()) return enc;
                enc = org.bytedeco.ffmpeg.global.avcodec.avcodec_find_encoder(
                        org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_ID_MPEG4);
                if (enc != null && !enc.isNull()) return enc;
            } catch (Throwable ignored) {}
        }
        return null;
    }

    public Stream addStream(String codecName, double rate) {
        ensureOpen();
        if (!writable) throw new FFmpegException("add_stream requires write mode");
        FFmpegNative.load();
        AVCodec enc = resolveEncoder(codecName);
        if (enc == null || enc.isNull()) {
            throw new FFmpegException("no encoder named: " + codecName
                    + " (tried aliases; javacpp-ffmpeg may ship openh264/mpeg4 instead of libx264)");
        }
        AVStream st = avformat_new_stream(fmtCtx, enc);
        if (st == null || st.isNull()) throw new FFmpegException("avformat_new_stream failed");
        st.id(fmtCtx.nb_streams() - 1);
        AVCodecParameters par = st.codecpar();
        par.codec_id(enc.id());
        par.codec_type(enc.type());

        // global header flag
        try {
            AVOutputFormat of = fmtCtx.oformat();
            if (of != null && (of.flags() & AVFMT_GLOBALHEADER) != 0) {
                // set later on codec context
            }
        } catch (Throwable ignored) {}

        rebuildStreams();
        Stream s = streamList.get(streamList.size() - 1);
        s.codec = new Codec(FFmpegNative.ptrToString(enc.name()), enc.id(), true, enc);
        if (s instanceof VideoStream && rate > 0) {
            ((VideoStream) s).rate(rate);
        } else if (s instanceof AudioStream && rate > 0) {
            ((AudioStream) s).sampleRate((int) rate);
            if (((AudioStream) s).channels() <= 0) ((AudioStream) s).channels(2);
        }
        // open encoder context lazily on first encode, but prepare context now
        s.ensureCodecContext(true);
        try {
            AVOutputFormat of = fmtCtx.oformat();
            if (of != null && (of.flags() & AVFMT_GLOBALHEADER) != 0 && s.codecCtx != null) {
                s.codecCtx.flags(s.codecCtx.flags() | org.bytedeco.ffmpeg.global.avcodec.AV_CODEC_FLAG_GLOBAL_HEADER);
            }
        } catch (Throwable ignored) {}
        return s;
    }

    /** PyAV {@code container.add_stream(template=other_stream)} — stream copy params. */
    public Stream addStream(Stream template) {
        ensureOpen();
        if (!writable) throw new FFmpegException("add_stream requires write mode");
        Objects.requireNonNull(template, "template");
        AVStream st = avformat_new_stream(fmtCtx, null);
        if (st == null || st.isNull()) throw new FFmpegException("avformat_new_stream failed");
        FFmpegNative.check(avcodec_parameters_copy(st.codecpar(), template.codecpar()), "avcodec_parameters_copy");
        st.codecpar().codec_tag(0);
        st.time_base(template.nativeStream().time_base());
        rebuildStreams();
        Stream s = streamList.get(streamList.size() - 1);
        s.codec = Codec.fromParameters(st.codecpar(), true);
        return s;
    }

    public VideoStream addVideoStream(String codecName, double fps) {
        Stream s = addStream(codecName, fps);
        if (!(s instanceof VideoStream)) throw new FFmpegException("codec is not video: " + codecName);
        return (VideoStream) s;
    }

    public AudioStream addAudioStream(String codecName, int sampleRate) {
        Stream s = addStream(codecName, sampleRate);
        if (!(s instanceof AudioStream)) throw new FFmpegException("codec is not audio: " + codecName);
        return (AudioStream) s;
    }

    /** PyAV: write container header (must be called before mux on output). */
    public void writeHeader() {
        ensureOpen();
        if (!writable) throw new FFmpegException("writeHeader on input container");
        if (headerWritten) return;
        // open any encoder contexts so extradata is set
        for (Stream s : streamList) {
            if (s.codec != null && s.codec.isEncoder()) {
                try {
                    s.openCodecIfNeeded(true);
                } catch (FFmpegException e) {
                    // stream-copy templates may not need encoder
                    if (s.codecCtx != null) {
                        // if context exists but open failed and we have codecpar already, continue
                    }
                }
            }
        }
        AVDictionary dict = pendingOptions != null ? pendingOptions.toNative() : null;
        // @ByPtrPtr AVDictionary — pass holder directly
        int ret = avformat_write_header(fmtCtx, dict != null ? dict : (AVDictionary) null);
        Dictionary.free(dict);
        pendingOptions = null;
        FFmpegNative.check(ret, "avformat_write_header");
        headerWritten = true;
    }

    // ── mux ───────────────────────────────────────────────────────────────

    /** PyAV {@code container.mux(packet)}. */
    public void mux(Packet packet) {
        ensureOpen();
        if (!writable) throw new FFmpegException("mux requires write mode");
        if (!headerWritten) writeHeader();
        if (packet == null) return;
        AVPacket pkt = packet.nativePacket();
        int ret = av_interleaved_write_frame(fmtCtx, pkt);
        // av_interleaved_write_frame takes ownership of packet data / unrefs
        // Our Packet still holds the struct — unref to mirror consumed state
        packet.unref();
        if (ret < 0) throw new FFmpegException("av_interleaved_write_frame", ret);
    }

    public void mux(List<Packet> packets) {
        if (packets == null) return;
        for (Packet p : packets) mux(p);
    }

    // ── demux ─────────────────────────────────────────────────────────────

    /**
     * PyAV {@code container.demux(streams...)} — iterate compressed packets.
     * Caller should close each {@link Packet}.
     */
    public Iterable<Packet> demux(Stream... filter) {
        ensureOpen();
        if (writable) throw new FFmpegException("demux requires read mode");
        final Set<Integer> want = new HashSet<>();
        if (filter != null) {
            for (Stream s : filter) if (s != null) want.add(s.index());
        }
        return () -> new Iterator<Packet>() {
            Packet next;
            boolean eof;

            private void advance() {
                if (eof) { next = null; return; }
                while (true) {
                    AVPacket raw = av_packet_alloc();
                    int ret = av_read_frame(fmtCtx, raw);
                    if (ret < 0) {
                        av_packet_free(raw);
                        eof = true;
                        next = null;
                        return;
                    }
                    int idx = raw.stream_index();
                    if (!want.isEmpty() && !want.contains(idx)) {
                        av_packet_unref(raw);
                        av_packet_free(raw);
                        continue;
                    }
                    Packet p = new Packet(raw);
                    if (idx >= 0 && idx < streamList.size()) p.setStream(streamList.get(idx));
                    next = p;
                    return;
                }
            }

            @Override
            public boolean hasNext() {
                if (next == null && !eof) advance();
                return next != null;
            }

            @Override
            public Packet next() {
                if (!hasNext()) throw new NoSuchElementException();
                Packet p = next;
                next = null;
                return p;
            }
        };
    }

    public Iterable<Packet> demux() {
        return demux(new Stream[0]);
    }

    // ── decode ────────────────────────────────────────────────────────────

    /**
     * PyAV {@code container.decode(stream)} / {@code decode(video=0)}.
     * Yields owned {@link Frame} instances (close when done, or rely on finalizer).
     */
    public Iterable<Frame> decode(Stream stream) {
        Objects.requireNonNull(stream, "stream");
        ensureOpen();
        if (writable) throw new FFmpegException("decode requires read mode");
        stream.ensureCodecContext(false);
        stream.openCodecIfNeeded(false);
        return () -> new Iterator<Frame>() {
            final Iterator<Packet> packets = demux(stream).iterator();
            final List<Frame> queue = new ArrayList<>();
            boolean flushed;

            private void fill() {
                while (queue.isEmpty()) {
                    if (packets.hasNext()) {
                        Packet pkt = packets.next();
                        try {
                            if (pkt.streamIndex() != stream.index()) continue;
                            queue.addAll(stream.decodePacket(pkt));
                        } finally {
                            pkt.close();
                        }
                    } else if (!flushed) {
                        flushed = true;
                        queue.addAll(stream.decodePacket(null));
                    } else {
                        return;
                    }
                }
            }

            @Override
            public boolean hasNext() {
                fill();
                return !queue.isEmpty();
            }

            @Override
            public Frame next() {
                if (!hasNext()) throw new NoSuchElementException();
                return queue.remove(0);
            }
        };
    }

    /** Decode video stream by ordinal among video streams (PyAV {@code decode(video=0)}). */
    public Iterable<Frame> decodeVideo(int videoOrdinal) {
        return decode(streams().video(videoOrdinal));
    }

    /** Decode audio stream by ordinal (PyAV {@code decode(audio=0)}). */
    public Iterable<Frame> decodeAudio(int audioOrdinal) {
        return decode(streams().audio(audioOrdinal));
    }

    // ── seek ──────────────────────────────────────────────────────────────

    /**
     * Seek to timestamp in stream time_base units.
     * PyAV {@code container.seek(ts, stream=...)}.
     */
    public void seek(long timestamp, Stream stream) {
        ensureOpen();
        int idx = stream != null ? stream.index() : -1;
        int ret = av_seek_frame(fmtCtx, idx, timestamp, AVSEEK_FLAG_BACKWARD);
        FFmpegNative.check(ret, "av_seek_frame");
        // flush codecs
        for (Stream s : streamList) {
            if (s.codecCtx != null && s.codecOpened) {
                org.bytedeco.ffmpeg.global.avcodec.avcodec_flush_buffers(s.codecCtx);
            }
        }
    }

    /** Seek to time in seconds on any / given stream. */
    public void seek(double seconds) {
        seek(seconds, null);
    }

    public void seek(double seconds, Stream stream) {
        ensureOpen();
        long ts;
        int idx = -1;
        if (stream != null) {
            idx = stream.index();
            Rational tb = stream.timeBase();
            ts = (long) (seconds * tb.den / Math.max(1, tb.num));
        } else {
            // AV_TIME_BASE
            ts = (long) (seconds * 1_000_000.0);
        }
        int ret = av_seek_frame(fmtCtx, idx, ts, AVSEEK_FLAG_BACKWARD);
        FFmpegNative.check(ret, "av_seek_frame");
        for (Stream s : streamList) {
            if (s.codecCtx != null && s.codecOpened) {
                org.bytedeco.ffmpeg.global.avcodec.avcodec_flush_buffers(s.codecCtx);
            }
        }
    }

    // ── close ─────────────────────────────────────────────────────────────

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            if (writable && headerWritten && !trailerWritten && fmtCtx != null) {
                av_write_trailer(fmtCtx);
                trailerWritten = true;
            }
        } catch (Throwable ignored) {}
        for (Stream s : streamList) {
            try { s.closeCodec(); } catch (Throwable ignored) {}
        }
        if (fmtCtx != null && !fmtCtx.isNull()) {
            if (writable) {
                if (ioOpened) {
                    try {
                        AVIOContext pb = fmtCtx.pb();
                        if (pb != null && !pb.isNull()) {
                            // @ByPtrPtr AVIOContext
                            avio_closep(pb);
                            fmtCtx.pb((AVIOContext) null);
                        }
                    } catch (Throwable ignored) {}
                }
                avformat_free_context(fmtCtx);
            } else {
                avformat_close_input(fmtCtx);
            }
            fmtCtx = null;
        }
    }

    private void rebuildStreams() {
        streamList.clear();
        if (fmtCtx == null || fmtCtx.isNull()) {
            streams = new StreamContainer(streamList);
            return;
        }
        int n = fmtCtx.nb_streams();
        for (int i = 0; i < n; i++) {
            AVStream st = fmtCtx.streams(i);
            int type = st.codecpar().codec_type();
            Stream s;
            if (type == AVMEDIA_TYPE_VIDEO) s = new VideoStream(this, st, i);
            else if (type == AVMEDIA_TYPE_AUDIO) s = new AudioStream(this, st, i);
            else s = new Stream(this, st, i);
            streamList.add(s);
        }
        streams = new StreamContainer(streamList);
    }

    private void ensureOpen() {
        if (closed || fmtCtx == null || fmtCtx.isNull()) {
            throw new FFmpegException("Container is closed");
        }
    }

    @Override
    public String toString() {
        return "Container(" + (writable ? "w" : "r") + ", " + name + ", streams="
                + (streams != null ? streams.size() : 0) + ")";
    }
}
