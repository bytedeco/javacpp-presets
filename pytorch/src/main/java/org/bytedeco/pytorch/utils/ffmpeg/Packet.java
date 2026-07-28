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

import org.bytedeco.ffmpeg.avcodec.AVPacket;
import org.bytedeco.ffmpeg.avutil.AVRational;

import static org.bytedeco.ffmpeg.global.avcodec.av_packet_alloc;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_clone;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_free;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_ref;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_rescale_ts;
import static org.bytedeco.ffmpeg.global.avcodec.av_packet_unref;

/**
 * Compressed media packet — PyAV {@code av.packet.Packet}.
 *
 * <p>Owns an {@link AVPacket}. Prefer try-with-resources; demux iterators return
 * packets the caller should close or let GC finalize (finalizer is best-effort).
 */
public final class Packet implements AutoCloseable {

    private AVPacket pkt;
    private Stream stream; // optional association
    private boolean closed;

    Packet(AVPacket pkt) {
        this.pkt = pkt;
    }

    /** Allocate empty packet. */
    public static Packet allocate() {
        FFmpegNative.load();
        AVPacket p = av_packet_alloc();
        if (p == null || p.isNull()) throw new FFmpegException("av_packet_alloc returned null");
        return new Packet(p);
    }

    /** Clone another packet (deep ref). */
    public Packet clonePacket() {
        ensureOpen();
        AVPacket c = av_packet_clone(pkt);
        if (c == null || c.isNull()) throw new FFmpegException("av_packet_clone failed");
        Packet out = new Packet(c);
        out.stream = this.stream;
        return out;
    }

    void setStream(Stream s) {
        this.stream = s;
    }

    public Stream stream() {
        return stream;
    }

    public int streamIndex() {
        ensureOpen();
        return pkt.stream_index();
    }

    public void streamIndex(int idx) {
        ensureOpen();
        pkt.stream_index(idx);
    }

    public long pts() {
        ensureOpen();
        return pkt.pts();
    }

    public void pts(long pts) {
        ensureOpen();
        pkt.pts(pts);
    }

    public long dts() {
        ensureOpen();
        return pkt.dts();
    }

    public void dts(long dts) {
        ensureOpen();
        pkt.dts(dts);
    }

    public int size() {
        ensureOpen();
        return pkt.size();
    }

    public long duration() {
        ensureOpen();
        return pkt.duration();
    }

    public void duration(long d) {
        ensureOpen();
        pkt.duration(d);
    }

    public int flags() {
        ensureOpen();
        return pkt.flags();
    }

    /** Time in seconds using associated stream time_base, or NaN if unknown. */
    public double time() {
        if (stream == null) return Double.NaN;
        long p = pts();
        if (p == Long.MIN_VALUE /* AV_NOPTS_VALUE lower bits vary */) return Double.NaN;
        // AV_NOPTS_VALUE is 0x8000000000000000L
        if (p == 0x8000000000000000L) return Double.NaN;
        return stream.timeBase().mul(p);
    }

    public void rescaleTs(Rational src, Rational dst) {
        ensureOpen();
        av_packet_rescale_ts(pkt, src.toAV(), dst.toAV());
    }

    public void rescaleTs(AVRational src, AVRational dst) {
        ensureOpen();
        av_packet_rescale_ts(pkt, src, dst);
    }

    /** Native handle (do not free externally). */
    public AVPacket nativePacket() {
        ensureOpen();
        return pkt;
    }

    /** Transfer ownership of native packet out; this Packet becomes empty/closed. */
    AVPacket steal() {
        ensureOpen();
        AVPacket p = pkt;
        pkt = null;
        closed = true;
        return p;
    }

    void unref() {
        if (pkt != null && !pkt.isNull()) av_packet_unref(pkt);
    }

    private void ensureOpen() {
        if (closed || pkt == null || pkt.isNull()) {
            throw new FFmpegException("Packet is closed");
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (pkt != null && !pkt.isNull()) {
            av_packet_free(pkt);
            pkt = null;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try { close(); } finally { super.finalize(); }
    }

    @Override
    public String toString() {
        if (closed || pkt == null) return "Packet(closed)";
        return "Packet(stream=" + streamIndex() + ", pts=" + pts() + ", size=" + size() + ")";
    }
}
