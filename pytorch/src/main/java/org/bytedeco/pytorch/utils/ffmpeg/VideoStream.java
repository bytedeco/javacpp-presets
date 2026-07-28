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

import org.bytedeco.ffmpeg.avformat.AVStream;
import org.bytedeco.ffmpeg.avutil.AVRational;
import org.bytedeco.javacpp.BytePointer;

import static org.bytedeco.ffmpeg.global.avutil.av_get_pix_fmt;
import static org.bytedeco.ffmpeg.global.avutil.av_get_pix_fmt_name;

/**
 * Video stream — PyAV {@code av.video.stream.VideoStream}.
 */
public final class VideoStream extends Stream {

    VideoStream(Container container, AVStream avStream, int index) {
        super(container, avStream, index);
    }

    public int width() {
        return codecpar().width();
    }

    public void width(int w) {
        codecpar().width(w);
        if (codecCtx != null) codecCtx.width(w);
    }

    public int height() {
        return codecpar().height();
    }

    public void height(int h) {
        codecpar().height(h);
        if (codecCtx != null) codecCtx.height(h);
    }

    /** Average / real frame rate as Rational (PyAV {@code stream.rate} / {@code average_rate}). */
    public Rational rate() {
        AVRational fr = avStream.avg_frame_rate();
        if (fr == null || fr.num() <= 0) fr = avStream.r_frame_rate();
        return Rational.of(fr);
    }

    public void rate(Rational r) {
        avStream.avg_frame_rate().num(r.num);
        avStream.avg_frame_rate().den(r.den);
        avStream.r_frame_rate().num(r.num);
        avStream.r_frame_rate().den(r.den);
        // encoder time_base often 1/fps
        if (container != null && container.isWritable()) {
            timeBase(new Rational(r.den, r.num));
        }
    }

    public void rate(double fps) {
        rate(Rational.fromDouble(fps, 1001));
    }

    public int pixFmt() {
        return codecpar().format();
    }

    public String pixFmtName() {
        BytePointer p = av_get_pix_fmt_name(pixFmt());
        String n = FFmpegNative.ptrToString(p);
        return n != null ? n : ("pix_" + pixFmt());
    }

    /** PyAV {@code stream.pix_fmt = "yuv420p"}. */
    public void pixFmt(String name) {
        int f = av_get_pix_fmt(name);
        if (f < 0) throw new FFmpegException("unknown pix_fmt: " + name);
        codecpar().format(f);
        if (codecCtx != null) codecCtx.pix_fmt(f);
    }

    public long bitRate() {
        return codecpar().bit_rate();
    }

    public void bitRate(long br) {
        codecpar().bit_rate(br);
        if (codecCtx != null) codecCtx.bit_rate(br);
    }
}
