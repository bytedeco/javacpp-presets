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

import org.bytedeco.ffmpeg.avutil.AVFrame;

import static org.bytedeco.ffmpeg.global.avutil.AVMEDIA_TYPE_AUDIO;
import static org.bytedeco.ffmpeg.global.avutil.av_frame_alloc;
import static org.bytedeco.ffmpeg.global.avutil.av_frame_clone;
import static org.bytedeco.ffmpeg.global.avutil.av_frame_free;

/**
 * Base decoded frame — PyAV {@code av.frame.Frame}.
 *
 * <p>Subclasses: {@link VideoFrame}, {@link AudioFrame}.
 */
public abstract class Frame implements AutoCloseable {

    protected AVFrame frame;
    protected Rational timeBase;
    protected Stream stream;
    protected boolean closed;
    protected boolean ownsFrame = true;

    protected Frame(AVFrame frame, Rational timeBase) {
        this.frame = frame;
        this.timeBase = timeBase != null ? timeBase : new Rational(0, 1);
    }

    public static Frame wrap(AVFrame raw, int mediaType, Rational timeBase, Stream stream) {
        Frame f;
        if (mediaType == AVMEDIA_TYPE_AUDIO) {
            f = new AudioFrame(raw, timeBase);
        } else {
            f = new VideoFrame(raw, timeBase);
        }
        f.stream = stream;
        return f;
    }

    static AVFrame allocFrame() {
        FFmpegNative.load();
        AVFrame f = av_frame_alloc();
        if (f == null || f.isNull()) throw new FFmpegException("av_frame_alloc returned null");
        return f;
    }

    public long pts() {
        ensureOpen();
        return frame.pts();
    }

    public void pts(long pts) {
        ensureOpen();
        frame.pts(pts);
    }

    public Rational timeBase() {
        return timeBase;
    }

    public void timeBase(Rational tb) {
        this.timeBase = tb != null ? tb : new Rational(0, 1);
    }

    /** Presentation time in seconds (pts * time_base). NaN if no pts. */
    public double time() {
        long p = pts();
        if (p == 0x8000000000000000L) return Double.NaN;
        return timeBase.mul(p);
    }

    public Stream stream() {
        return stream;
    }

    void setStream(Stream s) {
        this.stream = s;
        if (s != null) this.timeBase = s.timeBase();
    }

    public AVFrame nativeFrame() {
        ensureOpen();
        return frame;
    }

    /** Clone the underlying AVFrame (new ownership). */
    protected AVFrame cloneNative() {
        ensureOpen();
        AVFrame c = av_frame_clone(frame);
        if (c == null || c.isNull()) throw new FFmpegException("av_frame_clone failed");
        return c;
    }

    protected void ensureOpen() {
        if (closed || frame == null || frame.isNull()) {
            throw new FFmpegException("Frame is closed");
        }
    }

    @Override
    public void close() {
        if (closed) return;
        closed = true;
        if (ownsFrame && frame != null && !frame.isNull()) {
            av_frame_free(frame);
        }
        frame = null;
    }

    @Override
    protected void finalize() throws Throwable {
        try { close(); } finally { super.finalize(); }
    }
}
