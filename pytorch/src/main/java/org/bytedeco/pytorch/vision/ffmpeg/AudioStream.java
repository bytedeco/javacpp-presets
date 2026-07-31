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

import org.bytedeco.ffmpeg.avformat.AVStream;
import org.bytedeco.ffmpeg.avutil.AVChannelLayout;

import static org.bytedeco.ffmpeg.global.avutil.av_channel_layout_default;
import static org.bytedeco.ffmpeg.global.avutil.av_get_sample_fmt;
import static org.bytedeco.ffmpeg.global.avutil.av_get_sample_fmt_name;

/**
 * Audio stream — PyAV {@code av.audio.stream.AudioStream}.
 *
 * <p>FFmpeg 8 uses {@code AVChannelLayout} only (no deprecated {@code channels}/{@code channel_layout}).
 */
public final class AudioStream extends Stream {

    AudioStream(Container container, AVStream avStream, int index) {
        super(container, avStream, index);
    }

    public int sampleRate() {
        return codecpar().sample_rate();
    }

    public void sampleRate(int sr) {
        codecpar().sample_rate(sr);
        if (codecCtx != null) codecCtx.sample_rate(sr);
    }

    public int channels() {
        try {
            int ch = (int) codecpar().ch_layout().nb_channels();
            if (ch > 0) return ch;
        } catch (Throwable ignored) {}
        if (codecCtx != null) {
            try {
                int ch = (int) codecCtx.ch_layout().nb_channels();
                if (ch > 0) return ch;
            } catch (Throwable ignored) {}
        }
        return 0;
    }

    public void channels(int ch) {
        try {
            AVChannelLayout layout = codecpar().ch_layout();
            av_channel_layout_default(layout, ch);
        } catch (Throwable t) {
            try {
                codecpar().ch_layout().nb_channels(ch);
            } catch (Throwable ignored) {}
        }
        if (codecCtx != null) {
            try {
                AVChannelLayout layout = codecCtx.ch_layout();
                av_channel_layout_default(layout, ch);
            } catch (Throwable ignored) {}
        }
    }

    public int sampleFmt() {
        return codecpar().format();
    }

    public String sampleFmtName() {
        return FFmpegNative.ptrToString(av_get_sample_fmt_name(sampleFmt()));
    }

    public void sampleFmt(String name) {
        int f = av_get_sample_fmt(name);
        if (f < 0) throw new FFmpegException("unknown sample_fmt: " + name);
        codecpar().format(f);
        if (codecCtx != null) codecCtx.sample_fmt(f);
    }

    public long bitRate() {
        return codecpar().bit_rate();
    }

    public void bitRate(long br) {
        codecpar().bit_rate(br);
        if (codecCtx != null) codecCtx.bit_rate(br);
    }
}
