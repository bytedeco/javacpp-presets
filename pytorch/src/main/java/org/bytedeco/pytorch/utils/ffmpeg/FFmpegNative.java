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

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.Loader;

import static org.bytedeco.ffmpeg.global.avutil.av_strerror;

/**
 * Package-private FFmpeg native helpers shared by the PyAV-parity layer and
 * the existing VideoFile / AudioFile readers.
 */
final class FFmpegNative {

    /** FFmpeg AVERROR(EAGAIN): -11 Linux, -35 macOS. */
    static final int AVERROR_EAGAIN =
            System.getProperty("os.name", "").toLowerCase(java.util.Locale.ROOT).contains("mac") ? -35 : -11;

    /** AVERROR_EOF typically -541478725 (MKTAG 'EOF '). */
    static final int AVERROR_EOF = -541478725;

    private static volatile boolean loaded = false;

    private FFmpegNative() {}

    /** Ensure avformat / avcodec / avutil / avfilter / swscale / swresample natives are loaded. */
    static void load() {
        if (loaded) return;
        synchronized (FFmpegNative.class) {
            if (loaded) return;
            try {
                Loader.load(org.bytedeco.ffmpeg.global.avutil.class);
                Loader.load(org.bytedeco.ffmpeg.global.avcodec.class);
                Loader.load(org.bytedeco.ffmpeg.global.avformat.class);
                try {
                    Loader.load(org.bytedeco.ffmpeg.global.avfilter.class);
                } catch (Throwable ignored) {
                    // filter optional for pure demux/decode paths
                }
                try {
                    Loader.load(org.bytedeco.ffmpeg.global.swscale.class);
                } catch (Throwable ignored) {}
                try {
                    Loader.load(org.bytedeco.ffmpeg.global.swresample.class);
                } catch (Throwable ignored) {}
                loaded = true;
            } catch (Throwable t) {
                throw new FFmpegException("Failed to load FFmpeg natives: " + t.getMessage(), t);
            }
        }
    }

    static void check(int ret, String what) {
        if (ret < 0) {
            throw new FFmpegException(what + ": " + errorString(ret), ret);
        }
    }

    static String errorString(int code) {
        try {
            byte[] buf = new byte[256];
            av_strerror(code, buf, buf.length);
            int end = 0;
            while (end < buf.length && buf[end] != 0) end++;
            String s = new String(buf, 0, end);
            if (!s.isEmpty()) return s + " (" + code + ")";
        } catch (Throwable ignored) {}
        return "FFmpeg error " + code;
    }

    static boolean isEagain(int ret) {
        return ret == AVERROR_EAGAIN || ret == -11 || ret == -35;
    }

    static boolean isEof(int ret) {
        return ret == AVERROR_EOF || ret == -541478725;
    }

    static String ptrToString(BytePointer p) {
        if (p == null || p.isNull()) return null;
        try {
            return p.getString();
        } catch (Throwable t) {
            return null;
        }
    }
}
