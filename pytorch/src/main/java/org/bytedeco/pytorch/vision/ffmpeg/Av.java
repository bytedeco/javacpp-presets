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

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Top-level PyAV-style facade — mirrors {@code import av}.
 *
 * <p>Thin glue over javacpp-ffmpeg. Prefer this for container open / frame helpers;
 * use existing {@link VideoFile}/{@link AudioFile} for simple tensor decode.
 *
 * <pre>{@code
 * try (Container c = Av.open("video.mp4")) {
 *     for (Frame f : c.decodeVideo(0)) {
 *         NDArray rgb = ((VideoFrame) f).toNdarray("rgb24");
 *     }
 * }
 * try (Container out = Av.open("out.mp4", "w")) {
 *     VideoStream vs = out.addVideoStream("libx264", 24);
 *     vs.width(640); vs.height(480); vs.pixFmt("yuv420p");
 *     // ...
 * }
 * }</pre>
 */
public final class Av {

    private Av() {}

    /** PyAV {@code av.open(path)} — read mode. */
    public static Container open(String path) {
        return open(path, "r", null);
    }

    public static Container open(Path path) {
        return open(path.toString());
    }

    /**
     * PyAV {@code av.open(path, mode='r'|'w')}.
     *
     * @param mode {@code "r"} / {@code "read"} or {@code "w"} / {@code "write"}
     */
    public static Container open(String path, String mode) {
        return open(path, mode, null);
    }

    /**
     * PyAV {@code av.open(path, mode=..., options={...})}.
     *
     * @param options format/protocol options (e.g. {@code rtsp_transport=tcp})
     */
    public static Container open(String path, String mode, Map<String, String> options) {
        Objects.requireNonNull(path, "path");
        FFmpegNative.load();
        String m = mode == null ? "r" : mode.toLowerCase();
        if ("r".equals(m) || "read".equals(m) || "rb".equals(m)) {
            return Container.openInput(path, options);
        }
        if ("w".equals(m) || "write".equals(m) || "wb".equals(m)) {
            return Container.openOutput(path, null, options);
        }
        throw new FFmpegException("unknown mode: " + mode + " (use 'r' or 'w')");
    }

    /** Open for write with explicit format name (e.g. {@code "mp4"}, {@code "flv"}). */
    public static Container open(String path, String mode, String format, Map<String, String> options) {
        Objects.requireNonNull(path, "path");
        FFmpegNative.load();
        String m = mode == null ? "w" : mode.toLowerCase();
        if (!("w".equals(m) || "write".equals(m) || "wb".equals(m))) {
            throw new FFmpegException("format override only valid for write mode");
        }
        return Container.openOutput(path, format, options);
    }

    /** Convenience options builder: {@code Av.options("rtsp_transport", "tcp")}. */
    public static Map<String, String> options(String... kv) {
        if (kv == null || kv.length == 0) return Collections.emptyMap();
        if (kv.length % 2 != 0) throw new IllegalArgumentException("options require key/value pairs");
        Map<String, String> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) {
            m.put(kv[i], kv[i + 1]);
        }
        return m;
    }

    /** Ensure natives loaded (also done lazily by {@link #open}). */
    public static void load() {
        FFmpegNative.load();
    }
}
