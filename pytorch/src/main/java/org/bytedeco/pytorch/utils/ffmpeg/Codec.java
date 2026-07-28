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
import org.bytedeco.ffmpeg.avcodec.AVCodecParameters;
import org.bytedeco.javacpp.BytePointer;

import static org.bytedeco.ffmpeg.global.avcodec.avcodec_find_decoder;
import static org.bytedeco.ffmpeg.global.avcodec.avcodec_find_decoder_by_name;
import static org.bytedeco.ffmpeg.global.avcodec.avcodec_find_encoder;
import static org.bytedeco.ffmpeg.global.avcodec.avcodec_find_encoder_by_name;
import static org.bytedeco.ffmpeg.global.avcodec.avcodec_get_name;

/**
 * Codec descriptor — PyAV {@code av.codec.Codec} / stream.codec.
 */
public final class Codec {

    private final String name;
    private final int id;
    private final boolean encoder;
    private final AVCodec nativeCodec; // may be null for params-only

    Codec(String name, int id, boolean encoder, AVCodec nativeCodec) {
        this.name = name != null ? name : "unknown";
        this.id = id;
        this.encoder = encoder;
        this.nativeCodec = nativeCodec;
    }

    public static Codec decoder(int codecId) {
        FFmpegNative.load();
        AVCodec c = avcodec_find_decoder(codecId);
        String name = nameOf(codecId, c);
        return new Codec(name, codecId, false, c);
    }

    public static Codec encoder(int codecId) {
        FFmpegNative.load();
        AVCodec c = avcodec_find_encoder(codecId);
        String name = nameOf(codecId, c);
        return new Codec(name, codecId, true, c);
    }

    public static Codec decoderByName(String name) {
        FFmpegNative.load();
        AVCodec c = avcodec_find_decoder_by_name(name);
        if (c == null || c.isNull()) {
            throw new FFmpegException("no decoder named: " + name);
        }
        return new Codec(name, c.id(), false, c);
    }

    public static Codec encoderByName(String name) {
        FFmpegNative.load();
        // common aliases
        String n = alias(name);
        AVCodec c = avcodec_find_encoder_by_name(n);
        if (c == null || c.isNull()) {
            // try bare name without lib prefix
            c = avcodec_find_encoder_by_name(name);
        }
        if (c == null || c.isNull()) {
            throw new FFmpegException("no encoder named: " + name);
        }
        return new Codec(FFmpegNative.ptrToString(c.name()) != null
                ? FFmpegNative.ptrToString(c.name()) : name, c.id(), true, c);
    }

    static Codec fromParameters(AVCodecParameters par, boolean wantEncoder) {
        if (par == null || par.isNull()) {
            return new Codec("unknown", 0, wantEncoder, null);
        }
        int id = par.codec_id();
        return wantEncoder ? encoder(id) : decoder(id);
    }

    private static String alias(String name) {
        if (name == null) return "libx264";
        switch (name.toLowerCase()) {
            case "h264": return "libx264";
            case "hevc":
            case "h265": return "libx265";
            case "mp3": return "libmp3lame";
            case "vorbis": return "libvorbis";
            case "opus": return "libopus";
            default: return name;
        }
    }

    private static String nameOf(int codecId, AVCodec c) {
        if (c != null && !c.isNull()) {
            String n = FFmpegNative.ptrToString(c.name());
            if (n != null) return n;
        }
        try {
            BytePointer p = avcodec_get_name(codecId);
            String n = FFmpegNative.ptrToString(p);
            if (n != null) return n;
        } catch (Throwable ignored) {}
        return "codec_" + codecId;
    }

    public String name() { return name; }
    public int id() { return id; }
    public boolean isEncoder() { return encoder; }
    public boolean isDecoder() { return !encoder; }

    /** Native AVCodec pointer; may be null. */
    public AVCodec nativeCodec() { return nativeCodec; }

    @Override
    public String toString() {
        return "Codec(" + name + ", id=" + id + ", " + (encoder ? "encoder" : "decoder") + ")";
    }
}
