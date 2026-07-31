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

import org.bytedeco.ffmpeg.avutil.AVDictionary;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.ffmpeg.global.avutil.av_dict_free;
import static org.bytedeco.ffmpeg.global.avutil.av_dict_set;

/**
 * Thin map bridge to FFmpeg {@code AVDictionary} — PyAV open/options kwargs.
 *
 * <p>Uses JavaCPP {@code @ByPtrPtr} idiom: pass an empty {@link AVDictionary}
 * and native code fills the pointer.
 */
public final class Dictionary {

    private final Map<String, String> map;

    public Dictionary() {
        this.map = new LinkedHashMap<>();
    }

    public Dictionary(Map<String, String> options) {
        this.map = new LinkedHashMap<>();
        if (options != null) {
            for (Map.Entry<String, String> e : options.entrySet()) {
                if (e.getKey() != null && e.getValue() != null) {
                    map.put(e.getKey(), e.getValue());
                }
            }
        }
    }

    public static Dictionary of(Map<String, String> options) {
        return new Dictionary(options);
    }

    public Dictionary put(String key, String value) {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        map.put(key, value);
        return this;
    }

    public Dictionary put(String key, Object value) {
        return put(key, String.valueOf(value));
    }

    public boolean isEmpty() {
        return map.isEmpty();
    }

    public Map<String, String> asMap() {
        return Collections.unmodifiableMap(map);
    }

    /**
     * Build a native AVDictionary. Caller owns the result and must
     * {@link #free(AVDictionary)} after the FFmpeg call (or when leftover).
     *
     * @return native dict, or {@code null} if empty
     */
    public AVDictionary toNative() {
        if (map.isEmpty()) return null;
        // @ByPtrPtr idiom: empty holder, av_dict_set allocates into it
        AVDictionary dict = new AVDictionary(null);
        for (Map.Entry<String, String> e : map.entrySet()) {
            int ret = av_dict_set(dict, e.getKey(), e.getValue(), 0);
            if (ret < 0) {
                free(dict);
                throw new FFmpegException("av_dict_set(" + e.getKey() + ")", ret);
            }
        }
        return dict;
    }

    public static void free(AVDictionary dict) {
        if (dict == null || dict.isNull()) return;
        av_dict_free(dict);
    }
}
