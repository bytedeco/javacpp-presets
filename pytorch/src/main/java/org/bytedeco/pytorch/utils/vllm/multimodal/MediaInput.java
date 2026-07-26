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
package org.bytedeco.pytorch.utils.vllm.multimodal;

import org.bytedeco.pytorch.Tensor;

import java.nio.file.Path;
import java.util.Objects;

/**
 * One piece of multimodal input (text, image, audio, video, or precomputed embedding).
 */
public final class MediaInput {

    public final MediaType type;
    public final String text;          // for TEXT
    public final Path path;            // for IMAGE/AUDIO/VIDEO file
    public final byte[] bytes;         // raw content (overrides path if set)
    public final Tensor tensor;        // for EMBEDDING or raw pixel/audio tensor
    public final int width;
    public final int height;
    public final int durationMs;

    private MediaInput(Builder b) {
        this.type = b.type;
        this.text = b.text;
        this.path = b.path;
        this.bytes = b.bytes;
        this.tensor = b.tensor;
        this.width = b.width;
        this.height = b.height;
        this.durationMs = b.durationMs;
    }

    public static MediaInput text(String text) {
        return builder().type(MediaType.TEXT).text(text).build();
    }

    public static MediaInput image(Path path) {
        return builder().type(MediaType.IMAGE).path(path).build();
    }

    public static MediaInput image(Path path, int width, int height) {
        return builder().type(MediaType.IMAGE).path(path).width(width).height(height).build();
    }

    public static MediaInput imageBytes(byte[] bytes, int width, int height) {
        return builder().type(MediaType.IMAGE).bytes(bytes).width(width).height(height).build();
    }

    public static MediaInput audio(Path path) {
        return builder().type(MediaType.AUDIO).path(path).build();
    }

    public static MediaInput audio(Path path, int durationMs) {
        return builder().type(MediaType.AUDIO).path(path).durationMs(durationMs).build();
    }

    public static MediaInput video(Path path) {
        return builder().type(MediaType.VIDEO).path(path).build();
    }

    public static MediaInput embedding(Tensor t) {
        return builder().type(MediaType.EMBEDDING).tensor(t).build();
    }

    public static Builder builder() { return new Builder(); }

    @Override
    public String toString() {
        return "MediaInput{type=" + type + ", text=" + text + ", path=" + path + "}";
    }

    public static final class Builder {
        private MediaType type = MediaType.TEXT;
        private String text = null;
        private Path path = null;
        private byte[] bytes = null;
        private Tensor tensor = null;
        private int width = 0;
        private int height = 0;
        private int durationMs = 0;

        public Builder type(MediaType v) { this.type = v; return this; }
        public Builder text(String v) { this.text = v; return this; }
        public Builder path(Path v) { this.path = v; return this; }
        public Builder bytes(byte[] v) { this.bytes = v; return this; }
        public Builder tensor(Tensor v) { this.tensor = v; return this; }
        public Builder width(int v) { this.width = v; return this; }
        public Builder height(int v) { this.height = v; return this; }
        public Builder durationMs(int v) { this.durationMs = v; return this; }

        public MediaInput build() {
            if (type == null) throw new IllegalArgumentException("type required");
            return new MediaInput(this);
        }
    }
}
