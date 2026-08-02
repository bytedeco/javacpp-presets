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
package org.bytedeco.pytorch.audio.datasets;

import org.bytedeco.pytorch.audio.transforms.AudioTransform;

import java.nio.file.Path;
import java.util.Iterator;
import java.util.Objects;

/**
 * Base torchaudio-style dataset: {@code (waveform, target)} samples with optional transforms.
 */
public abstract class AudioDataset implements Iterable<AudioDataset.Sample> {
    protected final Path root;
    protected AudioTransform<?, ?> audioTransform;
    protected AudioTransform<?, ?> targetAudioTransform;

    protected AudioDataset(Path root) {
        this.root = root;
    }

    protected AudioDataset(String root) {
        this.root = root == null ? null : Path.of(root);
    }

    public AudioDataset setTransform(AudioTransform<?, ?> audioTransform) {
        this.audioTransform = audioTransform;
        return this;
    }

    public AudioDataset set_transform(AudioTransform<?, ?> audioTransform) {
        return setTransform(audioTransform);
    }

    public AudioDataset setTargetTransform(AudioTransform<?, ?> targetAudioTransform) {
        this.targetAudioTransform = targetAudioTransform;
        return this;
    }

    public abstract int size();

    public abstract Sample get(int index);

    public int length() {
        return size();
    }

    public Path root() {
        return root;
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    protected Object applyTransform(Object input) {
        if (audioTransform == null) {
            return input;
        }
        return ((AudioTransform) audioTransform).forward(input);
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    protected Object applyTargetTransform(Object target) {
        if (targetAudioTransform == null) {
            return target;
        }
        return ((AudioTransform) targetAudioTransform).forward(target);
    }

    @Override
    public Iterator<Sample> iterator() {
        return new Iterator<Sample>() {
            int i = 0;

            @Override
            public boolean hasNext() {
                return i < size();
            }

            @Override
            public Sample next() {
                return get(i++);
            }
        };
    }

    /** One dataset example. */
    public static final class Sample {
        public final Object data;
        public final Object target;

        public Sample(Object data, Object target) {
            this.data = data;
            this.target = target;
        }

        public Object data() {
            return data;
        }

        public Object target() {
            return target;
        }

        @Override
        public String toString() {
            return "Sample{data=" + Objects.toString(data) + ", target=" + Objects.toString(target) + "}";
        }
    }
}
