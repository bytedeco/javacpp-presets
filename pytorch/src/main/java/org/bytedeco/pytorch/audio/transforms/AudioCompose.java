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
package org.bytedeco.pytorch.audio.transforms;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Sequential composition of transforms (torchaudio / torchvision Compose-like).
 */
public final class AudioCompose implements AudioTransform<Object, Object> {
    private final List<AudioTransform<?, ?>> audioTransforms;

    @SafeVarargs
    public AudioCompose(AudioTransform<?, ?>... audioTransforms) {
        this.audioTransforms = Arrays.asList(Objects.requireNonNull(audioTransforms, "transforms"));
    }

    public AudioCompose(List<AudioTransform<?, ?>> audioTransforms) {
        this.audioTransforms = List.copyOf(Objects.requireNonNull(audioTransforms, "transforms"));
    }

    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public Object forward(Object input) {
        Object x = input;
        for (AudioTransform t : audioTransforms) {
            x = t.forward(x);
        }
        return x;
    }

    public List<AudioTransform<?, ?>> transforms() {
        return audioTransforms;
    }
}
