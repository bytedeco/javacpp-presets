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
package org.bytedeco.pytorch.utils.audio.transforms;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Sequential composition of transforms (torchaudio / torchvision Compose-like).
 */
public final class Compose implements Transform<Object, Object> {
    private final List<Transform<?, ?>> transforms;

    @SafeVarargs
    public Compose(Transform<?, ?>... transforms) {
        this.transforms = Arrays.asList(Objects.requireNonNull(transforms, "transforms"));
    }

    public Compose(List<Transform<?, ?>> transforms) {
        this.transforms = List.copyOf(Objects.requireNonNull(transforms, "transforms"));
    }

    @Override
    @SuppressWarnings({"rawtypes", "unchecked"})
    public Object forward(Object input) {
        Object x = input;
        for (Transform t : transforms) {
            x = t.forward(x);
        }
        return x;
    }

    public List<Transform<?, ?>> transforms() {
        return transforms;
    }
}
