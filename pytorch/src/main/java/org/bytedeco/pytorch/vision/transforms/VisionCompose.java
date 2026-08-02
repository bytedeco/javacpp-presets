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
package org.bytedeco.pytorch.vision.transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;

/**
 * Compose several transforms. Uses erased {@code Transform<Object,Object>} chaining so
 * image→tensor pipelines (e.g. Resize → ToTensor → Normalize) work like Python torchvision.
 */
@SuppressWarnings({"rawtypes", "unchecked"})
public final class VisionCompose implements VisionTransform<Object, Object> {
    private final List<VisionTransform> transforms;

    public VisionCompose(VisionTransform... transforms) {
        this(Arrays.asList(transforms));
    }

    public VisionCompose(List<? extends VisionTransform> transforms) {
        Objects.requireNonNull(transforms, "transforms");
        this.transforms = new ArrayList<>(transforms);
    }

    public static VisionCompose of(VisionTransform... transforms) {
        return new VisionCompose(transforms);
    }

    public List<VisionTransform> transforms() {
        return Collections.unmodifiableList(transforms);
    }

    @Override
    public Object forward(Object input) {
        Object x = input;
        for (VisionTransform t : transforms) {
            x = t.forward(x);
        }
        return x;
    }

    /** Randomly apply nested transforms with probability p. */
    public static final class RandomApply implements VisionTransform<Object, Object> {
        private final List<VisionTransform> transforms;
        private final double p;
        private final Random random;

        public RandomApply(List<? extends VisionTransform> transforms, double p) {
            this(transforms, p, new Random());
        }

        public RandomApply(List<? extends VisionTransform> transforms, double p, Random random) {
            this.transforms = new ArrayList<>(Objects.requireNonNull(transforms));
            this.p = p;
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            if (random.nextDouble() >= p) {
                return input;
            }
            Object x = input;
            for (VisionTransform t : transforms) {
                x = t.forward(x);
            }
            return x;
        }
    }

    /** Apply a single randomly selected transform. */
    public static final class RandomChoice implements VisionTransform<Object, Object> {
        private final List<VisionTransform> transforms;
        private final Random random;

        public RandomChoice(List<? extends VisionTransform> transforms) {
            this(transforms, new Random());
        }

        public RandomChoice(List<? extends VisionTransform> transforms, Random random) {
            this.transforms = new ArrayList<>(Objects.requireNonNull(transforms));
            if (this.transforms.isEmpty()) {
                throw new IllegalArgumentException("transforms empty");
            }
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            VisionTransform t = transforms.get(random.nextInt(transforms.size()));
            return t.forward(input);
        }
    }

    /** Apply transforms in a random order. */
    public static final class RandomOrder implements VisionTransform<Object, Object> {
        private final List<VisionTransform> transforms;
        private final Random random;

        public RandomOrder(List<? extends VisionTransform> transforms) {
            this(transforms, new Random());
        }

        public RandomOrder(List<? extends VisionTransform> transforms, Random random) {
            this.transforms = new ArrayList<>(Objects.requireNonNull(transforms));
            this.random = random != null ? random : new Random();
        }

        @Override
        public Object forward(Object input) {
            List<VisionTransform> order = new ArrayList<>(transforms);
            Collections.shuffle(order, random);
            Object x = input;
            for (VisionTransform t : order) {
                x = t.forward(x);
            }
            return x;
        }
    }
}
