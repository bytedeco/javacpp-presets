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
package org.bytedeco.pytorch.vision.datasets;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.vision.transforms.VisionTransform;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Random;

/**
 * Synthetic vision dataset for tests/benchmarks (torchvision.datasets.FakeData-like).
 */
public final class FakeData extends VisionDataset {
    private final int size;
    private final int imageSize;
    private final int numClasses;
    private final int channels;
    private final Random random;
    private final long seed;

    public FakeData(int size, int imageSize, int numClasses) {
        this(size, imageSize, numClasses, 3, 0L);
    }

    public FakeData(int size, int imageSize, int numClasses, int channels, long seed) {
        super((String) null);
        this.size = size;
        this.imageSize = imageSize;
        this.numClasses = Math.max(1, numClasses);
        this.channels = channels <= 1 ? 1 : 3;
        this.seed = seed;
        this.random = new Random(seed);
    }

    public FakeData setTransform(VisionTransform<?, ?> transform) {
        super.setTransform(transform);
        return this;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public Sample get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index=" + index + " size=" + size);
        }
        Random rng = new Random(seed + index * 9973L);
        int type = channels == 1 ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB;
        BufferedImage img = new BufferedImage(imageSize, imageSize, type);
        Graphics2D g = img.createGraphics();
        g.setColor(new Color(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)));
        g.fillRect(0, 0, imageSize, imageSize);
        g.setColor(new Color(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)));
        g.fillOval(rng.nextInt(Math.max(1, imageSize / 2)), rng.nextInt(Math.max(1, imageSize / 2)),
                imageSize / 2, imageSize / 2);
        g.dispose();
        int label = rng.nextInt(numClasses);
        Object data = applyTransform(img);
        Object target = applyTargetTransform(label);
        return new Sample(data, target);
    }

    /** Convenience: return a random float CHW tensor batch without transforms. */
    public static Tensor randomBatch(int n, int c, int h, int w) {
        float[] data = new float[n * c * h * w];
        Random r = new Random(42);
        for (int i = 0; i < data.length; i++) {
            data[i] = r.nextFloat();
        }
        return torch.tensor(data).reshape(n, c, h, w);
    }
}
