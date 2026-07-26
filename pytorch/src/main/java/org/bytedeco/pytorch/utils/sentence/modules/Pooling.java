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
package org.bytedeco.pytorch.utils.sentence.modules;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.utils.sentence.SentenceTransformer.PoolingStrategy;

import static org.bytedeco.pytorch.global.torch.ScalarType;
import static org.bytedeco.pytorch.global.torch.full;

/**
 * Sentence-Transformers Pooling module (MEAN / CLS / MAX).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class Pooling extends Module {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final PoolingStrategy strategy;

    public Pooling(PoolingStrategy strategy) {
        super("Pooling");
        this.strategy = strategy == null ? PoolingStrategy.MEAN : strategy;
    }

    public Pooling() {
        this(PoolingStrategy.MEAN);
    }

    public PoolingStrategy strategy() { return strategy; }

    public Tensor forward(Tensor hidden, Tensor mask) {
        long T = hidden.size(1);
        if (strategy == PoolingStrategy.CLS) {
            return hidden.slice(1, new LongOptional(0), new LongOptional(1), 1).squeeze(1);
        }
        if (strategy == PoolingStrategy.MAX) {
            if (mask == null) {
                return hidden.max(1L).get0();
            }
            Tensor m = mask.unsqueeze(-1);
            Tensor neg = full(new long[]{hidden.size(0), T, hidden.size(2)}, new Scalar(-1e9f));
            Tensor ones = full(new long[]{hidden.size(0), T, 1}, new Scalar(1.0f));
            Tensor masked = hidden.mul(m).add(neg.mul(ones.sub(m)));
            return masked.max(1L).get0();
        }
        // MEAN
        if (mask == null) {
            return hidden.mean(new long[]{1L});
        }
        Tensor m = mask.unsqueeze(-1).to(ScalarType.Float);
        Tensor summed = hidden.mul(m).sum(new long[]{1L});
        Tensor counts = m.sum(new long[]{1L}).clamp_min(new Scalar(1e-9));
        return summed.div(counts);
    }

    @Override
    public Tensor forward(Tensor hidden) {
        return forward(hidden, (Tensor) null);
    }
}
