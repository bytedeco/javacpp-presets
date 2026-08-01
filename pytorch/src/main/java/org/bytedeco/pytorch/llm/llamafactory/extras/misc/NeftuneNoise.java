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
package org.bytedeco.pytorch.llm.llamafactory.extras.misc;

import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.randn_like;

/**
 * NEFTune embedding noise (https://arxiv.org/abs/2310.05914).
 *
 * <p>{@code noisy = embed + (alpha / sqrt(dims)) * ε}, applied only during training.
 * {@code alpha <= 0} disables the transform (identity).
 */
public final class NeftuneNoise {

    private final double alpha;

    public NeftuneNoise(double alpha) {
        this.alpha = alpha;
    }

    public double alpha() {
        return alpha;
    }

    public boolean enabled() {
        return alpha > 0.0;
    }

    /**
     * @param embeddings token embeddings {@code [B, T, H]} (or any rank ≥ 1)
     * @param training   when false, returns embeddings unchanged
     */
    public Tensor apply(Tensor embeddings, boolean training) {
        Objects.requireNonNull(embeddings, "embeddings");
        if (!enabled() || !training) {
            return embeddings;
        }
        long dims = 1L;
        for (long d : embeddings.shape()) {
            dims *= Math.max(1L, d);
        }
        // per NEFTune: scale by alpha / sqrt(hidden); use last dim when 3D
        long hidden = embeddings.dim() >= 1 ? Math.max(1L, embeddings.size(embeddings.dim() - 1)) : dims;
        double scale = alpha / Math.sqrt((double) hidden);
        Tensor noise = randn_like(embeddings).mul(new Scalar(scale));
        return embeddings.add(noise);
    }

    public static Tensor maybeAdd(Tensor embeddings, double alpha, boolean training) {
        return new NeftuneNoise(alpha).apply(embeddings, training);
    }
}
