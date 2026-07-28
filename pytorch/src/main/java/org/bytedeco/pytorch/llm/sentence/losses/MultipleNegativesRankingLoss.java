/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.llm.sentence.losses;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;

import static org.bytedeco.pytorch.global.torch.matmul;
import static org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * In-batch negatives InfoNCE loss (MultipleNegativesRankingLoss).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class MultipleNegativesRankingLoss {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static Tensor forward(Tensor anchors, Tensor positives) {
        if (anchors.dim() == 1) anchors = anchors.unsqueeze(0);
        if (positives.dim() == 1) positives = positives.unsqueeze(0);
        // cosine sim matrix: [B, B]
        Tensor a = anchors.to(ScalarType.Float);
        Tensor p = positives.to(ScalarType.Float);
        Tensor sim = matmul(a, p.t());
        // labels: diagonal
        long B = a.size(0);
        Tensor labels = org.bytedeco.pytorch.global.torch.arange(new org.bytedeco.pytorch.Scalar(B),
                new org.bytedeco.pytorch.TensorOptions(org.bytedeco.pytorch.global.torch.ScalarType.Long));
        return org.bytedeco.pytorch.global.torch.cross_entropy_loss(sim, labels);
    }
}
