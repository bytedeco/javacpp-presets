/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or (at your option) any later version (collectively, the "License");
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
package org.bytedeco.pytorch.utils.sentence.losses;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.cosine_similarity;
import static org.bytedeco.pytorch.global.torch.ones_like;

/** CosineSimilarityLoss: MSE(cosine(sim(anchor, positive)), target). */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class CosineSimilarityLoss {

    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    public static Tensor forward(Tensor anchor, Tensor positive, double targetSim) {
        if (anchor.dim() == 1) anchor = anchor.unsqueeze(0);
        if (positive.dim() == 1) positive = positive.unsqueeze(0);
        Tensor cs = cosine_similarity(anchor, positive, 1L, 1e-8);
        Tensor target = ones_like(cs).fill_(new Scalar(targetSim));
        return org.bytedeco.pytorch.global.torch.mse_loss(cs, target);
    }

    public static Tensor forward(Tensor anchor, Tensor positive) {
        return forward(anchor, positive, 1.0);
    }
}
