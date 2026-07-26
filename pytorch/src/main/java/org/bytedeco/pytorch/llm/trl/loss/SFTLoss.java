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
package org.bytedeco.pytorch.llm.trl.loss;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.cross_entropy;

/**
 * Supervised fine-tuning (causal LM) token cross-entropy loss.
 *
 * <p>Standard shift: predict token {@code t+1} from position {@code t}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SFTLoss {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private SFTLoss() {}

    /**
     * @param logits {@code [B, T, V]}
     * @param labels {@code [B, T]}
     * @return scalar mean CE over shifted tokens
     */
    public static Tensor compute(Tensor logits, Tensor labels) {
        // shift: logits[:, :-1, :] vs labels[:, 1:]
        long t = logits.size(1);
        Tensor shiftLogits = logits.slice(1, new LongOptional(0), new LongOptional(t - 1), 1);
        Tensor shiftLabels = labels.slice(1, new LongOptional(1), new LongOptional(labels.size(1)), 1);

        long b = shiftLogits.size(0);
        long tt = shiftLogits.size(1);
        long v = shiftLogits.size(2);
        Tensor flatLogits = shiftLogits.reshape(b * tt, v);
        Tensor flatLabels = shiftLabels.reshape(b * tt);
        return cross_entropy(flatLogits, flatLabels);
    }
}
