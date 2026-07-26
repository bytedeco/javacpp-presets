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
package org.bytedeco.pytorch.llm.trl;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.LongOptional;
import org.bytedeco.pytorch.Tensor;

import static org.bytedeco.pytorch.global.torch.gather;
import static org.bytedeco.pytorch.global.torch.log_softmax;

/**
 * Helpers to extract per-sequence log-probabilities from causal-LM logits.
 *
 * <p>Given logits {@code [B, T, V]} and labels {@code [B, T]}, the standard shift
 * is applied (predict token t+1 from position t). An optional boolean/float mask
 * of shape {@code [B, T]} (aligned with labels) zeros out prompt tokens.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class LogProbUtils {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private LogProbUtils() {}

    /**
     * Per-sequence sum of token log-probs after the causal shift.
     *
     * @param logits {@code [B, T, V]}
     * @param labels {@code [B, T]} token ids (Long)
     * @param mask   optional {@code [B, T]} (1 = include); may be {@code null}
     * @return {@code [B]} summed log-probs
     */
    public static Tensor sequenceLogProbs(Tensor logits, Tensor labels, Tensor mask) {
        long t = logits.size(1);
        // shift
        Tensor shiftLogits = logits.slice(1, new LongOptional(0), new LongOptional(t - 1), 1);
        Tensor shiftLabels = labels.slice(1, new LongOptional(1), new LongOptional(labels.size(1)), 1);

        Tensor logProbs = log_softmax(shiftLogits, /*dim=*/-1); // [B, T-1, V]
        // gather along vocab dim
        Tensor idx = shiftLabels.unsqueeze(-1); // [B, T-1, 1]
        Tensor tokenLogp = gather(logProbs, /*dim=*/2, idx).squeeze(-1); // [B, T-1]

        if (mask != null && mask.defined()) {
            Tensor shiftMask = mask.slice(1, new LongOptional(1), new LongOptional(mask.size(1)), 1);
            tokenLogp = tokenLogp.mul(shiftMask);
        }
        return tokenLogp.sum(new long[]{1L}); // [B]
    }

    /** Sum log-probs with no mask. */
    public static Tensor sequenceLogProbs(Tensor logits, Tensor labels) {
        return sequenceLogProbs(logits, labels, null);
    }

    /**
     * Mean log-prob per non-masked token (length-normalized).
     */
    public static Tensor sequenceMeanLogProbs(Tensor logits, Tensor labels, Tensor mask) {
        long t = logits.size(1);
        Tensor shiftLogits = logits.slice(1, new LongOptional(0), new LongOptional(t - 1), 1);
        Tensor shiftLabels = labels.slice(1, new LongOptional(1), new LongOptional(labels.size(1)), 1);
        Tensor logProbs = log_softmax(shiftLogits, -1);
        Tensor idx = shiftLabels.unsqueeze(-1);
        Tensor tokenLogp = gather(logProbs, 2, idx).squeeze(-1);

        if (mask != null && mask.defined()) {
            Tensor shiftMask = mask.slice(1, new LongOptional(1), new LongOptional(mask.size(1)), 1);
            Tensor masked = tokenLogp.mul(shiftMask);
            Tensor denom = shiftMask.sum(new long[]{1L})
                    .clamp_min(new org.bytedeco.pytorch.Scalar(1.0));
            return masked.sum(new long[]{1L}).div(denom);
        }
        return tokenLogp.mean(new long[]{1L});
    }
}
