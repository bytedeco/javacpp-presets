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
package org.bytedeco.pytorch.llm.llamafactory.model;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.nn.modules.LinearImpl;

import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.ScalarType;

/**
 * Scalar value / reward head on top of last-token hidden states
 * (TRL AutoModelForCausalLMWithValueHead subset).
 *
 * <p>Used by RM and PPO critic. Expects hidden states {@code [B, T, H]} or
 * already-pooled {@code [B, H]}; returns scores {@code [B]}.
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class ValueHead extends Module {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final LinearImpl score;
    private final int hiddenSize;
    private final boolean dropoutEnabled;
    private final double dropoutP;

    public ValueHead(int hiddenSize) {
        this(hiddenSize, 0.0);
    }

    public ValueHead(int hiddenSize, double dropoutP) {
        this.hiddenSize = hiddenSize;
        this.dropoutP = Math.max(0.0, Math.min(1.0, dropoutP));
        this.dropoutEnabled = this.dropoutP > 0.0;
        this.score = new LinearImpl(hiddenSize, 1);
        register_module("score", this.score);
    }

    public int hiddenSize() { return hiddenSize; }
    public LinearImpl score() { return score; }

    /**
     * @param hidden {@code [B,T,H]} or {@code [B,H]}
     * @param attentionMask optional {@code [B,T]} — last non-pad hidden is pooled
     */
    public Tensor forward(Tensor hidden, Tensor attentionMask) {
        return apply(hidden, attentionMask);
    }

    @Override
    public Tensor forward(Tensor hidden) {
        return apply(hidden, null);
    }

    private Tensor apply(Tensor hidden, Tensor attentionMask) {
        Objects.requireNonNull(hidden, "hidden");
        Tensor h = hidden;
        if (h.dim() == 3) {
            // pool last non-pad token
            if (attentionMask != null && attentionMask.dim() == 2) {
                // lengths = mask.sum(-1) - 1
                Tensor lengths = attentionMask.to(ScalarType.Long)
                        .sum(new long[]{1})
                        .sub(new Scalar(1L));
                long b = h.size(0);
                // gather: h[i, lengths[i], :]
                java.util.List<Tensor> rows = new java.util.ArrayList<>((int) b);
                for (long i = 0; i < b; i++) {
                    long t = Math.max(0L, lengths.select(0, i).item_long());
                    t = Math.min(t, h.size(1) - 1);
                    rows.add(h.select(0, i).select(0, t).unsqueeze(0));
                }
                h = org.bytedeco.pytorch.global.torch.cat(
                        new org.bytedeco.pytorch.TensorVector(rows.toArray(new Tensor[0])), 0);
            } else {
                // last time step
                h = h.select(1, h.size(1) - 1);
            }
        }
        if (dropoutEnabled && is_training()) {
            h = org.bytedeco.pytorch.global.torch.dropout(h, dropoutP, true);
        }
        Tensor s = score.forward(h); // [B, 1]
        return s.squeeze(-1);       // [B]
    }
}
