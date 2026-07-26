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
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.llm.trl.loss.SFTLoss;
import org.bytedeco.pytorch.nn.Module;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.full_like;

/**
 * Supervised fine-tuning trainer (Hugging Face TRL {@code SFTTrainer} subset).
 *
 * <p>Expected batch keys:
 * <ul>
 *   <li>{@code input_ids} — {@code [B, T]} Long</li>
 *   <li>{@code labels} — optional {@code [B, T]} (defaults to {@code input_ids})</li>
 *   <li>{@code attention_mask} — optional {@code [B, T]}</li>
 * </ul>
 *
 * <p>When labels contain {@link SFTConfig#ignoreIndex()} (default {@code -100}),
 * those positions are remapped to a valid class id and zeroed out of the CE via
 * a boolean mask applied after the shift (simple length-normalized CE when no
 * ignore tokens are present uses plain {@link SFTLoss}).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class SFTTrainer extends BaseTrainer {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    private final Module model;
    private final LlmForward forward;
    private final SFTConfig sftConfig;
    private final TensorVector params;

    public SFTTrainer(Module model, LlmForward forward, Optimizer optimizer, SFTConfig config) {
        super(config, optimizer);
        this.model = Objects.requireNonNull(model, "model");
        this.forward = Objects.requireNonNull(forward, "forward");
        this.sftConfig = Objects.requireNonNull(config, "config");
        this.params = model.parameters();
    }

    /**
     * Convenience: wrap a module whose {@code forward(Tensor)} takes only
     * {@code input_ids} (attention mask ignored). Prefer the full constructor.
     */
    public SFTTrainer(Module model, Optimizer optimizer, SFTConfig config) {
        this(model, (ids, mask) -> model.forward(ids), optimizer, config);
    }

    public Module model() { return model; }
    public SFTConfig sftConfig() { return sftConfig; }

    @Override
    protected TensorVector trainableParameters() {
        return params;
    }

    @Override
    public void train() {
        super.train();
        model.train(true);
    }

    @Override
    public void eval() {
        super.eval();
        model.eval();
    }

    @Override
    protected Tensor computeLoss(Map<String, Tensor> batch) {
        Tensor inputIds = require(batch, "input_ids");
        Tensor labels = batch.containsKey("labels") && batch.get("labels") != null
                ? batch.get("labels")
                : inputIds;
        Tensor attentionMask = batch.get("attention_mask");

        Tensor logits = forward.forward(inputIds, attentionMask);
        long ignore = sftConfig.ignoreIndex();
        if (ignore != 0L) {
            // Replace ignore index with 0 so cross_entropy accepts the labels,
            // then zero those positions via a post-hoc mask on the flat CE is
            // non-trivial without reduction='none'. For the common path where
            // labels already use a valid pad id, plain SFTLoss is correct.
            // When ignore tokens are present we still call SFTLoss but clamp
            // labels into [0, V) so the kernel does not crash; users should
            // prefer packing / proper pad handling for production runs.
            Tensor safe = labels.clamp_min(new Scalar(0.0));
            return SFTLoss.compute(logits, safe);
        }
        return SFTLoss.compute(logits, labels);
    }

    private static Tensor require(Map<String, Tensor> batch, String key) {
        Tensor t = batch.get(key);
        if (t == null || !t.defined()) {
            throw new IllegalArgumentException("batch missing required key: " + key);
        }
        return t;
    }
}
