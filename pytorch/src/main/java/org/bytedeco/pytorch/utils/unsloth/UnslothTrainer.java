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
package org.bytedeco.pytorch.utils.unsloth;
import org.bytedeco.pytorch.optim.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.trl.SFTTrainer;
import org.bytedeco.pytorch.llm.trl.config.SFTConfig;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Thin Unsloth training helper bridging {@link FastLanguageModel} and {@link SFTTrainer}.
 */
public final class UnslothTrainer implements AutoCloseable {

    private final FastLanguageModel fastModel;
    private final SFTTrainer sftTrainer;
    private final SFTConfig sftConfig;
    private long steps;

    public UnslothTrainer(FastLanguageModel fastModel, Optimizer optimizer, SFTConfig sftConfig) {
        this.fastModel = Objects.requireNonNull(fastModel, "fastModel");
        this.sftConfig = sftConfig == null
                ? SFTConfig.builder().maxSeqLength(fastModel.fastConfig().maxSeqLength()).build()
                : sftConfig;
        this.sftTrainer = new SFTTrainer(fastModel.model(), optimizer, this.sftConfig);
        fastModel.forTraining();
    }

    public static UnslothTrainer create(FastLanguageModel model, Optimizer optimizer) {
        return new UnslothTrainer(model, optimizer, null);
    }

    public FastLanguageModel fastModel() { return fastModel; }
    public SFTTrainer sftTrainer() { return sftTrainer; }
    public SFTConfig sftConfig() { return sftConfig; }
    public long steps() { return steps; }

    /** One SFT-style step using CausalLM loss on input ids. */
    public double trainStep(Tensor inputIds) {
        Tensor loss = fastModel.trainStep(inputIds);
        steps++;
        try {
            return loss.item_float();
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    public void train() {
        sftTrainer.train();
        fastModel.forTraining();
    }

    public void eval() {
        sftTrainer.eval();
        fastModel.model().eval();
    }

    public Map<String, Object> stats() {
        Map<String, Object> m = new LinkedHashMap<>(fastModel.stats());
        m.put("unsloth_trainer_steps", steps);
        m.put("sft_max_seq_length", sftConfig.maxSeqLength());
        m.put("sft_packing", sftConfig.packing());
        return m;
    }

    @Override
    public void close() {
        try { sftTrainer.close(); } catch (Exception ignored) {}
    }
}
