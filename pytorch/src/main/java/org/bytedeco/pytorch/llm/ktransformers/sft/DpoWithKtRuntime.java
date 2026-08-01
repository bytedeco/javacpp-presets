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
package org.bytedeco.pytorch.llm.ktransformers.sft;

import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Scalar;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.llm.ktransformers.config.KtConfig;
import org.bytedeco.pytorch.llm.ktransformers.model.KtMiniMoECausalLM;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtTrainMonitor;
import org.bytedeco.pytorch.optim.Adam;
import org.bytedeco.pytorch.optim.options.AdamOptions;

import java.util.Objects;
import java.util.Random;

/**
 * Minimal DPO-style preference step on KT mini models.
 *
 * <p>Upstream: "RL-DPO fine-tuning with LLaMA-Factory". Full TRL {@code DPOTrainer}
 * is preferred when reference + policy forwards are available; this class provides
 * a numerically testable proxy loss:
 * {@code L = -log σ(β * (logπ_θ(y_w) - logπ_θ(y_l)))} approximated via CE on
 * chosen vs rejected sequences (same prompt prefix).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DpoWithKtRuntime {

    static {
        Loader.load(org.bytedeco.pytorch.presets.torch.class);
    }

    private final KtConfig config;
    private final KtTrainMonitor monitor;
    private final double beta;

    public DpoWithKtRuntime(KtConfig config, KtTrainMonitor monitor, double beta) {
        this.config = Objects.requireNonNull(config, "config");
        this.monitor = Objects.requireNonNull(monitor, "monitor");
        this.beta = beta > 0 ? beta : 0.1;
    }

    public DpoWithKtRuntime(KtConfig config, KtTrainMonitor monitor) {
        this(config, monitor, 0.1);
    }

    /**
     * Run synthetic DPO steps: two completions (chosen/rejected), proxy loss.
     *
     * @return last loss
     */
    public double runSyntheticSteps(KtMiniMoECausalLM model, HeterogeneousTrainerHooks hooks,
                                    int steps, int seqLen, long seed) {
        Objects.requireNonNull(model, "model");
        Objects.requireNonNull(hooks, "hooks");
        int nSteps = Math.max(1, steps);
        int T = Math.max(6, seqLen);
        int V = model.vocabSize();
        hooks.beforeTrain(model);

        Adam opt = new Adam(model.parameters(),
                new AdamOptions(config.sft().learningRate() > 0 ? config.sft().learningRate() : 1e-3));
        Random rng = new Random(seed);
        double last = Double.NaN;
        try {
            for (int step = 1; step <= nSteps; step++) {
                if (hooks.shouldStop()) break;
                long[] chosen = new long[T];
                long[] rejected = new long[T];
                // shared prompt prefix
                int prefix = Math.max(2, T / 3);
                for (int i = 0; i < prefix; i++) {
                    long t = rng.nextInt(Math.max(1, V));
                    chosen[i] = t;
                    rejected[i] = t;
                }
                for (int i = prefix; i < T; i++) {
                    chosen[i] = rng.nextInt(Math.max(1, V));
                    rejected[i] = rng.nextInt(Math.max(1, V));
                }
                Tensor c = torch.tensor(chosen).unsqueeze(0);
                Tensor r = torch.tensor(rejected).unsqueeze(0);
                model.train(true);
                opt.zero_grad();
                Tensor lossC = model.loss(c);
                Tensor lossR = model.loss(r);
                // proxy: encourage lower CE on chosen than rejected
                // L = softplus(β * (lossC - lossR)) ≈ -logσ(β*(r-c)) when loss~-logπ
                Tensor diff = lossC.sub(lossR).mul(new Scalar(beta));
                Tensor loss = torch.softplus(diff);
                loss.backward();
                opt.step();
                last = loss.item_double();
                hooks.afterStep(model, step, last, config.sft().learningRate(), 0.0);
                monitor.publish(java.util.Map.of(
                        "kt/dpo/loss_chosen", lossC.item_double(),
                        "kt/dpo/loss_rejected", lossR.item_double(),
                        "kt/dpo/beta", beta));
                loss.close();
                lossC.close();
                lossR.close();
                diff.close();
                c.close();
                r.close();
            }
            hooks.afterTrain(true, "dpo-ok");
        } catch (RuntimeException ex) {
            hooks.afterTrain(false, ex.getMessage());
            throw ex;
        } finally {
            try { opt.close(); } catch (Throwable ignored) {}
        }
        return last;
    }
}
