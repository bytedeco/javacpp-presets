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

import org.bytedeco.pytorch.llm.ktransformers.model.KtMiniMoECausalLM;
import org.bytedeco.pytorch.llm.ktransformers.monitor.KtTrainMonitor;
import org.bytedeco.pytorch.llm.ktransformers.moe.ExpertLoadBalanceMetrics;
import org.bytedeco.pytorch.nn.Module;

import java.util.Objects;
import java.util.function.BooleanSupplier;

/**
 * Step hooks: offload policy + expert metrics + board publish around train steps.
 */
public final class HeterogeneousTrainerHooks {

    private final FreezeAndOffloadPolicy offload;
    private final KtTrainMonitor monitor;
    private final BooleanSupplier stopCheck;

    public HeterogeneousTrainerHooks(FreezeAndOffloadPolicy offload,
                                     KtTrainMonitor monitor,
                                     BooleanSupplier stopCheck) {
        this.offload = offload != null ? offload : new FreezeAndOffloadPolicy(false, 0);
        this.monitor = Objects.requireNonNull(monitor, "monitor");
        this.stopCheck = stopCheck != null ? stopCheck : () -> false;
    }

    public boolean shouldStop() {
        return stopCheck.getAsBoolean() || monitor.board().stopRequested();
    }

    public void beforeTrain(Module model) {
        offload.applyModel(model);
        monitor.markRunning();
    }

    public void afterStep(Module model, int step, double loss, double lr, double gradNorm) {
        monitor.onTrainStep(step, loss, lr, gradNorm);
        if (model instanceof KtMiniMoECausalLM) {
            KtMiniMoECausalLM m = (KtMiniMoECausalLM) model;
            if (!m.layers.isEmpty()) {
                ExpertLoadBalanceMetrics metrics = m.layers.get(0).moe.metrics();
                monitor.onExperts(metrics);
            }
            monitor.publish(m.moeMetrics());
        }
    }

    public void afterTrain(boolean ok, String message) {
        if (ok) {
            monitor.markCompleted();
        } else {
            monitor.markFailed(message);
        }
    }

    public KtTrainMonitor monitor() { return monitor; }
    public FreezeAndOffloadPolicy offload() { return offload; }
}
