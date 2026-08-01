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
package org.bytedeco.pytorch.llm.llamafactory.extras.monitor;

import org.bytedeco.pytorch.llm.llamafactory.train.CallbackHub;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.logging.Logger;

/**
 * Thin trainer callback used by factory workflows: keeps last metrics, optional
 * save hook on logging steps, and delegates to a {@link CallbackHub}.
 */
public final class TrainerMonitorCallback implements TrainerCallback {

    private static final Logger LOG = Logger.getLogger(TrainerMonitorCallback.class.getName());

    private final CallbackHub hub;
    private final Consumer<Integer> onSaveStep;
    private final int saveEvery;
    private final AtomicReference<Map<String, Double>> last =
            new AtomicReference<>(Map.of());

    public TrainerMonitorCallback(CallbackHub hub) {
        this(hub, null, 0);
    }

    public TrainerMonitorCallback(CallbackHub hub, Consumer<Integer> onSaveStep, int saveEvery) {
        this.hub = hub == null ? new CallbackHub() : hub;
        this.onSaveStep = onSaveStep;
        this.saveEvery = Math.max(0, saveEvery);
    }

    public static TrainerMonitorCallback of(MonitorBundle bundle) {
        Objects.requireNonNull(bundle, "bundle");
        CallbackHub hub = new CallbackHub().add(bundle.asCallback());
        return new TrainerMonitorCallback(hub);
    }

    public CallbackHub hub() {
        return hub;
    }

    public Map<String, Double> lastMetrics() {
        return last.get();
    }

    @Override
    public void onTrainBegin(BaseTrainer trainer) {
        hub.onTrainBegin(trainer);
        LOG.info("TrainerMonitor: train begin max_steps=" + trainer.config().maxSteps());
    }

    @Override
    public void onTrainEnd(BaseTrainer trainer) {
        hub.onTrainEnd(trainer);
        LOG.info("TrainerMonitor: train end global_step=" + trainer.globalStep());
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        if (metrics != null) {
            last.set(CollectionsCopy(metrics, step));
        }
        hub.onStepEnd(trainer, step, metrics);
        if (saveEvery > 0 && onSaveStep != null && step > 0 && step % saveEvery == 0) {
            try {
                onSaveStep.accept(step);
            } catch (RuntimeException e) {
                LOG.warning("checkpoint save hook failed at step " + step + ": " + e.getMessage());
            }
        }
    }

    @Override
    public void onLog(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        if (metrics != null) {
            last.set(CollectionsCopy(metrics, step));
        }
        hub.onLog(trainer, step, metrics);
    }

    private static Map<String, Double> CollectionsCopy(Map<String, Double> metrics, int step) {
        Map<String, Double> m = new LinkedHashMap<>(metrics);
        m.putIfAbsent("global_step", (double) step);
        return java.util.Collections.unmodifiableMap(m);
    }
}
