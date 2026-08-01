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
package org.bytedeco.pytorch.llm.llamafactory.train;

import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Fans trainer lifecycle events out to multiple {@link TrainerCallback}s
 * (board / TensorBoard / WandB / SwanLab / user hooks).
 */
public final class CallbackHub implements TrainerCallback {

    private final List<TrainerCallback> delegates = new CopyOnWriteArrayList<>();

    public CallbackHub() {}

    public CallbackHub(List<? extends TrainerCallback> initial) {
        if (initial != null) {
            for (TrainerCallback c : initial) {
                add(c);
            }
        }
    }

    public CallbackHub add(TrainerCallback cb) {
        if (cb != null && cb != this) {
            delegates.add(cb);
        }
        return this;
    }

    public CallbackHub addAll(Iterable<? extends TrainerCallback> cbs) {
        if (cbs != null) {
            for (TrainerCallback c : cbs) {
                add(c);
            }
        }
        return this;
    }

    public boolean remove(TrainerCallback cb) {
        return delegates.remove(cb);
    }

    public List<TrainerCallback> delegates() {
        return Collections.unmodifiableList(new ArrayList<>(delegates));
    }

    public int size() {
        return delegates.size();
    }

    /** Attach this hub to a trainer (idempotent if already added). */
    public void install(BaseTrainer trainer) {
        Objects.requireNonNull(trainer, "trainer").addCallback(this);
    }

    @Override
    public void onTrainBegin(BaseTrainer trainer) {
        for (TrainerCallback c : delegates) {
            try {
                c.onTrainBegin(trainer);
            } catch (RuntimeException e) {
                // never abort train on monitor failure
            }
        }
    }

    @Override
    public void onTrainEnd(BaseTrainer trainer) {
        for (TrainerCallback c : delegates) {
            try {
                c.onTrainEnd(trainer);
            } catch (RuntimeException ignored) {
            }
        }
    }

    @Override
    public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        for (TrainerCallback c : delegates) {
            try {
                c.onStepEnd(trainer, step, metrics);
            } catch (RuntimeException ignored) {
            }
        }
    }

    @Override
    public void onLog(BaseTrainer trainer, int step, Map<String, Double> metrics) {
        for (TrainerCallback c : delegates) {
            try {
                c.onLog(trainer, step, metrics);
            } catch (RuntimeException ignored) {
            }
        }
    }
}
