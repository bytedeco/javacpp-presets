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

import org.bytedeco.pytorch.llm.llamafactory.hparams.FactoryArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.TrainingArgs;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;
import org.bytedeco.pytorch.llm.trl.TrainerCallback;

import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/**
 * Thin wrapper around a TRL {@link BaseTrainer} that adds factory-level save /
 * stop hooks without reimplementing the train loop.
 */
public final class CustomTrainer implements AutoCloseable {

    private final BaseTrainer delegate;
    private final FactoryArgs args;
    private final CallbackHub hub;
    private final Consumer<Integer> saveHook;

    public CustomTrainer(BaseTrainer delegate, FactoryArgs args) {
        this(delegate, args, new CallbackHub(), null);
    }

    public CustomTrainer(
            BaseTrainer delegate,
            FactoryArgs args,
            CallbackHub hub,
            Consumer<Integer> saveHook) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
        this.args = Objects.requireNonNull(args, "args");
        this.hub = hub == null ? new CallbackHub() : hub;
        this.saveHook = saveHook;
        TrainingArgs t = args.training();
        int every = t == null ? 0 : Math.max(0, t.saveSteps());
        delegate.addCallback(hub);
        if (saveHook != null && every > 0) {
            delegate.addCallback(new TrainerCallback() {
                @Override
                public void onStepEnd(BaseTrainer trainer, int step, Map<String, Double> metrics) {
                    if (step > 0 && step % every == 0) {
                        saveHook.accept(step);
                    }
                }
            });
        }
    }

    public BaseTrainer delegate() { return delegate; }
    public FactoryArgs args() { return args; }
    public CallbackHub hub() { return hub; }
    public int globalStep() { return delegate.globalStep(); }

    public CustomTrainer addCallback(TrainerCallback cb) {
        hub.add(cb);
        return this;
    }

    public void train(BaseTrainer.BatchSupplier supplier) {
        delegate.train(supplier);
    }

    public double trainingStep(Map<String, org.bytedeco.pytorch.Tensor> batch) {
        return delegate.trainingStep(batch);
    }

    @Override
    public void close() {
        try {
            delegate.close();
        } catch (Exception ignored) {
        }
    }
}
