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
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.llm.trl.config.TrainerConfig;
import org.bytedeco.pytorch.optim.Optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.bytedeco.pytorch.global.torch.clip_grad_norm_;

/**
 * Shared training loop utilities for TRL-style LLM trainers (HF TRL-inspired).
 *
 * <p>Handles gradient accumulation, grad clipping, step counters, and callbacks.
 * Subclasses implement a single {@link #computeLoss} (and optionally generation).
 */
@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public abstract class BaseTrainer implements AutoCloseable {
    static { Loader.load(org.bytedeco.pytorch.presets.torch.class); }

    protected final TrainerConfig config;
    protected final Optimizer optimizer;
    protected final List<TrainerCallback> callbacks = new ArrayList<>();

    private int globalStep;
    private int microStep;
    private boolean training;
    private double runningLoss;
    private int runningLossCount;

    protected BaseTrainer(TrainerConfig config, Optimizer optimizer) {
        this.config = Objects.requireNonNull(config, "config");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
    }

    public TrainerConfig config() { return config; }
    public Optimizer optimizer() { return optimizer; }
    public int globalStep() { return globalStep; }
    public boolean isTraining() { return training; }

    public BaseTrainer addCallback(TrainerCallback cb) {
        if (cb != null) {
            callbacks.add(cb);
        }
        return this;
    }

    /**
     * Parameters used for gradient clipping. Subclasses that wrap PEFT / only
     * train adapters should override this.
     */
    protected abstract TensorVector trainableParameters();

    /**
     * Compute scalar loss for one micro-batch. Gradients should flow through
     * the returned tensor; do not call {@code backward} here.
     */
    protected abstract Tensor computeLoss(Map<String, Tensor> batch);

    /**
     * One optimizer micro-step with gradient accumulation.
     *
     * @return detached scalar loss value for this micro-batch
     */
    public double trainingStep(Map<String, Tensor> batch) {
        Objects.requireNonNull(batch, "batch");
        if (!training) {
            train();
        }

        int accum = Math.max(1, config.gradientAccumulationSteps());
        Tensor loss = computeLoss(batch);
        // Scale for accumulation so effective loss matches full batch
        Tensor scaled = accum > 1
                ? loss.div(new org.bytedeco.pytorch.Scalar((double) accum))
                : loss;
        scaled.backward();
        microStep++;

        double lossValue = loss.item_double();
        runningLoss += lossValue;
        runningLossCount++;

        if (microStep % accum == 0) {
            if (config.maxGradNorm() > 0.0) {
                TensorVector params = trainableParameters();
                if (params != null && params.size() > 0) {
                    clip_grad_norm_(params, config.maxGradNorm());
                }
            }
            optimizer.step();
            optimizer.zero_grad();
            globalStep++;

            Map<String, Double> metrics = new LinkedHashMap<>();
            metrics.put("loss", lossValue);
            metrics.put("lr", config.learningRate());
            fireStepEnd(metrics);

            if (config.loggingSteps() > 0 && globalStep % config.loggingSteps() == 0) {
                double avg = runningLossCount > 0 ? runningLoss / runningLossCount : lossValue;
                metrics.put("loss_avg", avg);
                fireLog(metrics);
                runningLoss = 0.0;
                runningLossCount = 0;
            }
        }
        return lossValue;
    }

    /**
     * Run up to {@code config.maxSteps()} optimizer steps, pulling batches from
     * the given supplier. Stops early if the supplier returns {@code null}.
     */
    public void train(BatchSupplier supplier) {
        Objects.requireNonNull(supplier, "supplier");
        train();
        fireTrainBegin();
        int target = Math.max(1, config.maxSteps());
        while (globalStep < target) {
            Map<String, Tensor> batch = supplier.next();
            if (batch == null) {
                break;
            }
            trainingStep(batch);
        }
        fireTrainEnd();
    }

    public void train() {
        training = true;
        optimizer.zero_grad();
    }

    public void eval() {
        training = false;
    }

    protected void fireTrainBegin() {
        for (TrainerCallback cb : callbacks) {
            cb.onTrainBegin(this);
        }
    }

    protected void fireTrainEnd() {
        for (TrainerCallback cb : callbacks) {
            cb.onTrainEnd(this);
        }
    }

    protected void fireStepEnd(Map<String, Double> metrics) {
        Map<String, Double> view = Collections.unmodifiableMap(metrics);
        for (TrainerCallback cb : callbacks) {
            cb.onStepEnd(this, globalStep, view);
        }
    }

    protected void fireLog(Map<String, Double> metrics) {
        Map<String, Double> view = Collections.unmodifiableMap(metrics);
        for (TrainerCallback cb : callbacks) {
            cb.onLog(this, globalStep, view);
        }
    }

    @Override
    public void close() {
        // Optimizer / model ownership stays with the caller.
    }

    /** Supplies preference / SFT / RLHF micro-batches as named tensors. */
    @FunctionalInterface
    public interface BatchSupplier {
        /**
         * @return next batch map, or {@code null} when the epoch/dataset is exhausted
         */
        Map<String, Tensor> next();
    }
}
