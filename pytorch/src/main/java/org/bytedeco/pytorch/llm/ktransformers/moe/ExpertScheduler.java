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
package org.bytedeco.pytorch.llm.ktransformers.moe;

import org.bytedeco.pytorch.llm.ktransformers.config.KtMoEConfig;
import org.bytedeco.pytorch.llm.ktransformers.util.DeviceBudget;

import java.util.Objects;

/**
 * CPU–GPU Expert Scheduling control plane (upstream "CPU-GPU Expert Scheduling").
 *
 * <p>After each dispatch step (or on a cooldown), inspects
 * {@link ExpertLoadBalanceMetrics} and migrates hot experts to GPU / cold ones
 * to CPU according to {@link KtMoEConfig.SchedulePolicy}.
 */
public final class ExpertScheduler {

    private final ExpertPool pool;
    private final KtMoEConfig config;
    private final DeviceBudget budget;
    private int stepsSinceMigrate;

    public ExpertScheduler(ExpertPool pool, KtMoEConfig config, DeviceBudget budget) {
        this.pool = Objects.requireNonNull(pool, "pool");
        this.config = Objects.requireNonNull(config, "config");
        this.budget = budget;
        this.stepsSinceMigrate = 0;
    }

    public ExpertScheduler(ExpertPool pool, KtMoEConfig config) {
        this(pool, config, null);
    }

    public ExpertPool pool() { return pool; }
    public KtMoEConfig config() { return config; }

    /**
     * Record that experts in {@code topExpertIds} were selected this step and
     * optionally run migration.
     */
    public void onDispatch(int[] topExpertIds) {
        if (topExpertIds != null) {
            for (int id : topExpertIds) {
                if (id >= 0 && id < pool.numExperts()) {
                    pool.recordHit(id);
                }
            }
        }
        pool.metrics().recordDispatchStep();
        pool.advanceStep();
        stepsSinceMigrate++;
        maybeMigrate();
    }

    /** Force a migration pass regardless of cooldown. */
    public void forceMigrate() {
        stepsSinceMigrate = config.migrateCooldownSteps();
        maybeMigrate();
    }

    private void maybeMigrate() {
        if (config.schedule() == KtMoEConfig.SchedulePolicy.GPU_FIRST
                || config.schedule() == KtMoEConfig.SchedulePolicy.CPU_FIRST
                || config.schedule() == KtMoEConfig.SchedulePolicy.BALANCED) {
            // static policies: only enforce at construction; optional rebalance for BALANCED
            if (config.schedule() != KtMoEConfig.SchedulePolicy.BALANCED) {
                return;
            }
        }
        if (stepsSinceMigrate < config.migrateCooldownSteps()) {
            return;
        }
        stepsSinceMigrate = 0;
        if (config.schedule() == KtMoEConfig.SchedulePolicy.CPU_FIRST) {
            return;
        }
        if (budget != null && !budget.allowsGpuExpertPromote()) {
            return;
        }

        ExpertLoadBalanceMetrics m = pool.metrics();
        double[] freq = m.frequency();
        double thr = config.migrateHotThreshold();
        int hottest = m.hottestExpert();
        if (hottest >= 0 && freq[hottest] >= thr) {
            ExpertSpec e = pool.get(hottest);
            if (e.device() != ExpertDevice.GPU) {
                pool.promoteToGpu(hottest);
            }
        }
        // Demote cold GPU experts below half threshold when over-subscribed
        if (pool.gpuResidentCount() >= pool.gpuSlots() && pool.gpuSlots() > 0) {
            int cold = m.coldestOn(ExpertDevice.GPU, pool.all());
            if (cold >= 0 && freq[cold] < thr * 0.25) {
                pool.demoteToCpu(cold);
            }
        }
    }

    /**
     * Resolve compute device for an expert id (after optional migration).
     */
    public ExpertDevice deviceOf(int expertId) {
        return pool.get(expertId).device();
    }
}
