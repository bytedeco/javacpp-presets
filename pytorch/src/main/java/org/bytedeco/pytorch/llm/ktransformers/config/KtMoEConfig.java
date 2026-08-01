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
package org.bytedeco.pytorch.llm.ktransformers.config;

/**
 * Mixture-of-Experts configuration for CPU–GPU expert scheduling.
 *
 * <p>Aligns with upstream kt-kernel MoE knobs: expert count, top-k, shared expert
 * (DeepSeek-style), and scheduling policy across device tiers.
 */
public final class KtMoEConfig {

    /**
     * Expert compute placement policy.
     * <ul>
     *   <li>{@link #GPU_FIRST} — prefer GPU-resident experts; overflow to CPU</li>
     *   <li>{@link #CPU_FIRST} — keep experts on CPU; promote hot ones</li>
     *   <li>{@link #BALANCED} — split by id modulo / load histogram</li>
     *   <li>{@link #AUTO} — migrate by observed selection frequency + budget</li>
     * </ul>
     */
    public enum SchedulePolicy {
        GPU_FIRST,
        CPU_FIRST,
        BALANCED,
        AUTO
    }

    private final int numExperts;
    private final int topK;
    private final boolean sharedExpert;
    private final boolean normTopkProb;
    private final SchedulePolicy schedule;
    private final int gpuExpertSlots;
    private final boolean numaAware;
    private final int numaNodes;
    private final double migrateHotThreshold;
    private final int migrateCooldownSteps;

    private KtMoEConfig(Builder b) {
        if (b.numExperts < 1) {
            throw new IllegalArgumentException("numExperts must be >= 1");
        }
        if (b.topK < 1 || b.topK > b.numExperts) {
            throw new IllegalArgumentException("topK must be in [1, numExperts]");
        }
        if (b.gpuExpertSlots < 0) {
            throw new IllegalArgumentException("gpuExpertSlots must be >= 0");
        }
        if (b.numaNodes < 1) {
            throw new IllegalArgumentException("numaNodes must be >= 1");
        }
        this.numExperts = b.numExperts;
        this.topK = b.topK;
        this.sharedExpert = b.sharedExpert;
        this.normTopkProb = b.normTopkProb;
        this.schedule = b.schedule;
        this.gpuExpertSlots = b.gpuExpertSlots == 0 ? Math.min(b.numExperts, b.topK * 4) : b.gpuExpertSlots;
        this.numaAware = b.numaAware;
        this.numaNodes = b.numaNodes;
        this.migrateHotThreshold = b.migrateHotThreshold;
        this.migrateCooldownSteps = b.migrateCooldownSteps;
    }

    public int numExperts() { return numExperts; }
    public int topK() { return topK; }
    public boolean sharedExpert() { return sharedExpert; }
    public boolean normTopkProb() { return normTopkProb; }
    public SchedulePolicy schedule() { return schedule; }
    public int gpuExpertSlots() { return gpuExpertSlots; }
    public boolean numaAware() { return numaAware; }
    public int numaNodes() { return numaNodes; }
    public double migrateHotThreshold() { return migrateHotThreshold; }
    public int migrateCooldownSteps() { return migrateCooldownSteps; }

    public static Builder builder() { return new Builder(); }

    /** Mixtral-like defaults: 8 experts, top-2. */
    public static KtMoEConfig mixtral() {
        return builder().numExperts(8).topK(2).sharedExpert(false).build();
    }

    /** DeepSeek-like: many experts + shared + top-k. */
    public static KtMoEConfig deepseek(int numExperts, int topK) {
        return builder().numExperts(numExperts).topK(topK).sharedExpert(true).normTopkProb(true)
                .schedule(SchedulePolicy.AUTO).build();
    }

    public static final class Builder {
        private int numExperts = 8;
        private int topK = 2;
        private boolean sharedExpert = false;
        private boolean normTopkProb = true;
        private SchedulePolicy schedule = SchedulePolicy.AUTO;
        private int gpuExpertSlots = 0;
        private boolean numaAware = true;
        private int numaNodes = 1;
        private double migrateHotThreshold = 0.15;
        private int migrateCooldownSteps = 32;

        public Builder numExperts(int v) { this.numExperts = v; return this; }
        public Builder topK(int v) { this.topK = v; return this; }
        public Builder sharedExpert(boolean v) { this.sharedExpert = v; return this; }
        public Builder normTopkProb(boolean v) { this.normTopkProb = v; return this; }
        public Builder schedule(SchedulePolicy v) { this.schedule = v; return this; }
        public Builder gpuExpertSlots(int v) { this.gpuExpertSlots = v; return this; }
        public Builder numaAware(boolean v) { this.numaAware = v; return this; }
        public Builder numaNodes(int v) { this.numaNodes = v; return this; }
        public Builder migrateHotThreshold(double v) { this.migrateHotThreshold = v; return this; }
        public Builder migrateCooldownSteps(int v) { this.migrateCooldownSteps = v; return this; }

        public KtMoEConfig build() { return new KtMoEConfig(this); }
    }
}
