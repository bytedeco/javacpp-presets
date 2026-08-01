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
package org.bytedeco.pytorch.llm.unsloth.studio.model;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/** Live training progress event for SSE / Board / MCP. */
public final class TrainingProgressEvent {

    public enum Phase {
        QUEUED, PREPARING, TRAINING, EVALUATING, SAVING, COMPLETED, FAILED, CANCELLED
    }

    private final String runId;
    private final Phase phase;
    private final int step;
    private final int maxSteps;
    private final double loss;
    private final double learningRate;
    private final double epoch;
    private final double tokensPerSecond;
    private final double gpuMemoryUsedMb;
    private final double gpuMemoryTotalMb;
    private final String message;
    private final long timestampMs;
    private final Map<String, Double> metrics;

    private TrainingProgressEvent(Builder b) {
        this.runId = b.runId;
        this.phase = b.phase;
        this.step = b.step;
        this.maxSteps = b.maxSteps;
        this.loss = b.loss;
        this.learningRate = b.learningRate;
        this.epoch = b.epoch;
        this.tokensPerSecond = b.tokensPerSecond;
        this.gpuMemoryUsedMb = b.gpuMemoryUsedMb;
        this.gpuMemoryTotalMb = b.gpuMemoryTotalMb;
        this.message = b.message;
        this.timestampMs = b.timestampMs > 0 ? b.timestampMs : System.currentTimeMillis();
        this.metrics = Map.copyOf(b.metrics);
    }

    public static Builder builder() { return new Builder(); }

    public String runId() { return runId; }
    public Phase phase() { return phase; }
    public int step() { return step; }
    public int maxSteps() { return maxSteps; }
    public double loss() { return loss; }
    public double learningRate() { return learningRate; }
    public double epoch() { return epoch; }
    public double tokensPerSecond() { return tokensPerSecond; }
    public double gpuMemoryUsedMb() { return gpuMemoryUsedMb; }
    public double gpuMemoryTotalMb() { return gpuMemoryTotalMb; }
    public Optional<String> message() { return Optional.ofNullable(message); }
    public long timestampMs() { return timestampMs; }
    public Map<String, Double> metrics() { return metrics; }

    public double progress() {
        if (maxSteps <= 0) return 0;
        return Math.min(1.0, (double) step / (double) maxSteps);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("run_id", runId);
        m.put("phase", phase.name());
        m.put("step", step);
        m.put("max_steps", maxSteps);
        m.put("loss", loss);
        m.put("learning_rate", learningRate);
        m.put("epoch", epoch);
        m.put("tokens_per_second", tokensPerSecond);
        m.put("gpu_memory_used_mb", gpuMemoryUsedMb);
        m.put("gpu_memory_total_mb", gpuMemoryTotalMb);
        m.put("progress", progress());
        if (message != null) m.put("message", message);
        m.put("timestamp_ms", timestampMs);
        if (!metrics.isEmpty()) m.put("metrics", metrics);
        return m;
    }

    public static final class Builder {
        private String runId;
        private Phase phase = Phase.TRAINING;
        private int step;
        private int maxSteps;
        private double loss = Double.NaN;
        private double learningRate;
        private double epoch;
        private double tokensPerSecond;
        private double gpuMemoryUsedMb;
        private double gpuMemoryTotalMb;
        private String message;
        private long timestampMs;
        private Map<String, Double> metrics = Map.of();

        public Builder runId(String v) { this.runId = v; return this; }
        public Builder phase(Phase v) { this.phase = v; return this; }
        public Builder step(int v) { this.step = v; return this; }
        public Builder maxSteps(int v) { this.maxSteps = v; return this; }
        public Builder loss(double v) { this.loss = v; return this; }
        public Builder learningRate(double v) { this.learningRate = v; return this; }
        public Builder epoch(double v) { this.epoch = v; return this; }
        public Builder tokensPerSecond(double v) { this.tokensPerSecond = v; return this; }
        public Builder gpuMemoryUsedMb(double v) { this.gpuMemoryUsedMb = v; return this; }
        public Builder gpuMemoryTotalMb(double v) { this.gpuMemoryTotalMb = v; return this; }
        public Builder message(String v) { this.message = v; return this; }
        public Builder timestampMs(long v) { this.timestampMs = v; return this; }
        public Builder metrics(Map<String, Double> v) { this.metrics = v != null ? v : Map.of(); return this; }

        public TrainingProgressEvent build() {
            return new TrainingProgressEvent(this);
        }
    }
}
