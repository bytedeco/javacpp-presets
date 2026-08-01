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

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/** Persistent metadata for one training run. */
public final class TrainingRunRecord {

    public enum Status { QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED }

    private final String runId;
    private final String projectName;
    private final TrainingStartRequest request;
    private final Status status;
    private final Path outputDir;
    private final int globalStep;
    private final double lastLoss;
    private final String error;
    private final long createdAtMs;
    private final long updatedAtMs;
    private final long finishedAtMs;
    private final Map<String, Double> lastMetrics;

    private TrainingRunRecord(Builder b) {
        this.runId = b.runId;
        this.projectName = b.projectName;
        this.request = b.request;
        this.status = b.status;
        this.outputDir = b.outputDir;
        this.globalStep = b.globalStep;
        this.lastLoss = b.lastLoss;
        this.error = b.error;
        this.createdAtMs = b.createdAtMs;
        this.updatedAtMs = b.updatedAtMs;
        this.finishedAtMs = b.finishedAtMs;
        this.lastMetrics = Map.copyOf(b.lastMetrics);
    }

    public static Builder builder() { return new Builder(); }

    public String runId() { return runId; }
    public Optional<String> projectName() { return Optional.ofNullable(projectName); }
    public TrainingStartRequest request() { return request; }
    public Status status() { return status; }
    public Path outputDir() { return outputDir; }
    public int globalStep() { return globalStep; }
    public double lastLoss() { return lastLoss; }
    public Optional<String> error() { return Optional.ofNullable(error); }
    public long createdAtMs() { return createdAtMs; }
    public long updatedAtMs() { return updatedAtMs; }
    public long finishedAtMs() { return finishedAtMs; }
    public Map<String, Double> lastMetrics() { return lastMetrics; }

    public Builder toBuilder() {
        return builder()
                .runId(runId).projectName(projectName).request(request).status(status)
                .outputDir(outputDir).globalStep(globalStep).lastLoss(lastLoss).error(error)
                .createdAtMs(createdAtMs).updatedAtMs(updatedAtMs).finishedAtMs(finishedAtMs)
                .lastMetrics(lastMetrics);
    }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("run_id", runId);
        if (projectName != null) m.put("project_name", projectName);
        m.put("status", status.name());
        if (outputDir != null) m.put("output_dir", outputDir.toString());
        m.put("global_step", globalStep);
        m.put("last_loss", lastLoss);
        if (error != null) m.put("error", error);
        m.put("created_at_ms", createdAtMs);
        m.put("updated_at_ms", updatedAtMs);
        m.put("finished_at_ms", finishedAtMs);
        if (request != null) m.put("request", request.toMap());
        if (!lastMetrics.isEmpty()) m.put("last_metrics", lastMetrics);
        return m;
    }

    public static final class Builder {
        private String runId;
        private String projectName;
        private TrainingStartRequest request;
        private Status status = Status.QUEUED;
        private Path outputDir;
        private int globalStep;
        private double lastLoss = Double.NaN;
        private String error;
        private long createdAtMs = System.currentTimeMillis();
        private long updatedAtMs = System.currentTimeMillis();
        private long finishedAtMs;
        private Map<String, Double> lastMetrics = Map.of();

        public Builder runId(String v) { this.runId = v; return this; }
        public Builder projectName(String v) { this.projectName = v; return this; }
        public Builder request(TrainingStartRequest v) { this.request = v; return this; }
        public Builder status(Status v) { this.status = v; return this; }
        public Builder outputDir(Path v) { this.outputDir = v; return this; }
        public Builder globalStep(int v) { this.globalStep = v; return this; }
        public Builder lastLoss(double v) { this.lastLoss = v; return this; }
        public Builder error(String v) { this.error = v; return this; }
        public Builder createdAtMs(long v) { this.createdAtMs = v; return this; }
        public Builder updatedAtMs(long v) { this.updatedAtMs = v; return this; }
        public Builder finishedAtMs(long v) { this.finishedAtMs = v; return this; }
        public Builder lastMetrics(Map<String, Double> v) { this.lastMetrics = v != null ? v : Map.of(); return this; }
        public TrainingRunRecord build() { return new TrainingRunRecord(this); }
    }
}
