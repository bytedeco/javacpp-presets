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

package org.bytedeco.pytorch.llm.unsloth.studio.observe;

import org.bytedeco.pytorch.llm.unsloth.studio.model.TrainingProgressEvent;

import java.util.LinkedHashMap;
import java.util.Map;

public final class TrainingMetrics {
    private final String runId;
    private final int step;
    private final double loss;
    private final double learningRate;
    private final double tokensPerSecond;
    private final double gpuMemoryUsedMb;
    private final long timestampMs;
    private final Map<String, Double> extras;

    public TrainingMetrics(String runId, int step, double loss, double learningRate,
                           double tokensPerSecond, double gpuMemoryUsedMb, long timestampMs,
                           Map<String, Double> extras) {
        this.runId = runId;
        this.step = step;
        this.loss = loss;
        this.learningRate = learningRate;
        this.tokensPerSecond = tokensPerSecond;
        this.gpuMemoryUsedMb = gpuMemoryUsedMb;
        this.timestampMs = timestampMs;
        this.extras = extras != null ? Map.copyOf(extras) : Map.of();
    }

    public static TrainingMetrics from(TrainingProgressEvent ev) {
        return new TrainingMetrics(ev.runId(), ev.step(), ev.loss(), ev.learningRate(),
                ev.tokensPerSecond(), ev.gpuMemoryUsedMb(), ev.timestampMs(), ev.metrics());
    }

    public String runId() { return runId; }
    public int step() { return step; }
    public double loss() { return loss; }
    public double learningRate() { return learningRate; }
    public double tokensPerSecond() { return tokensPerSecond; }
    public double gpuMemoryUsedMb() { return gpuMemoryUsedMb; }
    public long timestampMs() { return timestampMs; }
    public Map<String, Double> extras() { return extras; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("run_id", runId);
        m.put("step", step);
        m.put("loss", loss);
        m.put("learning_rate", learningRate);
        m.put("tokens_per_second", tokensPerSecond);
        m.put("gpu_memory_used_mb", gpuMemoryUsedMb);
        m.put("timestamp_ms", timestampMs);
        if (!extras.isEmpty()) m.put("extras", extras);
        return m;
    }
}
