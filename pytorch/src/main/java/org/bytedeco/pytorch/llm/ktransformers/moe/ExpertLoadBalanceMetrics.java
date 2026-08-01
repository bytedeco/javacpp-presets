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

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/**
 * Per-expert selection histogram and migrate counters for CPU–GPU scheduling.
 *
 * <p>Feeds {@link ExpertScheduler} AUTO/BALANCED policies and
 * {@code monitor.ExpertHeatmapLogger} for visual training dashboards.
 */
public final class ExpertLoadBalanceMetrics {

    private final int numExperts;
    private final long[] hits;
    private final LongAdder gpuHits = new LongAdder();
    private final LongAdder cpuHits = new LongAdder();
    private final LongAdder promoteCount = new LongAdder();
    private final LongAdder demoteCount = new LongAdder();
    private final LongAdder dispatchSteps = new LongAdder();
    private long totalHits;

    public ExpertLoadBalanceMetrics(int numExperts) {
        if (numExperts < 1) {
            throw new IllegalArgumentException("numExperts must be >= 1");
        }
        this.numExperts = numExperts;
        this.hits = new long[numExperts];
        this.totalHits = 0L;
    }

    public synchronized void recordSelection(int expertId, ExpertDevice device) {
        if (expertId < 0 || expertId >= numExperts) {
            return;
        }
        hits[expertId]++;
        totalHits++;
        if (device == ExpertDevice.GPU) {
            gpuHits.increment();
        } else {
            cpuHits.increment();
        }
    }

    public synchronized void recordSelections(int[] expertIds, ExpertDevice[] devices) {
        if (expertIds == null) return;
        for (int i = 0; i < expertIds.length; i++) {
            ExpertDevice d = devices != null && i < devices.length ? devices[i] : ExpertDevice.CPU;
            recordSelection(expertIds[i], d);
        }
    }

    public void recordPromote() { promoteCount.increment(); }
    public void recordDemote() { demoteCount.increment(); }
    public void recordDispatchStep() { dispatchSteps.increment(); }

    public int numExperts() { return numExperts; }

    public synchronized long[] hitSnapshot() {
        return Arrays.copyOf(hits, hits.length);
    }

    public synchronized double[] frequency() {
        double[] f = new double[numExperts];
        double den = totalHits <= 0 ? 1.0 : (double) totalHits;
        for (int i = 0; i < numExperts; i++) {
            f[i] = hits[i] / den;
        }
        return f;
    }

    public synchronized int hottestExpert() {
        int best = 0;
        long bestH = -1;
        for (int i = 0; i < numExperts; i++) {
            if (hits[i] > bestH) {
                bestH = hits[i];
                best = i;
            }
        }
        return best;
    }

    public synchronized int coldestOn(ExpertDevice device, ExpertSpec[] specs) {
        int best = -1;
        long bestH = Long.MAX_VALUE;
        for (int i = 0; i < numExperts; i++) {
            if (specs[i] != null && specs[i].device() == device) {
                if (hits[i] < bestH) {
                    bestH = hits[i];
                    best = i;
                }
            }
        }
        return best;
    }

    public long gpuHits() { return gpuHits.sum(); }
    public long cpuHits() { return cpuHits.sum(); }
    public long promoteCount() { return promoteCount.sum(); }
    public long demoteCount() { return demoteCount.sum(); }
    public long dispatchSteps() { return dispatchSteps.sum(); }
    public synchronized long totalHits() { return totalHits; }

    public synchronized void reset() {
        Arrays.fill(hits, 0L);
        totalHits = 0L;
        gpuHits.reset();
        cpuHits.reset();
        promoteCount.reset();
        demoteCount.reset();
        dispatchSteps.reset();
    }

    /** Flat metrics map for TensorBoard / BoardState. */
    public synchronized Map<String, Double> toMetricMap() {
        Map<String, Double> m = new LinkedHashMap<>();
        m.put("kt/moe/total_hits", (double) totalHits);
        m.put("kt/moe/gpu_hits", (double) gpuHits.sum());
        m.put("kt/moe/cpu_hits", (double) cpuHits.sum());
        m.put("kt/moe/promotes", (double) promoteCount.sum());
        m.put("kt/moe/demotes", (double) demoteCount.sum());
        m.put("kt/moe/dispatch_steps", (double) dispatchSteps.sum());
        double[] f = frequency();
        for (int i = 0; i < Math.min(f.length, 32); i++) {
            m.put("kt/moe/expert_freq/" + i, f[i]);
        }
        return m;
    }
}
