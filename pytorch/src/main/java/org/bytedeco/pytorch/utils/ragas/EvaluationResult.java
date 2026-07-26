/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to "Classpath" exception),
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
package org.bytedeco.pytorch.utils.ragas;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Results of RAG evaluation. */
public final class EvaluationResult {
    private final Map<String, double[]> perMetricScores;
    private final List<String> metricNames;

    public EvaluationResult(Map<String, double[]> perMetricScores) {
        this.perMetricScores = new LinkedHashMap<>(perMetricScores);
        this.metricNames = List.copyOf(perMetricScores.keySet());
    }

    /** Mean score for a metric. */
    public double mean(String metric) {
        double[] s = perMetricScores.get(metric);
        if (s == null || s.length == 0) return Double.NaN;
        double sum = 0;
        for (double v : s) sum += v;
        return sum / s.length;
    }

    /** Per-sample scores for a metric. */
    public double[] scores(String metric) {
        return perMetricScores.getOrDefault(metric, new double[0]);
    }

    public Map<String, Double> toMap() {
        Map<String, Double> m = new LinkedHashMap<>();
        for (String n : metricNames) m.put(n, mean(n));
        return m;
    }

    public List<String> metricNames() { return metricNames; }
    public int numSamples() {
        var vals = perMetricScores.values();
        return vals.isEmpty() ? 0 : vals.iterator().next().length;
    }

    @Override
    public String toString() {
        return toMap().toString();
    }
}
