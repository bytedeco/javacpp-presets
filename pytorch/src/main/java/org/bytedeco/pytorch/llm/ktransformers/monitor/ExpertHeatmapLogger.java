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
package org.bytedeco.pytorch.llm.ktransformers.monitor;

import org.bytedeco.pytorch.llm.ktransformers.moe.ExpertLoadBalanceMetrics;

import java.util.Arrays;
import java.util.Locale;
import java.util.Map;

/**
 * Formats expert selection histograms for board / log / TensorBoard scalar export.
 */
public final class ExpertHeatmapLogger {

    private final KtMetrics metrics;

    public ExpertHeatmapLogger(KtMetrics metrics) {
        this.metrics = metrics != null ? metrics : new KtMetrics();
    }

    public void log(ExpertLoadBalanceMetrics load) {
        if (load == null) return;
        Map<String, Double> m = load.toMetricMap();
        metrics.setAll(m);
        double[] freq = load.frequency();
        if (freq.length > 0) {
            double max = 0;
            int argmax = 0;
            for (int i = 0; i < freq.length; i++) {
                if (freq[i] > max) {
                    max = freq[i];
                    argmax = i;
                }
            }
            metrics.set("kt/moe/hottest_expert", argmax);
            metrics.set("kt/moe/hottest_freq", max);
        }
    }

    /** ASCII bar row for console demos (no GUI dependency). */
    public static String asciiHeatmap(double[] freq, int width) {
        if (freq == null || freq.length == 0) return "(empty)";
        int w = Math.max(8, width);
        StringBuilder sb = new StringBuilder();
        double max = Arrays.stream(freq).max().orElse(1.0);
        if (max <= 0) max = 1.0;
        for (int i = 0; i < freq.length; i++) {
            int bars = (int) Math.round((freq[i] / max) * w);
            sb.append(String.format(Locale.ROOT, "E%02d |", i));
            for (int b = 0; b < bars; b++) sb.append('#');
            for (int b = bars; b < w; b++) sb.append(' ');
            sb.append(String.format(Locale.ROOT, "| %.3f%n", freq[i]));
        }
        return sb.toString();
    }

    public KtMetrics metrics() { return metrics; }
}
