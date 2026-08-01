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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/** Ring-buffer of scalar series for Board SVG charts. */
public final class LiveGraphBuffer implements MetricsSink {

    private final Map<String, CopyOnWriteArrayList<Double>> series = new ConcurrentHashMap<>();
    private final Map<String, CopyOnWriteArrayList<Integer>> steps = new ConcurrentHashMap<>();
    private final int capacity;

    public LiveGraphBuffer() { this(2048); }
    public LiveGraphBuffer(int capacity) { this.capacity = Math.max(16, capacity); }

    @Override
    public String name() { return "live-graph"; }

    @Override
    public void record(TrainingMetrics metrics) {
        if (metrics == null || Double.isNaN(metrics.loss())) return;
        push(metrics.runId() + "/loss", metrics.step(), metrics.loss());
        if (metrics.learningRate() > 0) {
            push(metrics.runId() + "/lr", metrics.step(), metrics.learningRate());
        }
        if (metrics.tokensPerSecond() > 0) {
            push(metrics.runId() + "/tps", metrics.step(), metrics.tokensPerSecond());
        }
    }

    public void push(String key, int step, double value) {
        CopyOnWriteArrayList<Double> s = series.computeIfAbsent(key, k -> new CopyOnWriteArrayList<>());
        CopyOnWriteArrayList<Integer> st = steps.computeIfAbsent(key, k -> new CopyOnWriteArrayList<>());
        s.add(value);
        st.add(step);
        while (s.size() > capacity) s.remove(0);
        while (st.size() > capacity) st.remove(0);
    }

    public List<Double> series(String key) {
        CopyOnWriteArrayList<Double> s = series.get(key);
        return s == null ? List.of() : List.copyOf(s);
    }

    public List<Integer> steps(String key) {
        CopyOnWriteArrayList<Integer> s = steps.get(key);
        return s == null ? List.of() : List.copyOf(s);
    }

    public int size(String key) {
        CopyOnWriteArrayList<Double> s = series.get(key);
        return s == null ? 0 : s.size();
    }

    /** Simple SVG polyline for a series. */
    public String toSvg(String key, int width, int height) {
        List<Double> vals = series(key);
        if (vals.isEmpty()) {
            return "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + width + "\" height=\"" + height
                    + "\"><text x=\"10\" y=\"20\" fill=\"#888\">no data</text></svg>";
        }
        double min = vals.stream().mapToDouble(d -> d).min().orElse(0);
        double max = vals.stream().mapToDouble(d -> d).max().orElse(1);
        if (max <= min) max = min + 1e-6;
        StringBuilder pts = new StringBuilder();
        int n = vals.size();
        for (int i = 0; i < n; i++) {
            double x = n == 1 ? width / 2.0 : (i * (width - 20.0) / (n - 1.0)) + 10;
            double y = height - 10 - ((vals.get(i) - min) / (max - min)) * (height - 20.0);
            if (i > 0) pts.append(' ');
            pts.append(String.format("%.1f,%.1f", x, y));
        }
        return "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + width + "\" height=\"" + height
                + "\" style=\"background:#0b1220\">"
                + "<polyline fill=\"none\" stroke=\"#5b9cff\" stroke-width=\"2\" points=\"" + pts + "\"/>"
                + "<text x=\"10\" y=\"14\" fill=\"#9fb3c8\" font-size=\"11\">" + key + "</text>"
                + "</svg>";
    }
}
