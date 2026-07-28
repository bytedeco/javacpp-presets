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
package org.bytedeco.pytorch.llm.nltk.probability;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * NLTK-style frequency distribution.
 */
public final class FreqDist {

    private final Map<String, Integer> counts = new LinkedHashMap<>();
    private int N;

    public FreqDist() {}

    public FreqDist(Collection<String> samples) {
        if (samples != null) {
            for (String s : samples) inc(s);
        }
    }

    public void inc(String sample) {
        inc(sample, 1);
    }

    public void inc(String sample, int n) {
        Objects.requireNonNull(sample, "sample");
        counts.merge(sample, n, Integer::sum);
        N += n;
    }

    public int count(String sample) {
        return counts.getOrDefault(sample, 0);
    }

    public double freq(String sample) {
        return N == 0 ? 0.0 : (double) count(sample) / (double) N;
    }

    public int N() { return N; }
    public int B() { return counts.size(); }

    public List<String> hapaxes() {
        List<String> h = new ArrayList<>();
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            if (e.getValue() == 1) h.add(e.getKey());
        }
        return h;
    }

    public List<Map.Entry<String, Integer>> mostCommon(int n) {
        List<Map.Entry<String, Integer>> list = new ArrayList<>(counts.entrySet());
        list.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));
        if (n >= 0 && n < list.size()) return list.subList(0, n);
        return list;
    }

    public List<Map.Entry<String, Integer>> mostCommon() {
        return mostCommon(-1);
    }

    public Map<String, Integer> counts() {
        return Collections.unmodifiableMap(counts);
    }

    public double[] plotData() {
        List<Map.Entry<String, Integer>> mc = mostCommon();
        double[] y = new double[mc.size()];
        for (int i = 0; i < mc.size(); i++) y[i] = mc.get(i).getValue();
        return y;
    }

    @Override
    public String toString() {
        return "FreqDist(N=" + N + ", B=" + B() + ")";
    }
}
