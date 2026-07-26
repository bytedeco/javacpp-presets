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
package org.bytedeco.pytorch.utils.nltk.probability;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * NLTK-style conditional frequency distribution: condition → {@link FreqDist}.
 */
public final class ConditionalFreqDist {

    private final Map<String, FreqDist> conditions = new LinkedHashMap<>();

    public void inc(String condition, String sample) {
        Objects.requireNonNull(condition, "condition");
        conditions.computeIfAbsent(condition, c -> new FreqDist()).inc(sample);
    }

    public void inc(String condition, String sample, int n) {
        Objects.requireNonNull(condition, "condition");
        conditions.computeIfAbsent(condition, c -> new FreqDist()).inc(sample, n);
    }

    public FreqDist get(String condition) {
        return conditions.getOrDefault(condition, new FreqDist());
    }

    public Set<String> conditions() {
        return Collections.unmodifiableSet(conditions.keySet());
    }

    public int N() {
        int n = 0;
        for (FreqDist fd : conditions.values()) n += fd.N();
        return n;
    }

    public Map<String, FreqDist> asMap() {
        return Collections.unmodifiableMap(conditions);
    }
}
