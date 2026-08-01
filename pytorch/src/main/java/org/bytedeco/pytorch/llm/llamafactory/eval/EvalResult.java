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
package org.bytedeco.pytorch.llm.llamafactory.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Result of an evaluation harness run (accuracy in {@code [0, 1]} + per-item log).
 */
public final class EvalResult {

    private final String task;
    private final int total;
    private final int correct;
    private final double accuracy;
    private final List<Map<String, Object>> items;
    private final Map<String, Object> meta;

    public EvalResult(
            String task,
            int total,
            int correct,
            List<Map<String, Object>> items,
            Map<String, Object> meta) {
        this.task = task == null ? "unknown" : task;
        this.total = Math.max(0, total);
        this.correct = Math.max(0, correct);
        this.accuracy = this.total == 0 ? 0.0 : (double) this.correct / (double) this.total;
        this.items = items == null
                ? List.of()
                : Collections.unmodifiableList(new ArrayList<>(items));
        this.meta = meta == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(meta));
    }

    public String task() { return task; }
    public int total() { return total; }
    public int correct() { return correct; }
    public double accuracy() { return accuracy; }
    public List<Map<String, Object>> items() { return items; }
    public Map<String, Object> meta() { return meta; }

    public Map<String, Object> toMap() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("task", task);
        m.put("total", total);
        m.put("correct", correct);
        m.put("accuracy", accuracy);
        m.put("items", items);
        m.put("meta", meta);
        return m;
    }

    @Override
    public String toString() {
        return "EvalResult{task=" + task + ", acc=" + accuracy
                + " (" + correct + "/" + total + ")}";
    }
}
