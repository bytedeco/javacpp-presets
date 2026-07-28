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
package org.bytedeco.pytorch.llm.ragas.dataset;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Evaluation dataset holding a list of samples. */
public final class EvaluationDataset {
    private final List<SingleTurnSample> samples = new ArrayList<>();

    public EvaluationDataset(List<SingleTurnSample> samples) {
        if (samples != null) this.samples.addAll(samples);
    }

    public static EvaluationDataset of(List<SingleTurnSample> samples) {
        return new EvaluationDataset(samples);
    }

    public void add(SingleTurnSample s) { samples.add(Objects.requireNonNull(s)); }
    public List<SingleTurnSample> samples() { return List.copyOf(samples); }
    public int size() { return samples.size(); }
    public SingleTurnSample get(int i) { return samples.get(i); }
}
