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
package org.bytedeco.pytorch.llm.nltk.metrics;

import org.bytedeco.pytorch.llm.nltk.util.Ngram;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Sentence / corpus BLEU (NLTK-style smoothed geometric mean of modified n-gram precisions).
 */
public final class BleuScore {

    private BleuScore() {}

    public static double sentenceBleu(List<String> hypothesis, List<String> reference) {
        return sentenceBleu(hypothesis, List.of(reference), 4);
    }

    public static double sentenceBleu(List<String> hypothesis, List<List<String>> references, int maxN) {
        Objects.requireNonNull(hypothesis, "hypothesis");
        Objects.requireNonNull(references, "references");
        if (hypothesis.isEmpty() || references.isEmpty()) return 0.0;
        maxN = Math.max(1, Math.min(maxN, hypothesis.size()));
        double[] precisions = new double[maxN];
        for (int n = 1; n <= maxN; n++) {
            Map<String, Integer> hypCounts = countNgrams(hypothesis, n);
            Map<String, Integer> maxRef = new HashMap<>();
            for (List<String> ref : references) {
                Map<String, Integer> rc = countNgrams(ref, n);
                for (Map.Entry<String, Integer> e : rc.entrySet()) {
                    maxRef.merge(e.getKey(), e.getValue(), Math::max);
                }
            }
            int overlap = 0, total = 0;
            for (Map.Entry<String, Integer> e : hypCounts.entrySet()) {
                total += e.getValue();
                overlap += Math.min(e.getValue(), maxRef.getOrDefault(e.getKey(), 0));
            }
            // additive smoothing
            precisions[n - 1] = (overlap + 1.0) / (total + 1.0);
        }
        double logAvg = 0;
        for (double p : precisions) logAvg += Math.log(p);
        logAvg /= maxN;
        int refLen = references.get(0).size();
        for (List<String> r : references) refLen = Math.min(refLen, r.size());
        double bp = brevityPenalty(hypothesis.size(), refLen);
        return bp * Math.exp(logAvg);
    }

    public static double corpusBleu(List<List<String>> hyps, List<List<String>> refs) {
        if (hyps == null || refs == null || hyps.isEmpty()) return 0.0;
        double sum = 0;
        int n = Math.min(hyps.size(), refs.size());
        for (int i = 0; i < n; i++) {
            sum += sentenceBleu(hyps.get(i), List.of(refs.get(i)), 4);
        }
        return sum / n;
    }

    private static double brevityPenalty(int hypLen, int refLen) {
        if (hypLen == 0) return 0;
        if (hypLen > refLen) return 1.0;
        return Math.exp(1.0 - (double) refLen / hypLen);
    }

    private static Map<String, Integer> countNgrams(List<String> tokens, int n) {
        Map<String, Integer> m = new HashMap<>();
        for (List<String> g : Ngram.ngrams(tokens, n)) {
            m.merge(String.join(" ", g), 1, Integer::sum);
        }
        return m;
    }
}
