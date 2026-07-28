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
package org.bytedeco.pytorch.llm.ragas.llms;

import java.util.Optional;
import java.util.Set;

/** Default offline heuristic judge using token overlap / word F1 / Jaccard. */
public final class HeuristicJudge implements LlmJudge {
    private static final Set<String> POSITIVE = Set.of("yes", "true", "1", "correct", "good");
    private static final Set<String> NEGATIVE = Set.of("no", "false", "0", "wrong", "bad");

    @Override public String generate(String prompt) { return ""; }
    @Override public boolean available() { return true; }

    @Override
    public Optional<Boolean> extractYesNo(String text) {
        if (text == null) return Optional.empty();
        String lower = text.trim().toLowerCase();
        for (String p : POSITIVE) if (lower.startsWith(p)) return Optional.of(true);
        for (String n : NEGATIVE) if (lower.startsWith(n)) return Optional.of(false);
        return Optional.empty();
    }

    @Override public float[] embed(String text) { return new float[64]; }

    /** Word-level F1 between two texts. */
    public static double wordF1(String a, String b) {
        if (a == null || b == null || a.isEmpty() || b.isEmpty()) return 0;
        Set<String> wa = java.util.Set.of(a.toLowerCase().split("\\s+"));
        Set<String> wb = java.util.Set.of(b.toLowerCase().split("\\s+"));
        int inter = 0;
        for (String w : wa) if (wb.contains(w)) inter++;
        double recall = wb.isEmpty() ? 0 : (double) inter / wb.size();
        double prec = wa.isEmpty() ? 0 : (double) inter / wa.size();
        return (recall + prec == 0) ? 0 : 2.0 * recall * prec / (recall + prec);
    }

    /** Jaccard similarity between two texts. */
    public static double jaccard(String a, String b) {
        if (a == null || b == null || a.isEmpty() || b.isEmpty()) return 0;
        Set<String> wa = java.util.Set.of(a.toLowerCase().split("\\s+"));
        Set<String> wb = java.util.Set.of(b.toLowerCase().split("\\s+"));
        int inter = 0;
        for (String w : wa) if (wb.contains(w)) inter++;
        int union = wa.size() + wb.size() - inter;
        return union == 0 ? 0 : (double) inter / union;
    }
}
