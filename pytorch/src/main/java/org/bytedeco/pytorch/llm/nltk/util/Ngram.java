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
package org.bytedeco.pytorch.llm.nltk.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * NLTK-style n-gram helpers: {@code ngrams}, {@code bigrams}, {@code everygrams}.
 */
public final class Ngram {

    private Ngram() {}

    public static List<List<String>> ngrams(List<String> tokens, int n) {
        List<List<String>> out = new ArrayList<>();
        if (tokens == null || n <= 0 || tokens.size() < n) return out;
        for (int i = 0; i <= tokens.size() - n; i++) {
            out.add(List.copyOf(tokens.subList(i, i + n)));
        }
        return out;
    }

    public static List<List<String>> bigrams(List<String> tokens) {
        return ngrams(tokens, 2);
    }

    public static List<List<String>> trigrams(List<String> tokens) {
        return ngrams(tokens, 3);
    }

    /** All n-grams for n in [1, maxN]. */
    public static List<List<String>> everygrams(List<String> tokens, int maxN) {
        List<List<String>> out = new ArrayList<>();
        if (tokens == null || maxN <= 0) return out;
        int lim = Math.min(maxN, tokens.size());
        for (int n = 1; n <= lim; n++) {
            out.addAll(ngrams(tokens, n));
        }
        return out;
    }

    public static List<List<String>> everygrams(List<String> tokens) {
        return everygrams(tokens, tokens == null ? 0 : tokens.size());
    }

    public static List<String> padLeft(List<String> tokens, int n, String pad) {
        if (tokens == null) return Collections.emptyList();
        List<String> out = new ArrayList<>(n + tokens.size());
        for (int i = 0; i < Math.max(0, n - 1); i++) out.add(pad);
        out.addAll(tokens);
        return out;
    }
}
