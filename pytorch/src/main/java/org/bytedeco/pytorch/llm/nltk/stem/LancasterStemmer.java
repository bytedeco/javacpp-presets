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
package org.bytedeco.pytorch.llm.nltk.stem;

/**
 * Aggressive Lancaster-style stemmer (simplified rule table).
 */
public final class LancasterStemmer {

    private static final String[][] RULES = {
            {"ness", ""}, {"ing", ""}, {"ied", "y"}, {"ies", "y"}, {"ed", ""},
            {"ly", ""}, {"ful", ""}, {"less", ""}, {"ment", ""}, {"able", ""},
            {"ible", ""}, {"al", ""}, {"er", ""}, {"est", ""}, {"s", ""}
    };

    public String stem(String word) {
        if (word == null || word.length() < 3) return word == null ? "" : word.toLowerCase();
        String w = word.toLowerCase();
        boolean changed = true;
        int guard = 0;
        while (changed && w.length() > 3 && guard++ < 8) {
            changed = false;
            for (String[] r : RULES) {
                if (w.endsWith(r[0]) && w.length() - r[0].length() + r[1].length() >= 3) {
                    w = w.substring(0, w.length() - r[0].length()) + r[1];
                    changed = true;
                    break;
                }
            }
        }
        return w;
    }
}
