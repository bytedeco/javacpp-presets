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
package org.bytedeco.pytorch.llm.nltk.tokenize;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Punkt-like sentence tokenizer (rule-based, no trained model).
 */
public final class SentTokenizer {

    private static final Pattern SPLIT = Pattern.compile("(?<=[.!?])\\s+(?=[A-Z\"'])");

    public List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        if (text == null || text.isEmpty()) return out;
        String t = text.trim();
        if (t.isEmpty()) return out;
        String[] parts = SPLIT.split(t);
        for (String p : parts) {
            String s = p.trim();
            if (!s.isEmpty()) out.add(s);
        }
        if (out.isEmpty()) out.add(t);
        return out;
    }

    public static List<String> tokenizeStatic(String text) {
        return new SentTokenizer().tokenize(text);
    }
}
