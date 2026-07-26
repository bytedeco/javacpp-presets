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
package org.bytedeco.pytorch.utils.nltk.tokenize;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Treebank-ish word tokenizer (simplified NLTK TreebankWordTokenizer rules).
 */
public final class TreebankWordTokenizer {

    private static final Pattern CONTRACTIONS = Pattern.compile(
            "(?i)\\b(can)(not)\\b|\\b(did)(n't)\\b|\\b(c)('m)\\b|\\b(let)('s)\\b|\\b(\\w+)('ll|'re|'ve|n't|'s|'d|'m)\\b");
    private static final Pattern PUNCT = Pattern.compile("([.,:;?!()\\[\\]{}\"'])");

    public List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        if (text == null || text.isEmpty()) return out;
        String s = text.trim();
        // split quotes / punct with spaces
        s = PUNCT.matcher(s).replaceAll(" $1 ");
        s = s.replaceAll("\\.\\.\\.", " ... ");
        // crude contraction split
        Matcher cm = Pattern.compile("(?i)(\\w+)(n't|'ll|'re|'ve|'s|'d|'m)").matcher(s);
        StringBuffer sb = new StringBuffer();
        while (cm.find()) {
            cm.appendReplacement(sb, cm.group(1) + " " + cm.group(2));
        }
        cm.appendTail(sb);
        s = sb.toString();
        for (String tok : s.split("\\s+")) {
            if (!tok.isEmpty()) out.add(tok);
        }
        return out;
    }

    public static List<String> tokenizeStatic(String text) {
        return new TreebankWordTokenizer().tokenize(text);
    }
}
