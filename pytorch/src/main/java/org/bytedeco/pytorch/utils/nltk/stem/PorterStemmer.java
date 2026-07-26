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
package org.bytedeco.pytorch.utils.nltk.stem;

/**
 * Porter stemming algorithm (M.F. Porter, 1980) — pure Java.
 */
public final class PorterStemmer {

    public String stem(String word) {
        if (word == null || word.length() <= 2) return word == null ? "" : word.toLowerCase();
        String w = word.toLowerCase();
        w = step1a(w);
        w = step1b(w);
        w = step1c(w);
        w = step2(w);
        w = step3(w);
        w = step4(w);
        w = step5a(w);
        w = step5b(w);
        return w;
    }

    private static boolean isConsonant(String w, int i) {
        char c = w.charAt(i);
        if ("aeiou".indexOf(c) >= 0) return false;
        if (c == 'y') return i == 0 || !isConsonant(w, i - 1);
        return true;
    }

    private static int m(String w) {
        int n = 0, i = 0, len = w.length();
        while (i < len && isConsonant(w, i)) i++;
        while (i < len) {
            while (i < len && !isConsonant(w, i)) i++;
            n++;
            while (i < len && isConsonant(w, i)) i++;
        }
        return n;
    }

    private static boolean hasVowel(String w) {
        for (int i = 0; i < w.length(); i++) if (!isConsonant(w, i)) return true;
        return false;
    }

    private static boolean cvc(String w) {
        int len = w.length();
        if (len < 3) return false;
        return isConsonant(w, len - 1) && !isConsonant(w, len - 2) && isConsonant(w, len - 3)
                && "wxy".indexOf(w.charAt(len - 1)) < 0;
    }

    private static String step1a(String w) {
        if (w.endsWith("sses")) return w.substring(0, w.length() - 2);
        if (w.endsWith("ies")) return w.substring(0, w.length() - 2);
        if (w.endsWith("ss")) return w;
        if (w.endsWith("s")) return w.substring(0, w.length() - 1);
        return w;
    }

    private static String step1b(String w) {
        if (w.endsWith("eed")) {
            String stem = w.substring(0, w.length() - 3);
            if (m(stem) > 0) return stem + "ee";
            return w;
        }
        boolean flag = false;
        if (w.endsWith("ed") && hasVowel(w.substring(0, w.length() - 2))) {
            w = w.substring(0, w.length() - 2);
            flag = true;
        } else if (w.endsWith("ing") && hasVowel(w.substring(0, w.length() - 3))) {
            w = w.substring(0, w.length() - 3);
            flag = true;
        }
        if (flag) {
            if (w.endsWith("at") || w.endsWith("bl") || w.endsWith("iz")) return w + "e";
            if (w.length() >= 2 && w.charAt(w.length() - 1) == w.charAt(w.length() - 2)
                    && "lsz".indexOf(w.charAt(w.length() - 1)) < 0
                    && isConsonant(w, w.length() - 1)) {
                return w.substring(0, w.length() - 1);
            }
            if (m(w) == 1 && cvc(w)) return w + "e";
        }
        return w;
    }

    private static String step1c(String w) {
        if (w.endsWith("y") && hasVowel(w.substring(0, w.length() - 1))) {
            return w.substring(0, w.length() - 1) + "i";
        }
        return w;
    }

    private static String replaceIf(String w, String suf, String rep, int minM) {
        if (w.endsWith(suf)) {
            String stem = w.substring(0, w.length() - suf.length());
            if (m(stem) > minM) return stem + rep;
        }
        return w;
    }

    private static String step2(String w) {
        String[][] rules = {
                {"ational", "ate"}, {"tional", "tion"}, {"enci", "ence"}, {"anci", "ance"},
                {"izer", "ize"}, {"abli", "able"}, {"alli", "al"}, {"entli", "ent"},
                {"eli", "e"}, {"ousli", "ous"}, {"ization", "ize"}, {"ation", "ate"},
                {"ator", "ate"}, {"alism", "al"}, {"iveness", "ive"}, {"fulness", "ful"},
                {"ousness", "ous"}, {"aliti", "al"}, {"iviti", "ive"}, {"biliti", "ble"}
        };
        for (String[] r : rules) {
            String n = replaceIf(w, r[0], r[1], 0);
            if (!n.equals(w)) return n;
        }
        return w;
    }

    private static String step3(String w) {
        String[][] rules = {
                {"icate", "ic"}, {"ative", ""}, {"alize", "al"}, {"iciti", "ic"},
                {"ical", "ic"}, {"ful", ""}, {"ness", ""}
        };
        for (String[] r : rules) {
            String n = replaceIf(w, r[0], r[1], 0);
            if (!n.equals(w)) return n;
        }
        return w;
    }

    private static String step4(String w) {
        String[] sufs = {"al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
                "ment", "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize"};
        for (String s : sufs) {
            if (w.endsWith(s)) {
                String stem = w.substring(0, w.length() - s.length());
                if ("ion".equals(s)) {
                    if (stem.endsWith("s") || stem.endsWith("t")) {
                        if (m(stem) > 1) return stem;
                    }
                } else if (m(stem) > 1) {
                    return stem;
                }
            }
        }
        return w;
    }

    private static String step5a(String w) {
        if (w.endsWith("e")) {
            String stem = w.substring(0, w.length() - 1);
            int mm = m(stem);
            if (mm > 1 || (mm == 1 && !cvc(stem))) return stem;
        }
        return w;
    }

    private static String step5b(String w) {
        if (m(w) > 1 && w.length() >= 2
                && w.charAt(w.length() - 1) == 'l'
                && w.charAt(w.length() - 2) == 'l') {
            return w.substring(0, w.length() - 1);
        }
        return w;
    }
}
