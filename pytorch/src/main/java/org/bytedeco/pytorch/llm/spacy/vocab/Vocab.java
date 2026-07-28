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
package org.bytedeco.pytorch.llm.spacy.vocab;
import org.bytedeco.pytorch.distributed.*;

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * spaCy-like Vocab: string store + lexemes + optional word vectors + stop words.
 */
public final class Vocab {

    private final StringStore strings = new StringStore();
    private final Map<String, Lexeme> lexemes = new HashMap<>();
    private final Set<String> stopWords = new HashSet<>();
    private int vectorsWidth = 0;

    public Vocab() {
        addDefaultStops();
    }

    public Vocab(Iterable<String> stopWords) {
        if (stopWords != null) {
            for (String s : stopWords) {
                if (s != null) {
                    this.stopWords.add(s.toLowerCase(Locale.ROOT));
                }
            }
        }
    }

    private void addDefaultStops() {
        String[] en = {
                "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by",
                "for", "with", "about", "against", "between", "into", "through",
                "during", "before", "after", "above", "below", "to", "from", "up",
                "down", "in", "out", "on", "off", "over", "under", "again", "further",
                "then", "once", "here", "there", "when", "where", "why", "how", "all",
                "any", "both", "each", "few", "more", "most", "other", "some", "such",
                "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
                "can", "will", "just", "don", "should", "now", "i", "me", "my", "myself",
                "we", "our", "you", "your", "he", "him", "his", "she", "her", "it", "its",
                "they", "them", "their", "what", "which", "who", "whom", "this", "that",
                "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
                "have", "has", "had", "having", "do", "does", "did", "doing", "would",
                "could", "ought", "i'm", "you're", "he's", "she's", "it's", "we're",
                "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
                "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll",
                "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
                "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't",
                "shouldn't", "can't", "cannot", "couldn't", "mustn't", "let's", "that's",
                "who's", "what's", "here's", "there's", "when's", "where's", "why's",
                "how's", "as", "until", "because"
        };
        for (String s : en) {
            stopWords.add(s);
        }
    }

    public Lexeme get(Object key) {
        String k = String.valueOf(key);
        return lexemes.computeIfAbsent(k, Lexeme::new);
    }

    public Lexeme getLexeme(String key) {
        return get(key);
    }

    public void setVector(String key, double[] vector) {
        Lexeme l = get(key);
        l.setVector(vector);
        if (vector != null) {
            vectorsWidth = vector.length;
        }
    }

    public void resetVectors(int width) {
        this.vectorsWidth = Math.max(0, width);
        for (Lexeme l : lexemes.values()) {
            l.setVector(null);
        }
    }

    public StringStore strings() {
        return strings;
    }

    public long addString(String s) {
        return strings.add(s);
    }

    public boolean isStop(String word) {
        if (word == null) {
            return false;
        }
        return stopWords.contains(word.toLowerCase(Locale.ROOT));
    }

    public void addStopWord(String word) {
        if (word != null) {
            stopWords.add(word.toLowerCase(Locale.ROOT));
        }
    }

    public Set<String> stopWords() {
        return Collections.unmodifiableSet(stopWords);
    }

    public int vectorsWidth() {
        return vectorsWidth;
    }

    public int size() {
        return lexemes.size();
    }
}
