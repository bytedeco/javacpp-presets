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
package org.bytedeco.pytorch.llm.nltk;

import org.bytedeco.pytorch.llm.nltk.metrics.BleuScore;
import org.bytedeco.pytorch.llm.nltk.metrics.EditDistance;
import org.bytedeco.pytorch.llm.nltk.probability.FreqDist;
import org.bytedeco.pytorch.llm.nltk.stem.LancasterStemmer;
import org.bytedeco.pytorch.llm.nltk.stem.PorterStemmer;
import org.bytedeco.pytorch.llm.nltk.stem.SnowballStemmer;
import org.bytedeco.pytorch.llm.nltk.tag.RegexpTagger;
import org.bytedeco.pytorch.llm.nltk.tokenize.SentTokenizer;
import org.bytedeco.pytorch.llm.nltk.tokenize.TreebankWordTokenizer;
import org.bytedeco.pytorch.llm.nltk.tokenize.WordPunctTokenizer;
import org.bytedeco.pytorch.llm.nltk.util.Ngram;
import org.bytedeco.pytorch.llm.nltk.wordnet.SimpleLexicon;

import java.util.List;

/**
 * NLTK-style pure-Java NLP utilities (tokenize, stem, tag, metrics, lexicon).
 *
 * <p>No corpus downloads required. WordNet is a tiny built-in {@link SimpleLexicon}.
 *
 * <pre>{@code
 * List<String> toks = Nltk.wordTokenize("Hello, world!");
 * List<String> sents = Nltk.sentTokenize("Hi. Bye.");
 * String stem = Nltk.porterStem("running");
 * double bleu = Nltk.sentenceBleu(hyp, ref);
 * }</pre>
 */
public final class Nltk {

    public static final String VERSION = "1.0";

    private static final WordPunctTokenizer WORD_PUNCT = new WordPunctTokenizer();
    private static final TreebankWordTokenizer TREEBANK = new TreebankWordTokenizer();
    private static final SentTokenizer SENT = new SentTokenizer();
    private static final PorterStemmer PORTER = new PorterStemmer();
    private static final SnowballStemmer SNOWBALL = new SnowballStemmer("english");
    private static final LancasterStemmer LANCASTER = new LancasterStemmer();
    private static final RegexpTagger TAGGER = new RegexpTagger();

    private Nltk() {}

    public static String version() { return VERSION; }

    public static List<String> wordTokenize(String text) {
        return WORD_PUNCT.tokenize(text);
    }

    public static List<String> word_tokenize(String text) {
        return wordTokenize(text);
    }

    public static List<String> treebankWordTokenize(String text) {
        return TREEBANK.tokenize(text);
    }

    public static List<String> sentTokenize(String text) {
        return SENT.tokenize(text);
    }

    public static List<String> sent_tokenize(String text) {
        return sentTokenize(text);
    }

    public static String porterStem(String word) {
        return PORTER.stem(word);
    }

    public static String snowballStem(String word) {
        return SNOWBALL.stem(word);
    }

    public static String lancasterStem(String word) {
        return LANCASTER.stem(word);
    }

    public static List<String[]> posTag(List<String> tokens) {
        return TAGGER.tag(tokens);
    }

    public static List<String[]> pos_tag(List<String> tokens) {
        return posTag(tokens);
    }

    public static FreqDist freqDist(List<String> tokens) {
        return new FreqDist(tokens);
    }

    public static double sentenceBleu(List<String> hypothesis, List<String> reference) {
        return BleuScore.sentenceBleu(hypothesis, reference);
    }

    public static double bleu_score(List<String> hypothesis, List<String> reference) {
        return sentenceBleu(hypothesis, reference);
    }

    public static int editDistance(String a, String b) {
        return EditDistance.editDistance(a, b);
    }

    public static int edit_distance(String a, String b) {
        return editDistance(a, b);
    }

    public static List<List<String>> ngrams(List<String> tokens, int n) {
        return Ngram.ngrams(tokens, n);
    }

    public static List<List<String>> bigrams(List<String> tokens) {
        return Ngram.bigrams(tokens);
    }

    public static SimpleLexicon wordnet() {
        return SimpleLexicon.getDefault();
    }

    public static String info() {
        return "org.bytedeco.pytorch.llm.nltk v" + VERSION
                + " (NLTK-like pure Java: tokenize/stem/tag/metrics/lexicon)";
    }
}
