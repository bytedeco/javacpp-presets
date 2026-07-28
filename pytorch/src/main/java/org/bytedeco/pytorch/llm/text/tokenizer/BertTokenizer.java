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
package org.bytedeco.pytorch.llm.text.tokenizer;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * BERT-style tokenizer: basic English split + WordPiece with [CLS]/[SEP]/##subwords.
 */
public final class BertTokenizer implements Tokenizer {

    private final WordPieceTokenizer wordPiece;
    private final boolean addSpecialTokens;

    public BertTokenizer(WordPieceTokenizer wordPiece) {
        this(wordPiece, true);
    }

    public BertTokenizer(WordPieceTokenizer wordPiece, boolean addSpecialTokens) {
        this.wordPiece = wordPiece == null ? defaultWordPiece() : wordPiece;
        this.addSpecialTokens = addSpecialTokens;
    }

    public BertTokenizer(Map<String, Integer> vocab) {
        this(new WordPieceTokenizer(vocab), true);
    }

    public static BertTokenizer fromFile(Path vocabFile) {
        return new BertTokenizer(WordPieceTokenizer.fromFile(vocabFile), true);
    }

    public static BertTokenizer fromCorpus(Iterable<? extends Iterable<String>> corpus, int minFreq, int maxVocab) {
        return new BertTokenizer(WordPieceTokenizer.buildFromCorpus(corpus, minFreq, maxVocab), true);
    }

    private static WordPieceTokenizer defaultWordPiece() {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put(WordPieceTokenizer.DEFAULT_PAD, 0);
        vocab.put(WordPieceTokenizer.DEFAULT_UNK, 1);
        vocab.put(WordPieceTokenizer.DEFAULT_CLS, 2);
        vocab.put(WordPieceTokenizer.DEFAULT_SEP, 3);
        vocab.put(WordPieceTokenizer.DEFAULT_MASK, 4);
        // minimal latin chars
        for (char c = 'a'; c <= 'z'; c++) {
            vocab.put(String.valueOf(c), vocab.size());
            vocab.put("##" + c, vocab.size());
        }
        for (char c = '0'; c <= '9'; c++) {
            vocab.put(String.valueOf(c), vocab.size());
            vocab.put("##" + c, vocab.size());
        }
        return new WordPieceTokenizer(vocab);
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> pieces = wordPiece.tokenize(text);
        if (!addSpecialTokens) {
            return pieces;
        }
        List<String> out = new ArrayList<>(pieces.size() + 2);
        out.add(WordPieceTokenizer.DEFAULT_CLS);
        out.addAll(pieces);
        out.add(WordPieceTokenizer.DEFAULT_SEP);
        return out;
    }

    /** Encode a single sentence (with optional specials). */
    @Override
    public int[] encode(String text) {
        return encodeTokens(tokenize(text));
    }

    /** Encode sentence pair as [CLS] A [SEP] B [SEP]. */
    public int[] encodePair(String textA, String textB) {
        List<String> a = wordPiece.tokenize(textA);
        List<String> b = wordPiece.tokenize(textB);
        List<String> tokens = new ArrayList<>(a.size() + b.size() + 3);
        tokens.add(WordPieceTokenizer.DEFAULT_CLS);
        tokens.addAll(a);
        tokens.add(WordPieceTokenizer.DEFAULT_SEP);
        tokens.addAll(b);
        tokens.add(WordPieceTokenizer.DEFAULT_SEP);
        return encodeTokens(tokens);
    }

    @Override
    public int[] encodeTokens(List<String> tokens) {
        return wordPiece.encodeTokens(tokens);
    }

    @Override
    public String decode(int[] ids) {
        return wordPiece.decode(ids);
    }

    public WordPieceTokenizer wordPiece() {
        return wordPiece;
    }

    public int vocabSize() {
        return wordPiece.vocabSize();
    }
}
