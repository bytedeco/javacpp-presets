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
package org.bytedeco.pytorch.utils.spacy.tokenizer;
import org.bytedeco.pytorch.nn.modules.*;

import org.bytedeco.pytorch.utils.spacy.Doc;
import org.bytedeco.pytorch.utils.spacy.Language;
import org.bytedeco.pytorch.utils.spacy.impl.DocImpl;
import org.bytedeco.pytorch.utils.spacy.impl.TokenImpl;
import org.bytedeco.pytorch.utils.spacy.vocab.Vocab;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Rule-based tokenizer: whitespace + punctuation split (spaCy-like prefix/suffix lite).
 * Handles contractions, URLs/emails roughly, and keeps character offsets.
 */
public final class SimpleTokenizer {

    // words, numbers, urls-ish, emails-ish, or single punct/symbol
    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "https?://\\S+|www\\.\\S+"
                    + "|[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}"
                    + "|\\d+(?:[.,]\\d+)*%?"
                    + "|[\\p{L}]+(?:[''][\\p{L}]+)*"
                    + "|[^\\s\\p{L}\\p{N}]"
                    + "|\\s+"
    );

    private final Vocab vocab;

    public SimpleTokenizer() {
        this(new Vocab());
    }

    public SimpleTokenizer(Vocab vocab) {
        this.vocab = vocab == null ? new Vocab() : vocab;
    }

    public Doc tokenize(String text) {
        return tokenize(text, null);
    }

    public Doc tokenize(String text, Language language) {
        if (text == null) {
            text = "";
        }
        List<TokenImpl> tokens = new ArrayList<>();
        Matcher m = TOKEN_PATTERN.matcher(text);
        int i = 0;
        while (m.find()) {
            String piece = m.group();
            int start = m.start();
            // skip pure whitespace tokens but track whitespace on previous token
            if (piece.trim().isEmpty()) {
                if (!tokens.isEmpty()) {
                    TokenImpl prev = tokens.get(tokens.size() - 1);
                    prev.setWhitespace(prev.whitespace() + piece);
                }
                continue;
            }
            // special: split trailing punctuation already handled by pattern
            TokenImpl tok = new TokenImpl(piece, start, i, vocab);
            if (vocab.isStop(piece)) {
                tok.setStop(true);
            }
            tokens.add(tok);
            i++;
        }
        // attach inter-token whitespace for pieces that weren't pure-ws (gap between matches)
        // recompute whitespace from text offsets
        for (int t = 0; t < tokens.size(); t++) {
            TokenImpl cur = tokens.get(t);
            int end = cur.getIdx() + cur.getText().length();
            int nextStart = (t + 1 < tokens.size()) ? tokens.get(t + 1).getIdx() : text.length();
            if (nextStart > end) {
                cur.setWhitespace(text.substring(end, nextStart));
            } else {
                cur.setWhitespace("");
            }
        }
        DocImpl doc = new DocImpl(text, tokens, vocab, language);
        for (TokenImpl t : tokens) {
            t.setDoc(doc);
        }
        return doc;
    }

    public Vocab vocab() {
        return vocab;
    }

    /** Tokenize to plain strings. */
    public List<String> tokenizeToStrings(String text) {
        Doc doc = tokenize(text);
        List<String> out = new ArrayList<>(doc.length());
        for (int i = 0; i < doc.length(); i++) {
            out.add(doc.getToken(i).getText());
        }
        return out;
    }
}
