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
package org.bytedeco.pytorch.llm.tokenizers.models;

import org.bytedeco.pytorch.llm.tokenizers.JsonMaps;
import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreToken;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace WordPiece model — greedy longest-match (BERT-style).
 */
public final class WordPieceModel implements Model {

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> idToToken;
    private final String unkToken;
    private final int unkId;
    private final String continuingSubwordPrefix;
    private final int maxInputCharsPerWord;

    public WordPieceModel(Map<String, Integer> vocab, String unkToken,
                          String continuingSubwordPrefix, int maxInputCharsPerWord) {
        this.vocab = Collections.unmodifiableMap(new LinkedHashMap<>(Objects.requireNonNull(vocab)));
        this.idToToken = new HashMap<>();
        for (Map.Entry<String, Integer> e : this.vocab.entrySet()) {
            idToToken.put(e.getValue(), e.getKey());
        }
        this.unkToken = unkToken == null ? "[UNK]" : unkToken;
        this.unkId = this.vocab.getOrDefault(this.unkToken, 0);
        this.continuingSubwordPrefix = continuingSubwordPrefix == null ? "##" : continuingSubwordPrefix;
        this.maxInputCharsPerWord = maxInputCharsPerWord <= 0 ? 100 : maxInputCharsPerWord;
    }

    public static WordPieceModel fromJson(Map<String, Object> m) {
        Map<String, Integer> vocab = JsonMaps.asStringIntMap(m.get("vocab"));
        String unk = JsonMaps.asString(m.get("unk_token"));
        String cont = JsonMaps.asString(m.get("continuing_subword_prefix"));
        Integer maxChars = JsonMaps.asInt(m.get("max_input_chars_per_word"));
        return new WordPieceModel(vocab, unk, cont, maxChars == null ? 100 : maxChars);
    }

    @Override
    public List<Token> tokenize(List<PreToken> pretokens) {
        List<Token> out = new ArrayList<>();
        if (pretokens == null) return out;
        for (PreToken p : pretokens) {
            if (p.added()) {
                out.add(new Token(p.addedId(), p.value(), p.start(), p.end(), true));
                continue;
            }
            String word = p.value();
            if (word.isEmpty()) continue;
            if (word.length() > maxInputCharsPerWord) {
                out.add(new Token(unkId, unkToken, p.start(), p.end()));
                continue;
            }
            List<String> sub = wordPiece(word);
            if (sub == null) {
                out.add(new Token(unkId, unkToken, p.start(), p.end()));
            } else {
                for (String s : sub) {
                    Integer id = vocab.get(s);
                    out.add(new Token(id == null ? unkId : id, s, p.start(), p.end()));
                }
            }
        }
        return out;
    }

    /** @return null if the word cannot be segmented (caller emits UNK). */
    public List<String> wordPiece(String word) {
        List<String> output = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String cur = null;
            while (start < end) {
                String substr = word.substring(start, end);
                if (start > 0) substr = continuingSubwordPrefix + substr;
                if (vocab.containsKey(substr)) {
                    cur = substr;
                    break;
                }
                end--;
            }
            if (cur == null) return null;
            output.add(cur);
            start = end;
        }
        return output;
    }

    @Override
    public Map<String, Integer> getVocab() { return vocab; }

    @Override
    public String idToToken(int id) { return idToToken.get(id); }

    @Override
    public int tokenToId(String token) {
        Integer id = vocab.get(token);
        return id == null ? -1 : id;
    }
}
