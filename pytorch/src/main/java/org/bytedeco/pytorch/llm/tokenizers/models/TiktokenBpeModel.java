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

import org.bytedeco.pytorch.llm.tokenizers.pretokenizers.PreToken;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Tiktoken-style BPE: merge by lowest rank of concatenated pieces (no explicit merges list).
 * Vocab keys are GPT-2 {@code bytes_to_unicode} strings (one char per byte).
 * Used by ChatGLM4 {@code tokenizer.model} text dumps and similar rank tables.
 */
public final class TiktokenBpeModel implements Model {

    private final Map<String, Integer> vocab; // piece -> rank/id
    private final Map<Integer, String> idToToken;

    public TiktokenBpeModel(Map<String, Integer> vocab) {
        this.vocab = Collections.unmodifiableMap(new LinkedHashMap<>(Objects.requireNonNull(vocab)));
        this.idToToken = new HashMap<>(this.vocab.size() * 2);
        for (Map.Entry<String, Integer> e : this.vocab.entrySet()) {
            idToToken.put(e.getValue(), e.getKey());
        }
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
            if (word == null || word.isEmpty()) continue;
            for (String piece : bpe(word)) {
                Integer id = vocab.get(piece);
                if (id != null) {
                    out.add(new Token(id, piece, p.start(), p.end()));
                }
                // drop unknown pieces (tiktoken typically has full byte coverage)
            }
        }
        return out;
    }

    /**
     * Tiktoken byte-pair merge over a single pretok already in bytes_to_unicode form.
     * Initial symbols = individual chars (each representing one byte).
     */
    public List<String> bpe(String token) {
        if (token == null || token.isEmpty()) return List.of();
        // Fast path: whole token in vocab
        if (vocab.containsKey(token)) {
            return List.of(token);
        }
        List<String> parts = new ArrayList<>(token.length());
        for (int i = 0; i < token.length(); ) {
            int cp = token.codePointAt(i);
            int n = Character.charCount(cp);
            parts.add(token.substring(i, i + n));
            i += n;
        }
        while (parts.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i + 1 < parts.size(); i++) {
                String merged = parts.get(i) + parts.get(i + 1);
                Integer rank = vocab.get(merged);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) break;
            String merged = parts.get(bestIdx) + parts.get(bestIdx + 1);
            List<String> next = new ArrayList<>(parts.size() - 1);
            int i = 0;
            while (i < parts.size()) {
                if (i == bestIdx) {
                    next.add(merged);
                    i += 2;
                } else {
                    next.add(parts.get(i));
                    i++;
                }
            }
            parts = next;
        }
        return parts;
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
