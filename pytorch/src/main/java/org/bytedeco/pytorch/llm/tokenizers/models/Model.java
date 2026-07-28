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
 * HuggingFace model stage: pre-tokens → token ids.
 */
public interface Model {

    List<Token> tokenize(List<PreToken> pretokens);

    Map<String, Integer> getVocab();

    default int tokenToId(String token) {
        Integer id = getVocab().get(token);
        return id == null ? -1 : id;
    }

    default String idToToken(int id) {
        for (Map.Entry<String, Integer> e : getVocab().entrySet()) {
            if (e.getValue() == id) return e.getKey();
        }
        return null;
    }

    default int vocabSize() {
        int max = -1;
        for (Integer v : getVocab().values()) {
            if (v != null && v > max) max = v;
        }
        return max + 1;
    }

    static Model fromJson(Map<String, Object> m) {
        if (m == null) throw new IllegalArgumentException("model is null");
        String type = JsonMaps.asString(m.get("type"));
        if (type == null || type.isEmpty()) {
            type = inferType(m);
        }
        return switch (type) {
            case "BPE" -> BpeModel.fromJson(m);
            case "WordPiece" -> WordPieceModel.fromJson(m);
            case "Unigram" -> UnigramModel.fromJson(m);
            case "WordLevel" -> WordLevelModel.fromJson(m);
            default -> throw new IllegalArgumentException("Unsupported model type: " + type);
        };
    }

    /**
     * Older HF tokenizer.json often omits {@code model.type}. Infer from shape:
     * <ul>
     *   <li>vocab as list of [token, score] → Unigram</li>
     *   <li>unk_id present, no merges → Unigram</li>
     *   <li>continuing_subword_prefix / max_input_chars_per_word, no merges → WordPiece</li>
     *   <li>merges present → BPE</li>
     *   <li>else BPE</li>
     * </ul>
     */
    static String inferType(Map<String, Object> m) {
        Object vocab = m.get("vocab");
        if (vocab instanceof List<?>) return "Unigram";
        if (m.get("unk_id") != null && m.get("merges") == null) return "Unigram";
        if (m.get("merges") != null) return "BPE";
        if (m.get("continuing_subword_prefix") != null
                || m.get("max_input_chars_per_word") != null) {
            return "WordPiece";
        }
        // WordPiece always has unk_token string + dict vocab without merges
        if (m.get("unk_token") != null && m.get("merges") == null && vocab instanceof Map<?, ?>) {
            // could be WordPiece or WordLevel; WordPiece is far more common in HF dumps
            return "WordPiece";
        }
        return "BPE";
    }

    // ---- WordLevel (trivial) ------------------------------------------------

    final class WordLevelModel implements Model {
        private final Map<String, Integer> vocab;
        private final Map<Integer, String> idToToken;
        private final String unkToken;
        private final int unkId;

        public WordLevelModel(Map<String, Integer> vocab, String unkToken) {
            this.vocab = Collections.unmodifiableMap(new LinkedHashMap<>(Objects.requireNonNull(vocab)));
            this.idToToken = new HashMap<>();
            for (Map.Entry<String, Integer> e : this.vocab.entrySet()) {
                idToToken.put(e.getValue(), e.getKey());
            }
            this.unkToken = unkToken == null ? "[UNK]" : unkToken;
            this.unkId = this.vocab.getOrDefault(this.unkToken, 0);
        }

        static WordLevelModel fromJson(Map<String, Object> m) {
            Map<String, Integer> vocab = JsonMaps.asStringIntMap(m.get("vocab"));
            String unk = JsonMaps.asString(m.get("unk_token"));
            return new WordLevelModel(vocab, unk);
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
                Integer id = vocab.get(p.value());
                if (id == null) {
                    out.add(new Token(unkId, unkToken, p.start(), p.end()));
                } else {
                    out.add(new Token(id, p.value(), p.start(), p.end()));
                }
            }
            return out;
        }

        @Override
        public Map<String, Integer> getVocab() { return vocab; }

        @Override
        public String idToToken(int id) { return idToToken.get(id); }
    }
}
