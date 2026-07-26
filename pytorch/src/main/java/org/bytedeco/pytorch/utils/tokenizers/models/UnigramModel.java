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
package org.bytedeco.pytorch.utils.tokenizers.models;
import org.bytedeco.pytorch.jit.*;

import org.bytedeco.pytorch.utils.tokenizers.JsonMaps;
import org.bytedeco.pytorch.utils.tokenizers.pretokenizers.PreToken;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * HuggingFace Unigram model — Viterbi best segmentation over scored pieces.
 */
public final class UnigramModel implements Model {

    private final Map<String, Integer> vocab;       // token → id
    private final Map<Integer, String> idToToken;
    private final double[] scores;                  // id → score (log prob)
    private final int unkId;
    private final boolean byteFallback;
    private final boolean fuseUnk;
    private final double minScore;
    private final double unkPenalty;

    // Simple prefix trie for piece lookup
    private final TrieNode root;

    public UnigramModel(List<Piece> pieces, Integer unkId, boolean byteFallback) {
        this(pieces, unkId, byteFallback, true);
    }

    public UnigramModel(List<Piece> pieces, Integer unkId, boolean byteFallback, boolean fuseUnk) {
        Objects.requireNonNull(pieces, "pieces");
        this.vocab = new LinkedHashMap<>();
        this.idToToken = new HashMap<>();
        this.scores = new double[pieces.size()];
        double min = Double.POSITIVE_INFINITY;
        for (int i = 0; i < pieces.size(); i++) {
            Piece p = pieces.get(i);
            vocab.put(p.token, i);
            idToToken.put(i, p.token);
            scores[i] = p.score;
            if (p.score < min) min = p.score;
        }
        this.minScore = Double.isInfinite(min) ? 0.0 : min;
        this.unkId = unkId == null ? 0 : unkId;
        this.byteFallback = byteFallback;
        this.fuseUnk = fuseUnk;
        // HF uses unk_score = min_score - 10.0 roughly
        this.unkPenalty = this.minScore - 10.0;
        this.root = buildTrie(pieces);
    }

    public static UnigramModel fromJson(Map<String, Object> m) {
        List<Object> rawVocab = JsonMaps.asList(m.get("vocab"));
        List<Piece> pieces = new ArrayList<>();
        if (rawVocab != null) {
            for (Object item : rawVocab) {
                if (item instanceof List<?> pair && pair.size() >= 2) {
                    String tok = String.valueOf(pair.get(0));
                    Double score = JsonMaps.asDouble(pair.get(1));
                    pieces.add(new Piece(tok, score == null ? 0.0 : score));
                }
            }
        }
        Integer unkId = JsonMaps.asInt(m.get("unk_id"));
        boolean byteFallback = JsonMaps.asBoolean(m, "byte_fallback", false);
        // HF default fuse_unk is true for Unigram
        boolean fuseUnk = JsonMaps.asBoolean(m, "fuse_unk", true);
        return new UnigramModel(pieces, unkId, byteFallback, fuseUnk);
    }

    public record Piece(String token, double score) {}

    private static final class TrieNode {
        final Map<Character, TrieNode> children = new HashMap<>();
        int id = -1; // vocab id if a piece ends here
    }

    private static TrieNode buildTrie(List<Piece> pieces) {
        TrieNode root = new TrieNode();
        for (int i = 0; i < pieces.size(); i++) {
            String t = pieces.get(i).token;
            TrieNode cur = root;
            for (int j = 0; j < t.length(); j++) {
                char c = t.charAt(j);
                cur = cur.children.computeIfAbsent(c, k -> new TrieNode());
            }
            cur.id = i;
        }
        return root;
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
            List<Token> segs = encodeOne(word, p.start(), p.end());
            out.addAll(segs);
        }
        if (fuseUnk && out.size() > 1) {
            List<Token> fused = new ArrayList<>(out.size());
            for (Token t : out) {
                if (!fused.isEmpty()
                        && t.id() == unkId
                        && fused.get(fused.size() - 1).id() == unkId) {
                    continue;
                }
                fused.add(t);
            }
            return fused;
        }
        return out;
    }

    private List<Token> encodeOne(String text, int start, int end) {
        int n = text.length();
        if (n == 0) return List.of();

        // best[i] = best log-prob to reach character offset i
        double[] best = new double[n + 1];
        int[] backId = new int[n + 1];   // piece id used to arrive
        int[] backPos = new int[n + 1];  // previous offset
        java.util.Arrays.fill(best, Double.NEGATIVE_INFINITY);
        java.util.Arrays.fill(backId, -1);
        java.util.Arrays.fill(backPos, -1);
        best[0] = 0.0;

        for (int i = 0; i < n; i++) {
            if (Double.isInfinite(best[i]) && best[i] < 0) continue;
            // trie walk from i
            TrieNode node = root;
            boolean found = false;
            for (int j = i; j < n; j++) {
                char c = text.charAt(j);
                node = node.children.get(c);
                if (node == null) break;
                if (node.id >= 0) {
                    found = true;
                    double score = best[i] + scores[node.id];
                    if (score > best[j + 1]) {
                        best[j + 1] = score;
                        backId[j + 1] = node.id;
                        backPos[j + 1] = i;
                    }
                }
            }
            // unk single char fallback if nothing starts here... actually always allow unk of 1 char
            // HF: if no piece, use unk for one character (or byte fallback)
            double unkScore = best[i] + unkPenalty;
            int next = i + Character.charCount(text.codePointAt(i));
            if (next <= n && unkScore > best[next]) {
                // only use unk path if no better real piece ends at next? keep as option
                // Actually always record as fallback option
                if (!found || best[next] == Double.NEGATIVE_INFINITY) {
                    best[next] = unkScore;
                    backId[next] = -2; // unk marker
                    backPos[next] = i;
                } else if (unkScore > best[next]) {
                    // don't override better real pieces
                }
            }
        }

        // If end unreachable, force char-by-char unk
        if (Double.isInfinite(best[n]) && best[n] < 0) {
            return forceUnk(text, start, end);
        }

        // Backtrack
        List<Token> rev = new ArrayList<>();
        int pos = n;
        while (pos > 0) {
            int prev = backPos[pos];
            int id = backId[pos];
            if (prev < 0) {
                return forceUnk(text, start, end);
            }
            String piece = text.substring(prev, pos);
            if (id == -2) {
                if (byteFallback) {
                    rev.addAll(byteFallbackTokens(piece, start, end));
                } else {
                    rev.add(new Token(unkId, idToToken.getOrDefault(unkId, "<unk>"), start, end));
                }
            } else {
                String tok = idToToken.get(id);
                rev.add(new Token(id, tok == null ? piece : tok, start, end));
            }
            pos = prev;
        }
        Collections.reverse(rev);
        return rev;
    }

    private List<Token> forceUnk(String text, int start, int end) {
        List<Token> out = new ArrayList<>();
        if (byteFallback) {
            out.addAll(byteFallbackTokens(text, start, end));
        } else {
            out.add(new Token(unkId, idToToken.getOrDefault(unkId, "<unk>"), start, end));
        }
        return out;
    }

    private List<Token> byteFallbackTokens(String piece, int start, int end) {
        List<Token> out = new ArrayList<>();
        byte[] raw = piece.getBytes(StandardCharsets.UTF_8);
        for (byte b : raw) {
            String bt = String.format("<0x%02X>", b & 0xff);
            Integer id = vocab.get(bt);
            if (id != null) {
                out.add(new Token(id, bt, start, end));
            } else {
                out.add(new Token(unkId, idToToken.getOrDefault(unkId, "<unk>"), start, end));
            }
        }
        return out;
    }

    @Override
    public Map<String, Integer> getVocab() {
        return Collections.unmodifiableMap(vocab);
    }

    @Override
    public String idToToken(int id) {
        return idToToken.get(id);
    }

    @Override
    public int tokenToId(String token) {
        Integer id = vocab.get(token);
        return id == null ? -1 : id;
    }
}
