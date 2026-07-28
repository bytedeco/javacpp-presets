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

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Simplified GPT-2 style byte-level BPE tokenizer.
 * Uses a GPT-2-like pre-tokenization regex, then applies BPE merges (or falls back to bytes).
 */
public final class GPT2BPETokenizer implements Tokenizer {

    // Approximate GPT-2 pre-tokenizer pattern
    private static final Pattern GPT2_PATTERN = Pattern.compile(
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
    );

    private final Map<String, Integer> encoder; // token -> id
    private final Map<Integer, String> decoder;
    private final List<String> merges;
    private final Map<String, Integer> bpeRanks;
    private final String unkToken;

    public GPT2BPETokenizer(Map<String, Integer> encoder, List<String> merges) {
        this.encoder = new LinkedHashMap<>(Objects.requireNonNull(encoder, "encoder"));
        this.decoder = new HashMap<>();
        for (Map.Entry<String, Integer> e : this.encoder.entrySet()) {
            this.decoder.put(e.getValue(), e.getKey());
        }
        this.merges = merges == null ? List.of() : new ArrayList<>(merges);
        this.bpeRanks = new HashMap<>();
        for (int i = 0; i < this.merges.size(); i++) {
            bpeRanks.put(this.merges.get(i), i);
        }
        this.unkToken = "<|endoftext|>";
        this.encoder.putIfAbsent(unkToken, this.encoder.size());
        this.decoder.putIfAbsent(this.encoder.get(unkToken), unkToken);
    }

    /** Build a minimal byte-level vocab (0-255 printable mapping) with optional merges. */
    public static GPT2BPETokenizer byteLevel(List<String> merges) {
        Map<String, Integer> enc = new LinkedHashMap<>();
        // GPT-2 style: map bytes to unicode-ish printable tokens; simplified as "Ġ" etc not needed —
        // use direct latin1 single-char strings for bytes 0-255 via char cast of mapped printable.
        for (int i = 0; i < 256; i++) {
            enc.put(byteToken(i), i);
        }
        enc.put("<|endoftext|>", 256);
        return new GPT2BPETokenizer(enc, merges == null ? List.of() : merges);
    }

    public static GPT2BPETokenizer fromMergesFile(Path mergesFile, Map<String, Integer> encoder) {
        BPETokenizer bpe = BPETokenizer.fromMergesFile(mergesFile, encoder);
        return fromMerges(bpe);
    }

    private static GPT2BPETokenizer fromMerges(BPETokenizer bpe) {
        Map<String, Integer> enc = new LinkedHashMap<>(bpe.vocab());
        for (int i = 0; i < 256; i++) {
            enc.putIfAbsent(byteToken(i), enc.size());
        }
        enc.putIfAbsent("<|endoftext|>", enc.size());
        return new GPT2BPETokenizer(enc, new ArrayList<>(bpe.merges()));
    }

    /** Learn small BPE on corpus, then wrap as GPT2-style. */
    public static GPT2BPETokenizer learn(Iterable<String> corpus, int numMerges) {
        BPETokenizer bpe = BPETokenizer.learn(corpus, numMerges);
        Map<String, Integer> enc = new LinkedHashMap<>(bpe.vocab());
        for (int i = 0; i < 256; i++) {
            enc.putIfAbsent(byteToken(i), enc.size());
        }
        enc.putIfAbsent("<|endoftext|>", enc.size());
        return new GPT2BPETokenizer(enc, new ArrayList<>(bpe.merges()));
    }

    private static String byteToken(int b) {
        // printable-ish single code unit representation of a byte
        return String.valueOf((char) (b & 0xff));
    }

    private static List<String> bytesToTokens(String piece) {
        byte[] bytes = piece.getBytes(StandardCharsets.UTF_8);
        List<String> tokens = new ArrayList<>(bytes.length);
        for (byte value : bytes) {
            tokens.add(byteToken(value & 0xff));
        }
        return tokens;
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return out;
        }
        Matcher m = GPT2_PATTERN.matcher(text);
        while (m.find()) {
            String piece = m.group();
            out.addAll(bpe(bytesToTokens(piece)));
        }
        return out;
    }

    private List<String> bpe(List<String> symbols) {
        if (symbols.isEmpty() || bpeRanks.isEmpty()) {
            return symbols;
        }
        List<String> word = new ArrayList<>(symbols);
        while (word.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i + 1 < word.size(); i++) {
                String pair = word.get(i) + " " + word.get(i + 1);
                Integer rank = bpeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) {
                break;
            }
            String merged = word.get(bestIdx) + word.get(bestIdx + 1);
            List<String> next = new ArrayList<>(word.size() - 1);
            for (int i = 0; i < word.size(); ) {
                if (i == bestIdx) {
                    next.add(merged);
                    i += 2;
                } else {
                    next.add(word.get(i));
                    i++;
                }
            }
            word = next;
        }
        return word;
    }

    @Override
    public int[] encode(String text) {
        return encodeTokens(tokenize(text));
    }

    @Override
    public int[] encodeTokens(List<String> tokens) {
        if (tokens == null) {
            return new int[0];
        }
        int unk = encoder.getOrDefault(unkToken, 0);
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = encoder.getOrDefault(tokens.get(i), unk);
        }
        return ids;
    }

    @Override
    public String decode(int[] ids) {
        if (ids == null || ids.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String t = decoder.get(id);
            if (t == null) {
                continue;
            }
            if (unkToken.equals(t)) {
                continue;
            }
            // reverse byte tokens
            for (int i = 0; i < t.length(); i++) {
                sb.append(t.charAt(i));
            }
        }
        // interpret as latin1/utf-8 bytes
        byte[] raw = new byte[sb.length()];
        for (int i = 0; i < sb.length(); i++) {
            raw[i] = (byte) (sb.charAt(i) & 0xff);
        }
        return new String(raw, StandardCharsets.UTF_8);
    }

    public Map<String, Integer> encoder() {
        return Collections.unmodifiableMap(encoder);
    }

    public int vocabSize() {
        return encoder.size();
    }
}
