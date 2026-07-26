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
package org.bytedeco.pytorch.utils.text.tokenizer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Simplified BPE tokenizer. Loads merges from a file or learns a small BPE from a corpus.
 */
public final class BPETokenizer implements Tokenizer {

    private static final Pattern WORD = Pattern.compile("\\S+");

    private final Map<String, Integer> vocab;
    private final List<String> merges; // "a b" ordered by priority
    private final Map<String, Integer> mergeRank;
    private final String unkToken;
    private final boolean lower;

    public BPETokenizer(Map<String, Integer> vocab, List<String> merges) {
        this(vocab, merges, "<unk>", true);
    }

    public BPETokenizer(Map<String, Integer> vocab, List<String> merges, String unkToken, boolean lower) {
        this.vocab = new LinkedHashMap<>(Objects.requireNonNull(vocab, "vocab"));
        this.merges = merges == null ? List.of() : new ArrayList<>(merges);
        this.mergeRank = new HashMap<>();
        for (int i = 0; i < this.merges.size(); i++) {
            mergeRank.put(this.merges.get(i), i);
        }
        this.unkToken = unkToken == null ? "<unk>" : unkToken;
        this.lower = lower;
        this.vocab.putIfAbsent(this.unkToken, this.vocab.size());
    }

    /** Load merges file (tokenizers/sentencepiece style: one "a b" pair per line). */
    public static BPETokenizer fromMergesFile(Path mergesFile, Map<String, Integer> vocab) {
        List<String> merges = new ArrayList<>();
        try (BufferedReader br = Files.newBufferedReader(mergesFile, StandardCharsets.UTF_8)) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#") || line.startsWith("version")) {
                    continue;
                }
                merges.add(line);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (vocab == null) {
            vocab = new LinkedHashMap<>();
            vocab.put("<unk>", 0);
            vocab.put("<pad>", 1);
            for (String m : merges) {
                String[] p = m.split("\\s+");
                if (p.length >= 2) {
                    vocab.putIfAbsent(p[0], vocab.size());
                    vocab.putIfAbsent(p[1], vocab.size());
                    vocab.putIfAbsent(p[0] + p[1], vocab.size());
                }
            }
        }
        return new BPETokenizer(vocab, merges);
    }

    /**
     * Learn a simple BPE from a small corpus of whitespace-tokenized words.
     *
     * @param corpus words / sentences
     * @param numMerges number of merge operations
     */
    public static BPETokenizer learn(Iterable<String> corpus, int numMerges) {
        Map<String, Integer> wordFreq = new HashMap<>();
        for (String line : corpus) {
            if (line == null) {
                continue;
            }
            for (String w : line.toLowerCase(Locale.ROOT).split("\\s+")) {
                if (!w.isEmpty()) {
                    wordFreq.merge(w, 1, Integer::sum);
                }
            }
        }
        // symbol sequences for each word
        Map<String, List<String>> splits = new HashMap<>();
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("<unk>", 0);
        vocab.put("<pad>", 1);
        vocab.put("</w>", 2);
        for (String w : wordFreq.keySet()) {
            List<String> chars = new ArrayList<>();
            for (int i = 0; i < w.length(); i++) {
                String ch = String.valueOf(w.charAt(i));
                chars.add(ch);
                vocab.putIfAbsent(ch, vocab.size());
            }
            chars.add("</w>");
            splits.put(w, chars);
        }

        List<String> merges = new ArrayList<>();
        for (int step = 0; step < numMerges; step++) {
            Map<String, Integer> pairCounts = new HashMap<>();
            for (Map.Entry<String, List<String>> e : splits.entrySet()) {
                List<String> seq = e.getValue();
                int freq = wordFreq.get(e.getKey());
                for (int i = 0; i + 1 < seq.size(); i++) {
                    String pair = seq.get(i) + " " + seq.get(i + 1);
                    pairCounts.merge(pair, freq, Integer::sum);
                }
            }
            if (pairCounts.isEmpty()) {
                break;
            }
            String best = null;
            int bestC = -1;
            for (Map.Entry<String, Integer> e : pairCounts.entrySet()) {
                if (e.getValue() > bestC || (e.getValue() == bestC && (best == null || e.getKey().compareTo(best) < 0))) {
                    best = e.getKey();
                    bestC = e.getValue();
                }
            }
            if (best == null || bestC < 1) {
                break;
            }
            merges.add(best);
            String[] parts = best.split(" ", 2);
            String a = parts[0];
            String b = parts[1];
            String merged = a + b;
            vocab.putIfAbsent(merged, vocab.size());
            for (Map.Entry<String, List<String>> e : splits.entrySet()) {
                e.setValue(mergePair(e.getValue(), a, b, merged));
            }
        }
        return new BPETokenizer(vocab, merges);
    }

    private static List<String> mergePair(List<String> seq, String a, String b, String merged) {
        List<String> out = new ArrayList<>(seq.size());
        int i = 0;
        while (i < seq.size()) {
            if (i + 1 < seq.size() && seq.get(i).equals(a) && seq.get(i + 1).equals(b)) {
                out.add(merged);
                i += 2;
            } else {
                out.add(seq.get(i));
                i++;
            }
        }
        return out;
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return out;
        }
        String src = lower ? text.toLowerCase(Locale.ROOT) : text;
        Matcher m = WORD.matcher(src);
        while (m.find()) {
            out.addAll(bpe(m.group()));
        }
        return out;
    }

    private List<String> bpe(String word) {
        if (word.isEmpty()) {
            return List.of();
        }
        List<String> symbols = new ArrayList<>();
        for (int i = 0; i < word.length(); i++) {
            symbols.add(String.valueOf(word.charAt(i)));
        }
        symbols.add("</w>");
        if (mergeRank.isEmpty()) {
            // no merges: return characters (drop end marker as separate if not in vocab)
            List<String> simple = new ArrayList<>();
            for (int i = 0; i < word.length(); i++) {
                simple.add(String.valueOf(word.charAt(i)));
            }
            return simple;
        }
        while (symbols.size() > 1) {
            int bestRank = Integer.MAX_VALUE;
            int bestIdx = -1;
            for (int i = 0; i + 1 < symbols.size(); i++) {
                String pair = symbols.get(i) + " " + symbols.get(i + 1);
                Integer rank = mergeRank.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }
            if (bestIdx < 0) {
                break;
            }
            String merged = symbols.get(bestIdx) + symbols.get(bestIdx + 1);
            List<String> next = new ArrayList<>(symbols.size() - 1);
            for (int i = 0; i < symbols.size(); ) {
                if (i == bestIdx) {
                    next.add(merged);
                    i += 2;
                } else {
                    next.add(symbols.get(i));
                    i++;
                }
            }
            symbols = next;
        }
        // strip end-of-word marker for presentation
        List<String> cleaned = new ArrayList<>();
        for (String s : symbols) {
            if ("</w>".equals(s)) {
                continue;
            }
            if (s.endsWith("</w>")) {
                cleaned.add(s.substring(0, s.length() - 4));
            } else {
                cleaned.add(s);
            }
        }
        return cleaned.isEmpty() ? List.of(unkToken) : cleaned;
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
        int unk = vocab.getOrDefault(unkToken, 0);
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = vocab.getOrDefault(tokens.get(i), unk);
        }
        return ids;
    }

    @Override
    public String decode(int[] ids) {
        if (ids == null || ids.length == 0) {
            return "";
        }
        Map<Integer, String> inv = new HashMap<>();
        for (Map.Entry<String, Integer> e : vocab.entrySet()) {
            inv.putIfAbsent(e.getValue(), e.getKey());
        }
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String t = inv.getOrDefault(id, unkToken);
            if (t.endsWith("</w>")) {
                sb.append(t, 0, t.length() - 4).append(' ');
            } else {
                sb.append(t);
            }
        }
        return sb.toString().trim();
    }

    public Map<String, Integer> vocab() {
        return Collections.unmodifiableMap(vocab);
    }

    public List<String> merges() {
        return Collections.unmodifiableList(merges);
    }
}
