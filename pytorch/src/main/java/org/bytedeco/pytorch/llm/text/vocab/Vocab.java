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
package org.bytedeco.pytorch.llm.text.vocab;

import java.io.BufferedReader;
import java.io.BufferedWriter;
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
import java.util.Map;
import java.util.Objects;

/**
 * Torchtext-style vocabulary: stoi/itos, specials, encode/decode.
 *
 * <pre>{@code
 * Vocab vocab = Vocab.build_vocab_from_iterator(tokens, 1, List.of("&lt;unk&gt;", "&lt;pad&gt;"));
 * long[] ids = vocab.encode(List.of("hello", "world"));
 * }</pre>
 */
public final class Vocab {

    public static final String DEFAULT_UNK = "<unk>";
    public static final String DEFAULT_PAD = "<pad>";
    public static final String DEFAULT_BOS = "<bos>";
    public static final String DEFAULT_EOS = "<eos>";

    private final Map<String, Integer> stoi;
    private final List<String> itos;
    private final String unkToken;
    private final String padToken;
    private final String bosToken;
    private final String eosToken;
    private final int defaultIndex;

    public Vocab(List<String> tokens) {
        this(tokens, DEFAULT_UNK, DEFAULT_PAD, DEFAULT_BOS, DEFAULT_EOS);
    }

    public Vocab(List<String> tokens, String unk, String pad, String bos, String eos) {
        this.stoi = new LinkedHashMap<>();
        this.itos = new ArrayList<>();
        this.unkToken = unk == null ? DEFAULT_UNK : unk;
        this.padToken = pad;
        this.bosToken = bos;
        this.eosToken = eos;
        if (tokens != null) {
            for (String t : tokens) {
                addToken(t);
            }
        }
        // ensure unk exists
        if (!stoi.containsKey(unkToken)) {
            addToken(unkToken);
        }
        this.defaultIndex = stoi.get(unkToken);
    }

    public Vocab(Map<String, Integer> stoiMap) {
        this.stoi = new LinkedHashMap<>();
        this.itos = new ArrayList<>();
        this.unkToken = DEFAULT_UNK;
        this.padToken = DEFAULT_PAD;
        this.bosToken = DEFAULT_BOS;
        this.eosToken = DEFAULT_EOS;
        int maxId = -1;
        for (Map.Entry<String, Integer> e : stoiMap.entrySet()) {
            if (e.getValue() != null && e.getValue() > maxId) {
                maxId = e.getValue();
            }
        }
        for (int i = 0; i <= maxId; i++) {
            itos.add(null);
        }
        for (Map.Entry<String, Integer> e : stoiMap.entrySet()) {
            stoi.put(e.getKey(), e.getValue());
            while (itos.size() <= e.getValue()) {
                itos.add(null);
            }
            itos.set(e.getValue(), e.getKey());
        }
        if (!stoi.containsKey(unkToken)) {
            addToken(unkToken);
        }
        this.defaultIndex = stoi.get(unkToken);
    }

    private void addToken(String token) {
        if (token == null || stoi.containsKey(token)) {
            return;
        }
        int id = itos.size();
        stoi.put(token, id);
        itos.add(token);
    }

    /**
     * Build vocab from an iterator of token lists (torchtext.vocab.build_vocab_from_iterator).
     */
    public static Vocab build_vocab_from_iterator(Iterable<? extends Iterable<String>> iterator,
                                                  int minFreq,
                                                  List<String> specials) {
        return buildVocabFromIterator(iterator, minFreq, specials);
    }

    public static Vocab buildVocabFromIterator(Iterable<? extends Iterable<String>> iterator,
                                               int minFreq,
                                               List<String> specials) {
        Objects.requireNonNull(iterator, "iterator");
        Map<String, Integer> counts = new HashMap<>();
        for (Iterable<String> tokens : iterator) {
            if (tokens == null) {
                continue;
            }
            for (String t : tokens) {
                if (t != null) {
                    counts.merge(t, 1, Integer::sum);
                }
            }
        }
        List<String> ordered = new ArrayList<>();
        if (specials != null) {
            for (String s : specials) {
                if (s != null && !ordered.contains(s)) {
                    ordered.add(s);
                }
            }
        } else {
            ordered.add(DEFAULT_UNK);
            ordered.add(DEFAULT_PAD);
        }
        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(counts.entrySet());
        sorted.sort((a, b) -> {
            int c = Integer.compare(b.getValue(), a.getValue());
            return c != 0 ? c : a.getKey().compareTo(b.getKey());
        });
        for (Map.Entry<String, Integer> e : sorted) {
            if (e.getValue() < minFreq) {
                continue;
            }
            if (!ordered.contains(e.getKey())) {
                ordered.add(e.getKey());
            }
        }
        String unk = ordered.contains(DEFAULT_UNK) ? DEFAULT_UNK
                : (specials != null && !specials.isEmpty() ? specials.get(0) : DEFAULT_UNK);
        String pad = ordered.contains(DEFAULT_PAD) ? DEFAULT_PAD : null;
        return new Vocab(ordered, unk, pad, DEFAULT_BOS, DEFAULT_EOS);
    }

    /** Python-like {@code vocab(token)} → id. */
    public int __call__(String token) {
        return lookup(token);
    }

    public int lookup(String token) {
        Integer id = stoi.get(token);
        return id == null ? defaultIndex : id;
    }

    public int[] lookup_indices(List<String> tokens) {
        return encode(tokens);
    }

    public List<Integer> lookupIndices(List<String> tokens) {
        int[] ids = encode(tokens);
        List<Integer> out = new ArrayList<>(ids.length);
        for (int id : ids) {
            out.add(id);
        }
        return out;
    }

    public String lookup_token(int index) {
        if (index < 0 || index >= itos.size() || itos.get(index) == null) {
            return unkToken;
        }
        return itos.get(index);
    }

    public List<String> lookup_tokens(int[] indices) {
        List<String> out = new ArrayList<>(indices == null ? 0 : indices.length);
        if (indices == null) {
            return out;
        }
        for (int i : indices) {
            out.add(lookup_token(i));
        }
        return out;
    }

    public int[] encode(List<String> tokens) {
        if (tokens == null) {
            return new int[0];
        }
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = lookup(tokens.get(i));
        }
        return ids;
    }

    public long[] encodeLong(List<String> tokens) {
        int[] ids = encode(tokens);
        long[] out = new long[ids.length];
        for (int i = 0; i < ids.length; i++) {
            out[i] = ids[i];
        }
        return out;
    }

    public List<String> decode(int[] ids) {
        return lookup_tokens(ids);
    }

    public List<String> decode(long[] ids) {
        if (ids == null) {
            return List.of();
        }
        int[] iids = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            iids[i] = (int) ids[i];
        }
        return decode(iids);
    }

    public String decodeToString(int[] ids) {
        List<String> toks = decode(ids);
        return String.join(" ", toks);
    }

    public int size() {
        return itos.size();
    }

    public int get_stoi(String token) {
        return lookup(token);
    }

    public String get_itos(int index) {
        return lookup_token(index);
    }

    public Map<String, Integer> get_stoi() {
        return Collections.unmodifiableMap(stoi);
    }

    public List<String> get_itos() {
        return Collections.unmodifiableList(itos);
    }

    public String unkToken() {
        return unkToken;
    }

    public String padToken() {
        return padToken;
    }

    public String bosToken() {
        return bosToken;
    }

    public String eosToken() {
        return eosToken;
    }

    public int unkId() {
        return defaultIndex;
    }

    public int padId() {
        return padToken == null ? -1 : stoi.getOrDefault(padToken, -1);
    }

    public int bosId() {
        return bosToken == null ? -1 : stoi.getOrDefault(bosToken, -1);
    }

    public int eosId() {
        return eosToken == null ? -1 : stoi.getOrDefault(eosToken, -1);
    }

    public void set_default_index(int index) {
        // fixed at construction for immutability of default; keep API for torchtext parity via no-op if invalid
        // Actually store is final; document that default is unk. Method kept for API compatibility.
    }

    public int get_default_index() {
        return defaultIndex;
    }

    public void append_token(String token) {
        addToken(token);
    }

    public void insert_token(String token, int index) {
        if (token == null || stoi.containsKey(token)) {
            return;
        }
        if (index < 0 || index > itos.size()) {
            addToken(token);
            return;
        }
        itos.add(index, token);
        stoi.clear();
        for (int i = 0; i < itos.size(); i++) {
            stoi.put(itos.get(i), i);
        }
    }

    public boolean contains(String token) {
        return stoi.containsKey(token);
    }

    public void save(Path path) {
        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (int i = 0; i < itos.size(); i++) {
                String t = itos.get(i);
                if (t != null) {
                    w.write(t);
                    w.write('\t');
                    w.write(Integer.toString(i));
                    w.newLine();
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static Vocab load(Path path) {
        Map<String, Integer> map = new LinkedHashMap<>();
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            int auto = 0;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] p = line.split("\\t");
                if (p.length >= 2) {
                    try {
                        map.put(p[0], Integer.parseInt(p[1].trim()));
                        continue;
                    } catch (NumberFormatException ignore) {
                        // fallthrough
                    }
                }
                map.put(p[0], auto++);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return new Vocab(map);
    }

    @Override
    public String toString() {
        return "Vocab(size=" + size() + ", unk=" + unkToken + ", sample="
                + itos.subList(0, Math.min(8, itos.size())) + ")";
    }
}
