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
 * WordPiece tokenizer (BERT-style greedy longest-match).
 * Loads a vocab file of {@code token\\t?id?} lines, or builds from a training corpus.
 */
public final class WordPieceTokenizer implements Tokenizer {

    public static final String DEFAULT_UNK = "[UNK]";
    public static final String DEFAULT_CLS = "[CLS]";
    public static final String DEFAULT_SEP = "[SEP]";
    public static final String DEFAULT_PAD = "[PAD]";
    public static final String DEFAULT_MASK = "[MASK]";
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]+|[^\\p{L}\\p{N}\\s]+");

    private final Map<String, Integer> tokenToId;
    private final List<String> idToToken;
    private final String unkToken;
    private final String continuingPrefix;
    private final int maxInputCharsPerWord;
    private final boolean doLowerCase;

    public WordPieceTokenizer(Map<String, Integer> vocab) {
        this(vocab, DEFAULT_UNK, "##", 100, true);
    }

    public WordPieceTokenizer(Map<String, Integer> vocab, String unkToken,
                              String continuingPrefix, int maxInputCharsPerWord, boolean doLowerCase) {
        this.tokenToId = new LinkedHashMap<>(Objects.requireNonNull(vocab, "vocab"));
        this.idToToken = new ArrayList<>(Collections.nCopies(Math.max(1, maxId(vocab) + 1), null));
        for (Map.Entry<String, Integer> e : this.tokenToId.entrySet()) {
            int id = e.getValue();
            while (idToToken.size() <= id) {
                idToToken.add(null);
            }
            idToToken.set(id, e.getKey());
        }
        this.unkToken = unkToken == null ? DEFAULT_UNK : unkToken;
        this.continuingPrefix = continuingPrefix == null ? "##" : continuingPrefix;
        this.maxInputCharsPerWord = Math.max(1, maxInputCharsPerWord);
        this.doLowerCase = doLowerCase;
        this.tokenToId.putIfAbsent(this.unkToken, this.tokenToId.size());
        ensureId(this.unkToken);
    }

    private void ensureId(String token) {
        Integer id = tokenToId.get(token);
        if (id == null) {
            id = tokenToId.size();
            tokenToId.put(token, id);
        }
        while (idToToken.size() <= id) {
            idToToken.add(null);
        }
        idToToken.set(id, token);
    }

    private static int maxId(Map<String, Integer> vocab) {
        int m = -1;
        for (Integer v : vocab.values()) {
            if (v != null && v > m) {
                m = v;
            }
        }
        return m;
    }

    /** Load vocab from a text file: one token per line, optional {@code token id}. */
    public static WordPieceTokenizer fromFile(Path vocabFile) {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        try (BufferedReader br = Files.newBufferedReader(vocabFile, StandardCharsets.UTF_8)) {
            String line;
            int autoId = 0;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] parts = line.split("\\s+");
                if (parts.length >= 2) {
                    try {
                        int id = Integer.parseInt(parts[parts.length - 1]);
                        String tok = line.substring(0, line.length() - parts[parts.length - 1].length()).trim();
                        vocab.put(tok, id);
                        continue;
                    } catch (NumberFormatException ignore) {
                        // fall through
                    }
                }
                vocab.put(parts[0], autoId++);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return new WordPieceTokenizer(vocab);
    }

    /**
     * Build a simple WordPiece-like vocab from a training corpus of tokenized words.
     * Character pieces and whole words above {@code minFreq} are kept.
     */
    public static WordPieceTokenizer buildFromCorpus(Iterable<? extends Iterable<String>> corpus,
                                                     int minFreq, int maxVocabSize) {
        Map<String, Integer> counts = new HashMap<>();
        for (Iterable<String> sent : corpus) {
            for (String w : sent) {
                if (w == null || w.isEmpty()) {
                    continue;
                }
                String word = w.toLowerCase(Locale.ROOT);
                counts.merge(word, 1, Integer::sum);
                for (int i = 0; i < word.length(); i++) {
                    String ch = String.valueOf(word.charAt(i));
                    counts.merge(ch, 1, Integer::sum);
                    if (i > 0) {
                        counts.merge("##" + ch, 1, Integer::sum);
                    }
                }
                for (int i = 1; i < word.length(); i++) {
                    counts.merge("##" + word.substring(i), 1, Integer::sum);
                }
            }
        }
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put(DEFAULT_PAD, 0);
        vocab.put(DEFAULT_UNK, 1);
        vocab.put(DEFAULT_CLS, 2);
        vocab.put(DEFAULT_SEP, 3);
        vocab.put(DEFAULT_MASK, 4);
        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(counts.entrySet());
        sorted.sort((a, b) -> {
            int c = Integer.compare(b.getValue(), a.getValue());
            return c != 0 ? c : a.getKey().compareTo(b.getKey());
        });
        for (Map.Entry<String, Integer> e : sorted) {
            if (e.getValue() < minFreq) {
                continue;
            }
            if (vocab.size() >= maxVocabSize) {
                break;
            }
            vocab.putIfAbsent(e.getKey(), vocab.size());
        }
        return new WordPieceTokenizer(vocab);
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return out;
        }
        String src = doLowerCase ? text.toLowerCase(Locale.ROOT) : text;
        Matcher m = WORD.matcher(src);
        while (m.find()) {
            out.addAll(wordPiece(m.group()));
        }
        return out;
    }

    private List<String> wordPiece(String token) {
        List<String> output = new ArrayList<>();
        if (token.length() > maxInputCharsPerWord) {
            output.add(unkToken);
            return output;
        }
        int start = 0;
        boolean isBad = false;
        List<String> subTokens = new ArrayList<>();
        while (start < token.length()) {
            int end = token.length();
            String cur = null;
            while (start < end) {
                String substr = token.substring(start, end);
                if (start > 0) {
                    substr = continuingPrefix + substr;
                }
                if (tokenToId.containsKey(substr)) {
                    cur = substr;
                    break;
                }
                end--;
            }
            if (cur == null) {
                isBad = true;
                break;
            }
            subTokens.add(cur);
            start = end;
        }
        if (isBad) {
            output.add(unkToken);
        } else {
            output.addAll(subTokens);
        }
        return output;
    }

    @Override
    public int[] encode(String text) {
        List<String> tokens = tokenize(text);
        return encodeTokens(tokens);
    }

    @Override
    public int[] encodeTokens(List<String> tokens) {
        if (tokens == null) {
            return new int[0];
        }
        int unk = tokenToId.getOrDefault(unkToken, 0);
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = tokenToId.getOrDefault(tokens.get(i), unk);
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
            String t = id >= 0 && id < idToToken.size() ? idToToken.get(id) : unkToken;
            if (t == null) {
                t = unkToken;
            }
            if (t.startsWith(continuingPrefix)) {
                sb.append(t.substring(continuingPrefix.length()));
            } else {
                if (sb.length() > 0) {
                    sb.append(' ');
                }
                sb.append(t);
            }
        }
        return sb.toString();
    }

    public Map<String, Integer> vocab() {
        return Collections.unmodifiableMap(tokenToId);
    }

    public int vocabSize() {
        return tokenToId.size();
    }

    public int tokenToId(String token) {
        return tokenToId.getOrDefault(token, tokenToId.getOrDefault(unkToken, 0));
    }

    public String idToToken(int id) {
        if (id < 0 || id >= idToToken.size() || idToToken.get(id) == null) {
            return unkToken;
        }
        return idToToken.get(id);
    }
}
