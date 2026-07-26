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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Character-level BPE tokenizer: trains/applies BPE over characters of whole text
 * (no whitespace pre-tokenization). Useful for CJK or character-centric models.
 */
public final class CharBPETokenizer implements Tokenizer {

    private final BPETokenizer delegate;
    private final boolean lower;

    public CharBPETokenizer(BPETokenizer delegate) {
        this(delegate, false);
    }

    public CharBPETokenizer(BPETokenizer delegate, boolean lower) {
        this.delegate = delegate == null ? BPETokenizer.learn(List.of(), 0) : delegate;
        this.lower = lower;
    }

    public static CharBPETokenizer learn(Iterable<String> corpus, int numMerges) {
        // Feed character-spaced "words" so classic BPE learns on chars
        List<String> charLines = new ArrayList<>();
        for (String line : corpus) {
            if (line == null) {
                continue;
            }
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < line.length(); i++) {
                if (i > 0) {
                    sb.append(' ');
                }
                sb.append(line.charAt(i));
            }
            charLines.add(sb.toString());
        }
        return new CharBPETokenizer(BPETokenizer.learn(charLines, numMerges), false);
    }

    public static CharBPETokenizer fromVocab(Map<String, Integer> vocab, List<String> merges) {
        return new CharBPETokenizer(new BPETokenizer(vocab, merges, "<unk>", false), false);
    }

    public static CharBPETokenizer empty() {
        Map<String, Integer> vocab = new LinkedHashMap<>();
        vocab.put("<unk>", 0);
        vocab.put("<pad>", 1);
        return new CharBPETokenizer(new BPETokenizer(vocab, List.of()), false);
    }

    @Override
    public List<String> tokenize(String text) {
        if (text == null || text.isEmpty()) {
            return new ArrayList<>();
        }
        String src = lower ? text.toLowerCase(Locale.ROOT) : text;
        // Space-separate characters so underlying BPE word splitter sees each char
        StringBuilder spaced = new StringBuilder(src.length() * 2);
        for (int i = 0; i < src.length(); i++) {
            if (i > 0) {
                spaced.append(' ');
            }
            spaced.append(src.charAt(i));
        }
        return delegate.tokenize(spaced.toString());
    }

    @Override
    public int[] encode(String text) {
        return delegate.encodeTokens(tokenize(text));
    }

    @Override
    public int[] encodeTokens(List<String> tokens) {
        return delegate.encodeTokens(tokens);
    }

    @Override
    public String decode(int[] ids) {
        // Join without spaces for character model
        String decoded = delegate.decode(ids);
        return decoded == null ? "" : decoded.replace(" ", "");
    }

    public BPETokenizer delegate() {
        return delegate;
    }
}
