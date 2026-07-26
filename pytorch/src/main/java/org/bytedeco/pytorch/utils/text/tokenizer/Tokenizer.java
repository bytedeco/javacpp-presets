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
import java.util.List;

/**
 * Torchtext-style tokenizer interface.
 *
 * <pre>{@code
 * Tokenizer tok = new BasicEnglishTokenizer();
 * List&lt;String&gt; tokens = tok.tokenize("Hello, world!");
 * }</pre>
 */
public interface Tokenizer {

    /** Split {@code text} into tokens. */
    List<String> tokenize(String text);

    /**
     * Encode tokens (or raw text via {@link #tokenize}) into integer ids.
     * Default implementation tokenizes then maps with a simple hash (subclasses override).
     */
    default int[] encode(String text) {
        List<String> tokens = tokenize(text);
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = tokens.get(i).hashCode() & 0x7fffffff;
        }
        return ids;
    }

    /**
     * Decode integer ids back to a string. Default joins string forms of ids.
     */
    default String decode(int[] ids) {
        if (ids == null || ids.length == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < ids.length; i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(ids[i]);
        }
        return sb.toString();
    }

    /** Encode a list of tokens (already tokenized) into ids. */
    default int[] encodeTokens(List<String> tokens) {
        if (tokens == null) {
            return new int[0];
        }
        int[] ids = new int[tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            ids[i] = tokens.get(i).hashCode() & 0x7fffffff;
        }
        return ids;
    }

    /** Decode a list of tokens back to a space-joined string. */
    default String detokenize(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return "";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < tokens.size(); i++) {
            if (i > 0) {
                sb.append(' ');
            }
            sb.append(tokens.get(i));
        }
        return sb.toString();
    }

    /** Batch tokenize. */
    default List<List<String>> tokenizeBatch(List<String> texts) {
        List<List<String>> out = new ArrayList<>(texts == null ? 0 : texts.size());
        if (texts == null) {
            return out;
        }
        for (String t : texts) {
            out.add(tokenize(t));
        }
        return out;
    }
}
