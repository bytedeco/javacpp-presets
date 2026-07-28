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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Basic English tokenizer (torchtext.transforms.BasicEnglishNormalize-style).
 * Lowercases and splits on non-alphanumeric characters.
 */
public final class BasicEnglishTokenizer implements Tokenizer {

    private static final Pattern TOKEN = Pattern.compile("[\\p{L}\\p{N}]+");

    private final boolean lower;

    public BasicEnglishTokenizer() {
        this(true);
    }

    public BasicEnglishTokenizer(boolean lower) {
        this.lower = lower;
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return tokens;
        }
        String src = lower ? text.toLowerCase(Locale.ROOT) : text;
        Matcher m = TOKEN.matcher(src);
        while (m.find()) {
            tokens.add(m.group());
        }
        return tokens;
    }
}
