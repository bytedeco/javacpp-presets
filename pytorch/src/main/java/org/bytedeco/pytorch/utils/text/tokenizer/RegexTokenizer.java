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
import java.util.Locale;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Regex-based tokenizer. By default splits on non-alphanumeric runs (same as BasicEnglish).
 */
public final class RegexTokenizer implements Tokenizer {

    private final Pattern pattern;
    private final boolean lower;
    private final boolean gaps;

    /**
     * @param regex token pattern when {@code gaps=false}, or delimiter pattern when {@code gaps=true}
     * @param lower lowercase input first
     * @param gaps if true, split on matches of {@code regex}; else extract matches of {@code regex}
     */
    public RegexTokenizer(String regex, boolean lower, boolean gaps) {
        this.pattern = Pattern.compile(Objects.requireNonNull(regex, "regex"));
        this.lower = lower;
        this.gaps = gaps;
    }

    public RegexTokenizer(String regex) {
        this(regex, true, false);
    }

    /** Default: extract {@code [\\p{L}\\p{N}]+}, lowercased. */
    public RegexTokenizer() {
        this("[\\p{L}\\p{N}]+", true, false);
    }

    @Override
    public List<String> tokenize(String text) {
        List<String> tokens = new ArrayList<>();
        if (text == null || text.isEmpty()) {
            return tokens;
        }
        String src = lower ? text.toLowerCase(Locale.ROOT) : text;
        if (gaps) {
            String[] parts = pattern.split(src);
            for (String p : parts) {
                if (!p.isEmpty()) {
                    tokens.add(p);
                }
            }
        } else {
            Matcher m = pattern.matcher(src);
            while (m.find()) {
                tokens.add(m.group());
            }
        }
        return tokens;
    }

    public Pattern pattern() {
        return pattern;
    }
}
