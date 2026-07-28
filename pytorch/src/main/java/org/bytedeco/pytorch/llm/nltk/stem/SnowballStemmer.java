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
package org.bytedeco.pytorch.llm.nltk.stem;

/**
 * English Snowball-ish stemmer — delegates to Porter with light extra suffix rules.
 */
public final class SnowballStemmer {
    private final PorterStemmer porter = new PorterStemmer();
    private final String language;

    public SnowballStemmer() { this("english"); }
    public SnowballStemmer(String language) {
        this.language = language == null ? "english" : language.toLowerCase();
    }

    public String getLanguage() { return language; }

    public String stem(String word) {
        if (word == null) return "";
        String w = word.toLowerCase();
        // light English extras before Porter
        if ("english".equals(language) || "en".equals(language)) {
            if (w.endsWith("ingly") && w.length() > 6) w = w.substring(0, w.length() - 4) + "e";
            else if (w.endsWith("'s") && w.length() > 2) w = w.substring(0, w.length() - 2);
        }
        return porter.stem(w);
    }
}
