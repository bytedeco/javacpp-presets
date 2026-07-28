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
package org.bytedeco.pytorch.llm.nltk.tag;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * Rule-based POS tagger (NLTK {@code RegexpTagger} subset).
 */
public final class RegexpTagger {

    public static final class Rule {
        public final Pattern pattern;
        public final String tag;
        public Rule(String regex, String tag) {
            this.pattern = Pattern.compile(regex);
            this.tag = tag;
        }
    }

    private final List<Rule> rules;
    private final String defaultTag;

    public RegexpTagger(List<Rule> rules, String defaultTag) {
        this.rules = rules == null ? defaultRules() : List.copyOf(rules);
        this.defaultTag = defaultTag == null ? "NN" : defaultTag;
    }

    public RegexpTagger() {
        this(null, "NN");
    }

    public static List<Rule> defaultRules() {
        List<Rule> r = new ArrayList<>();
        r.add(new Rule("^-?[0-9]+(\\.[0-9]+)?$", "CD"));
        r.add(new Rule(".*ing$", "VBG"));
        r.add(new Rule(".*ed$", "VBD"));
        r.add(new Rule(".*es$", "VBZ"));
        r.add(new Rule(".*ould$", "MD"));
        r.add(new Rule(".*'s$", "NN$"));
        r.add(new Rule(".*s$", "NNS"));
        r.add(new Rule("^(The|the|A|a|An|an)$", "DT"));
        r.add(new Rule(".*ly$", "RB"));
        r.add(new Rule(".*", "NN"));
        return r;
    }

    public List<String[]> tag(List<String> tokens) {
        List<String[]> out = new ArrayList<>();
        if (tokens == null) return out;
        for (String t : tokens) {
            String tag = defaultTag;
            for (Rule r : rules) {
                if (r.pattern.matcher(t).matches()) {
                    tag = r.tag;
                    break;
                }
            }
            out.add(new String[]{t, tag});
        }
        return out;
    }

    public Map<String, String> tagMap(List<String> tokens) {
        Map<String, String> m = new LinkedHashMap<>();
        for (String[] p : tag(tokens)) m.put(p[0], p[1]);
        return m;
    }
}
