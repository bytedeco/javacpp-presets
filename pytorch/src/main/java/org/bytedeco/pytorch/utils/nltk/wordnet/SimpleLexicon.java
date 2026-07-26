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
package org.bytedeco.pytorch.utils.nltk.wordnet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Tiny built-in synonym lexicon (not full WordNet). Enough for offline demos / Ragas heuristics.
 */
public final class SimpleLexicon {

    public static final class Synset {
        public final String id;
        public final String pos; // n|v|a|r
        public final List<String> lemmas;
        public final String definition;

        public Synset(String id, String pos, List<String> lemmas, String definition) {
            this.id = id;
            this.pos = pos;
            this.lemmas = List.copyOf(lemmas);
            this.definition = definition == null ? "" : definition;
        }

        @Override
        public String toString() {
            return "Synset{" + id + " " + pos + " " + lemmas + "}";
        }
    }

    private final Map<String, List<Synset>> index = new LinkedHashMap<>();

    public SimpleLexicon() {
        add("car.n.01", "n", Arrays.asList("car", "auto", "automobile", "machine"), "a motor vehicle");
        add("vehicle.n.01", "n", Arrays.asList("vehicle", "transport"), "a conveyance");
        add("dog.n.01", "n", Arrays.asList("dog", "canine", "hound"), "a domestic animal");
        add("cat.n.01", "n", Arrays.asList("cat", "feline"), "a domestic animal");
        add("happy.a.01", "a", Arrays.asList("happy", "glad", "joyful", "pleased"), "feeling pleasure");
        add("sad.a.01", "a", Arrays.asList("sad", "unhappy", "sorrowful"), "feeling sorrow");
        add("big.a.01", "a", Arrays.asList("big", "large", "great"), "above average size");
        add("small.a.01", "a", Arrays.asList("small", "little", "tiny"), "below average size");
        add("run.v.01", "v", Arrays.asList("run", "jog", "sprint"), "move fast on foot");
        add("walk.v.01", "v", Arrays.asList("walk", "stroll"), "move on foot");
        add("good.a.01", "a", Arrays.asList("good", "excellent", "fine"), "of high quality");
        add("bad.a.01", "a", Arrays.asList("bad", "poor", "awful"), "of low quality");
        add("person.n.01", "n", Arrays.asList("person", "human", "individual"), "a human being");
        add("say.v.01", "v", Arrays.asList("say", "tell", "state"), "express in words");
        add("quickly.r.01", "r", Arrays.asList("quickly", "rapidly", "fast"), "with speed");
    }

    private void add(String id, String pos, List<String> lemmas, String def) {
        Synset s = new Synset(id, pos, lemmas, def);
        for (String lemma : lemmas) {
            index.computeIfAbsent(lemma.toLowerCase(Locale.ROOT), k -> new ArrayList<>()).add(s);
        }
    }

    public List<Synset> synsets(String word) {
        if (word == null) return Collections.emptyList();
        return List.copyOf(index.getOrDefault(word.toLowerCase(Locale.ROOT), List.of()));
    }

    public List<String> lemmas(String word) {
        List<String> out = new ArrayList<>();
        for (Synset s : synsets(word)) {
            out.addAll(s.lemmas);
        }
        return out.stream().distinct().toList();
    }

    public boolean areSynonyms(String a, String b) {
        if (a == null || b == null) return false;
        if (a.equalsIgnoreCase(b)) return true;
        for (Synset s : synsets(a)) {
            for (String l : s.lemmas) {
                if (l.equalsIgnoreCase(b)) return true;
            }
        }
        return false;
    }

    public int size() {
        return index.size();
    }

    public Map<String, List<Synset>> index() {
        return Collections.unmodifiableMap(index);
    }

    public static SimpleLexicon getDefault() {
        return Holder.INSTANCE;
    }

    private static final class Holder {
        static final SimpleLexicon INSTANCE = new SimpleLexicon();
    }
}
