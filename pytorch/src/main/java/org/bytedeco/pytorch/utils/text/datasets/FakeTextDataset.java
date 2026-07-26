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
package org.bytedeco.pytorch.utils.text.datasets;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Synthetic text classification dataset for tests/benchmarks.
 * Produces random sentences with integer labels.
 */
public final class FakeTextDataset {

    public static final class Sample {
        public final String text;
        public final int label;

        public Sample(String text, int label) {
            this.text = text;
            this.label = label;
        }

        @Override
        public String toString() {
            return "Sample{label=" + label + ", text='" + text + "'}";
        }
    }

    private static final String[] SUBJECTS = {
            "the cat", "a dog", "the model", "our team", "the network",
            "researchers", "the system", "this paper", "the algorithm", "users"
    };
    private static final String[] VERBS = {
            "likes", "hates", "studies", "improves", "learns",
            "predicts", "classifies", "generates", "analyzes", "optimizes"
    };
    private static final String[] OBJECTS = {
            "data", "text", "images", "language", "features",
            "gradients", "embeddings", "tokens", "sentences", "labels"
    };
    private static final String[] ADVERBS = {
            "quickly", "carefully", "often", "rarely", "efficiently",
            "accurately", "poorly", "well", "sometimes", "always"
    };

    private final int size;
    private final int numClasses;
    private final long seed;
    private final List<Sample> cache;

    public FakeTextDataset(int size, int numClasses) {
        this(size, numClasses, 42L, true);
    }

    public FakeTextDataset(int size, int numClasses, long seed) {
        this(size, numClasses, seed, true);
    }

    public FakeTextDataset(int size, int numClasses, long seed, boolean cacheAll) {
        this.size = Math.max(0, size);
        this.numClasses = Math.max(1, numClasses);
        this.seed = seed;
        if (cacheAll) {
            this.cache = new ArrayList<>(this.size);
            for (int i = 0; i < this.size; i++) {
                this.cache.add(generate(i));
            }
        } else {
            this.cache = null;
        }
    }

    public int size() {
        return size;
    }

    public int numClasses() {
        return numClasses;
    }

    public Sample get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("index=" + index + " size=" + size);
        }
        if (cache != null) {
            return cache.get(index);
        }
        return generate(index);
    }

    public List<Sample> asList() {
        if (cache != null) {
            return Collections.unmodifiableList(cache);
        }
        List<Sample> list = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            list.add(generate(i));
        }
        return list;
    }

    public List<String> texts() {
        List<String> t = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            t.add(get(i).text);
        }
        return t;
    }

    public int[] labels() {
        int[] y = new int[size];
        for (int i = 0; i < size; i++) {
            y[i] = get(i).label;
        }
        return y;
    }

    private Sample generate(int index) {
        Random r = new Random(seed + index * 9973L);
        int nSentences = 1 + r.nextInt(3);
        StringBuilder sb = new StringBuilder();
        for (int s = 0; s < nSentences; s++) {
            if (s > 0) {
                sb.append(' ');
            }
            String subj = SUBJECTS[r.nextInt(SUBJECTS.length)];
            String verb = VERBS[r.nextInt(VERBS.length)];
            String obj = OBJECTS[r.nextInt(OBJECTS.length)];
            if (r.nextBoolean()) {
                sb.append(subj).append(' ').append(verb).append(' ')
                        .append(ADVERBS[r.nextInt(ADVERBS.length)]).append(' ').append(obj);
            } else {
                sb.append(subj).append(' ').append(verb).append(' ').append(obj);
            }
            sb.append('.');
        }
        // weakly correlate label with presence of certain words for demo
        String text = sb.toString();
        int label = Math.floorMod(text.toLowerCase(Locale.ROOT).hashCode(), numClasses);
        if (text.contains("improves") || text.contains("optimizes")) {
            label = 0 % numClasses;
        } else if (text.contains("hates") || text.contains("poorly")) {
            label = Math.min(1, numClasses - 1);
        }
        return new Sample(text, label);
    }

    @Override
    public String toString() {
        return "FakeTextDataset(size=" + size + ", numClasses=" + numClasses + ")";
    }
}
