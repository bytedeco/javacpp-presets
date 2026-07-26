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
package org.bytedeco.pytorch.utils.text.vocab;

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
import java.util.Random;

/**
 * Word vector table loaded from a text file ({@code word f1 f2 ... fd}).
 * Missing words get a deterministic hash-based random vector.
 */
public class Vectors {

    protected final Map<String, float[]> table;
    protected final int dim;
    protected final boolean lower;
    protected final long seed;

    public Vectors(Map<String, float[]> table, int dim) {
        this(table, dim, true, 42L);
    }

    public Vectors(Map<String, float[]> table, int dim, boolean lower, long seed) {
        this.table = new LinkedHashMap<>(Objects.requireNonNull(table, "table"));
        this.dim = dim;
        this.lower = lower;
        this.seed = seed;
    }

    /** Load text word vectors: each line {@code word dim_floats...}. Optional header {@code n dim}. */
    public static Vectors fromFile(Path path) {
        Map<String, float[]> table = new LinkedHashMap<>();
        int dim = -1;
        try (BufferedReader br = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            boolean first = true;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                String[] parts = line.split("\\s+");
                if (first && parts.length == 2) {
                    try {
                        Integer.parseInt(parts[0]);
                        dim = Integer.parseInt(parts[1]);
                        first = false;
                        continue;
                    } catch (NumberFormatException ignore) {
                        // not a header
                    }
                }
                first = false;
                if (parts.length < 2) {
                    continue;
                }
                String word = parts[0];
                int d = parts.length - 1;
                if (dim < 0) {
                    dim = d;
                }
                float[] vec = new float[dim];
                int n = Math.min(dim, d);
                for (int i = 0; i < n; i++) {
                    try {
                        vec[i] = Float.parseFloat(parts[i + 1]);
                    } catch (NumberFormatException e) {
                        vec[i] = 0f;
                    }
                }
                table.put(word, vec);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (dim < 0) {
            dim = 0;
        }
        return new Vectors(table, dim);
    }

    /** Create empty vectors of given dimension. */
    public static Vectors empty(int dim) {
        return new Vectors(new HashMap<>(), dim);
    }

    public float[] get(String word) {
        if (word == null) {
            return randomVector("__null__");
        }
        String key = lower ? word.toLowerCase(Locale.ROOT) : word;
        float[] v = table.get(key);
        if (v != null) {
            return v.clone();
        }
        // try original case
        v = table.get(word);
        if (v != null) {
            return v.clone();
        }
        return randomVector(key);
    }

    public boolean contains(String word) {
        if (word == null) {
            return false;
        }
        if (table.containsKey(word)) {
            return true;
        }
        return lower && table.containsKey(word.toLowerCase(Locale.ROOT));
    }

    protected float[] randomVector(String key) {
        Random r = new Random(seed ^ (long) key.hashCode() * 0x9E3779B97F4A7C15L);
        float[] v = new float[dim];
        float norm = 0f;
        for (int i = 0; i < dim; i++) {
            v[i] = (float) r.nextGaussian() * 0.1f;
            norm += v[i] * v[i];
        }
        if (norm > 0 && dim > 0) {
            float s = (float) (1.0 / Math.sqrt(norm));
            for (int i = 0; i < dim; i++) {
                v[i] *= s;
            }
        }
        return v;
    }

    public int dim() {
        return dim;
    }

    public int size() {
        return table.size();
    }

    public Map<String, float[]> table() {
        return Collections.unmodifiableMap(table);
    }

    public List<String> words() {
        return new ArrayList<>(table.keySet());
    }

    /** Cosine similarity between two words (OOV uses random fallback). */
    public double similarity(String a, String b) {
        float[] va = get(a);
        float[] vb = get(b);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < dim; i++) {
            dot += va[i] * vb[i];
            na += va[i] * va[i];
            nb += vb[i] * vb[i];
        }
        if (na == 0 || nb == 0) {
            return 0;
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    /** Average vector for a list of tokens. */
    public float[] getMean(List<String> tokens) {
        float[] mean = new float[dim];
        if (tokens == null || tokens.isEmpty()) {
            return mean;
        }
        int n = 0;
        for (String t : tokens) {
            float[] v = get(t);
            for (int i = 0; i < dim; i++) {
                mean[i] += v[i];
            }
            n++;
        }
        if (n > 0) {
            for (int i = 0; i < dim; i++) {
                mean[i] /= n;
            }
        }
        return mean;
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(size=" + size() + ", dim=" + dim + ")";
    }
}
