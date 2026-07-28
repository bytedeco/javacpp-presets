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
package org.bytedeco.pytorch.llm.text.vocab;

import java.nio.file.Path;
import java.util.Map;

/**
 * GloVe word vectors loader (text format). Subclass of {@link Vectors} with named presets.
 *
 * <pre>{@code
 * GloVe glove = GloVe.fromFile(Path.of("glove.6B.50d.txt"));
 * float[] v = glove.get("king");
 * }</pre>
 */
public final class GloVe extends Vectors {

    public enum Name {
        GLOVE_6B_50D("glove.6B.50d", 50),
        GLOVE_6B_100D("glove.6B.100d", 100),
        GLOVE_6B_200D("glove.6B.200d", 200),
        GLOVE_6B_300D("glove.6B.300d", 300),
        GLOVE_42B_300D("glove.42B.300d", 300),
        GLOVE_840B_300D("glove.840B.300d", 300);

        private final String fileStem;
        private final int dim;

        Name(String fileStem, int dim) {
            this.fileStem = fileStem;
            this.dim = dim;
        }

        public String fileStem() {
            return fileStem;
        }

        public int dim() {
            return dim;
        }
    }

    private final String name;

    public GloVe(Map<String, float[]> table, int dim) {
        this(table, dim, "glove");
    }

    public GloVe(Map<String, float[]> table, int dim, String name) {
        super(table, dim, true, 42L);
        this.name = name == null ? "glove" : name;
    }

    public static GloVe fromFile(Path path) {
        Vectors v = Vectors.fromFile(path);
        return new GloVe(v.table, v.dim, path.getFileName().toString());
    }

    public static GloVe fromFile(Path path, String name) {
        Vectors v = Vectors.fromFile(path);
        return new GloVe(v.table, v.dim, name);
    }

    /** Create an empty GloVe table of the given preset dimension. */
    public static GloVe empty(Name name) {
        return new GloVe(Map.of(), name.dim(), name.fileStem());
    }

    public String name() {
        return name;
    }

    @Override
    public String toString() {
        return "GloVe(name=" + name + ", size=" + size() + ", dim=" + dim + ")";
    }
}
