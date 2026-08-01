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

package org.bytedeco.pytorch.llm.llamacpp;

import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Loaded GGUF model: hparams + tensor inventory (+ optional float caches). */
public final class LlamaModel {

    public static final class TensorBlob {
        public final String name;
        public final long[] shape;
        public final int ggmlType;
        public final long offset;
        public final long nBytes;
        /** Optional dequantized/cached float payload. */
        public final float[] floats;

        public TensorBlob(String name, long[] shape, int ggmlType, long offset, long nBytes, float[] floats) {
            this.name = name;
            this.shape = shape != null ? shape.clone() : new long[0];
            this.ggmlType = ggmlType;
            this.offset = offset;
            this.nBytes = nBytes;
            this.floats = floats;
        }

        public long nElements() {
            long n = 1;
            for (long s : shape) n *= Math.max(1, s);
            return shape.length == 0 ? 0 : n;
        }
    }

    private final Path path;
    private final LlamaHParams hparams;
    private final Map<String, Object> metadata;
    private final Map<String, TensorBlob> tensors;
    private final int ggufVersion;
    private final Map<String, float[]> floatCache = new LinkedHashMap<>();

    public LlamaModel(Path path, LlamaHParams hparams, Map<String, Object> metadata,
                      Map<String, TensorBlob> tensors, int ggufVersion) {
        this.path = Objects.requireNonNull(path);
        this.hparams = Objects.requireNonNull(hparams);
        this.metadata = metadata != null ? Map.copyOf(metadata) : Map.of();
        this.tensors = tensors != null ? Collections.unmodifiableMap(new LinkedHashMap<>(tensors)) : Map.of();
        this.ggufVersion = ggufVersion;
        for (TensorBlob b : this.tensors.values()) {
            if (b.floats != null) floatCache.put(b.name, b.floats);
        }
    }

    public Path path() { return path; }
    public LlamaHParams hparams() { return hparams; }
    public Map<String, Object> metadata() { return metadata; }
    public Map<String, TensorBlob> tensors() { return tensors; }
    public int ggufVersion() { return ggufVersion; }
    public int tensorCount() { return tensors.size(); }

    public Optional<TensorBlob> tensor(String name) {
        return Optional.ofNullable(tensors.get(name));
    }

    public synchronized float[] floats(String name) throws Exception {
        float[] c = floatCache.get(name);
        if (c != null) return c;
        float[] loaded = GgufModelLoader.loadFloatTensor(path.toFile(), name);
        floatCache.put(name, loaded);
        return loaded;
    }

    public Map<String, Object> summary() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("path", path.toString());
        m.put("version", ggufVersion);
        m.put("tensor_count", tensors.size());
        m.putAll(hparams.toMap());
        return m;
    }
}
