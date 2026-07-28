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
package org.bytedeco.pytorch.llm.accelerate;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Objects;

/**
 * Rank-strided view over a list (HF prepare_data_loader shard semantics, simplified).
 */
public final class DataLoaderShard<T> implements Iterable<T> {

    private final List<T> data;
    private final int rank;
    private final int worldSize;
    private final boolean evenBatches;
    private final List<T> local;

    public DataLoaderShard(List<T> data, int rank, int worldSize, boolean evenBatches) {
        this.data = Objects.requireNonNull(data, "data");
        this.rank = Math.max(0, rank);
        this.worldSize = Math.max(1, worldSize);
        this.evenBatches = evenBatches;
        this.local = buildLocal();
    }

    public static <T> DataLoaderShard<T> of(List<T> data, PartialState state) {
        return new DataLoaderShard<>(data, state.processIndex(), state.numProcesses(), true);
    }

    public static <T> DataLoaderShard<T> of(List<T> data, int rank, int worldSize) {
        return new DataLoaderShard<>(data, rank, worldSize, true);
    }

    private List<T> buildLocal() {
        List<T> out = new ArrayList<>();
        for (int i = rank; i < data.size(); i += worldSize) {
            out.add(data.get(i));
        }
        if (evenBatches && worldSize > 1 && !data.isEmpty()) {
            int target = (data.size() + worldSize - 1) / worldSize;
            while (out.size() < target && !data.isEmpty()) {
                out.add(data.get(out.size() % data.size()));
            }
        }
        return out;
    }

    public List<T> localData() { return List.copyOf(local); }
    public int size() { return local.size(); }
    public int rank() { return rank; }
    public int worldSize() { return worldSize; }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<T>() {
            int i = 0;
            @Override public boolean hasNext() { return i < local.size(); }
            @Override public T next() {
                if (!hasNext()) throw new NoSuchElementException();
                return local.get(i++);
            }
        };
    }
}
