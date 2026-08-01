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
package org.bytedeco.pytorch.llm.llamafactory.data;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.llm.llamafactory.data.collator.DataCollator;
import org.bytedeco.pytorch.llm.llamafactory.hparams.DataArgs;
import org.bytedeco.pytorch.llm.llamafactory.hparams.Stage;
import org.bytedeco.pytorch.llm.trl.BaseTrainer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Mini data-loader over tokenized features (shuffle / drop-last / epoch cycling).
 *
 * <p>Produces {@link BaseTrainer.BatchSupplier} instances for factory train loops.
 * Not a full PyTorch DataLoader — enough for offline fine-tune and benchmarks.
 */
public final class DataLoaderFactory {

    private final List<Map<String, Object>> features;
    private final DataCollator collator;
    private final int batchSize;
    private final boolean shuffle;
    private final boolean dropLast;
    private final long seed;

    public DataLoaderFactory(
            List<Map<String, Object>> features,
            DataCollator collator,
            int batchSize,
            boolean shuffle,
            boolean dropLast,
            long seed) {
        this.features = Collections.unmodifiableList(new ArrayList<>(
                Objects.requireNonNull(features, "features")));
        this.collator = Objects.requireNonNull(collator, "collator");
        this.batchSize = Math.max(1, batchSize);
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.seed = seed;
        if (this.features.isEmpty()) {
            throw new IllegalArgumentException("features must be non-empty");
        }
    }

    public static DataLoaderFactory fromBuilder(
            DatasetBuilder builder,
            List<Map<String, Object>> rawRows,
            int batchSize,
            long seed) {
        List<Map<String, Object>> feats = builder.buildFeatures(rawRows);
        return new DataLoaderFactory(feats, builder.collator(), batchSize, true, false, seed);
    }

    public static DataLoaderFactory fromArgs(
            DataArgs dataArgs,
            Stage stage,
            List<Map<String, Object>> rawRows,
            int batchSize,
            long seed) {
        DatasetBuilder b = DatasetBuilder.from(dataArgs, stage);
        return fromBuilder(b, rawRows, batchSize, seed);
    }

    public int size() { return features.size(); }
    public int batchSize() { return batchSize; }
    public int batchesPerEpoch() {
        int n = features.size();
        if (dropLast) {
            return Math.max(1, n / batchSize);
        }
        return Math.max(1, (n + batchSize - 1) / batchSize);
    }

    public List<Map<String, Object>> features() { return features; }
    public DataCollator collator() { return collator; }

    /**
     * One-epoch supplier; returns {@code null} after the last batch.
     */
    public BaseTrainer.BatchSupplier oneEpoch() {
        List<Integer> order = order();
        AtomicInteger idx = new AtomicInteger(0);
        int total = dropLast
                ? (order.size() / batchSize) * batchSize
                : order.size();
        return () -> {
            int start = idx.getAndAdd(batchSize);
            if (start >= total) {
                return null;
            }
            int end = Math.min(start + batchSize, order.size());
            if (dropLast && end - start < batchSize) {
                return null;
            }
            List<Map<String, Object>> slice = new ArrayList<>(end - start);
            for (int i = start; i < end; i++) {
                slice.add(features.get(order.get(i)));
            }
            return collator.collate(slice);
        };
    }

    /**
     * Cycles epochs forever (or until {@code maxBatches}); used with {@code max_steps}.
     */
    public BaseTrainer.BatchSupplier cycling(int maxBatches) {
        AtomicInteger emitted = new AtomicInteger(0);
        AtomicInteger cursor = new AtomicInteger(0);
        List<Integer> order = new ArrayList<>(order());
        Random rng = new Random(seed);
        return () -> {
            if (maxBatches > 0 && emitted.get() >= maxBatches) {
                return null;
            }
            if (cursor.get() + batchSize > order.size()) {
                // reshuffle each epoch
                cursor.set(0);
                if (shuffle) {
                    Collections.shuffle(order, rng);
                }
            }
            int start = cursor.getAndAdd(batchSize);
            List<Map<String, Object>> slice = new ArrayList<>(batchSize);
            for (int i = 0; i < batchSize; i++) {
                int oi = order.get((start + i) % order.size());
                slice.add(features.get(oi));
            }
            emitted.incrementAndGet();
            return collator.collate(slice);
        };
    }

    /** Cooperative cancel wrapper around a supplier. */
    public static BaseTrainer.BatchSupplier cancellable(
            BaseTrainer.BatchSupplier inner, AtomicBoolean stop) {
        return () -> {
            if (stop != null && stop.get()) {
                return null;
            }
            return inner.next();
        };
    }

    private List<Integer> order() {
        List<Integer> idx = new ArrayList<>(features.size());
        for (int i = 0; i < features.size(); i++) idx.add(i);
        if (shuffle) {
            Collections.shuffle(idx, new Random(seed));
        }
        return idx;
    }

    /** Materialize all batches of one epoch (debug / bench). */
    public List<Map<String, Tensor>> collectEpoch() {
        List<Map<String, Tensor>> out = new ArrayList<>();
        BaseTrainer.BatchSupplier s = oneEpoch();
        Map<String, Tensor> b;
        while ((b = s.next()) != null) {
            out.add(b);
        }
        return out;
    }
}
