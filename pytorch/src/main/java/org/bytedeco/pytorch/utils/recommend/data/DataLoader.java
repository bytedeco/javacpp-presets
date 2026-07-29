/*
 * Ported from torch-rechub-scala: torchrec/data/DataLoader.scala + JavaDataLoaderFactory.scala
 *
 * Convenience factories over native RandomDataLoader / SequentialDataLoader for
 * any {@link RecommendDataset}. Also provides a pure-Java named-Batch iterator
 * that stacks samples without going through the C++ DataLoader (useful when you
 * need Map&lt;String,Tensor&gt; features rather than packed Example tensors).
 */
package org.bytedeco.pytorch.utils.recommend.data;

import org.bytedeco.javacpp.annotation.Properties;
import org.bytedeco.pytorch.Device;
import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.TensorVector;
import org.bytedeco.pytorch.data.Example;
import org.bytedeco.pytorch.data.ExampleVector;
import org.bytedeco.pytorch.data.dataloader.RandomDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialDataLoader;
import org.bytedeco.pytorch.dataframe.dataset.NativeBatchSupport;
import org.bytedeco.pytorch.global.torch;
import org.bytedeco.pytorch.global.torch.ScalarType;
import org.bytedeco.pytorch.utils.recommend.DeviceSupport;
import org.bytedeco.pytorch.utils.recommend.TensorHelpers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

@Properties(inherit = org.bytedeco.pytorch.presets.torch.class)
public final class DataLoader {

    private DataLoader() {}

    // ---- native loaders -----------------------------------------------------

    public static RandomDataLoader random(RecommendDataset ds, long batchSize) {
        return ds.randomDataLoader(batchSize);
    }

    public static RandomDataLoader random(RecommendDataset ds, long batchSize, long workers) {
        return ds.randomDataLoader(batchSize, workers);
    }

    public static SequentialDataLoader sequential(RecommendDataset ds, long batchSize) {
        return ds.sequentialDataLoader(batchSize);
    }

    public static SequentialDataLoader sequential(RecommendDataset ds, long batchSize, long workers) {
        return ds.sequentialDataLoader(batchSize, workers);
    }

    /** Stack an ExampleVector from a native loader into one Example [B, ...]. */
    public static Example stack(ExampleVector batch) {
        return NativeBatchSupport.stack(batch);
    }

    // ---- pure-Java named Batch iterator -------------------------------------

    /**
     * Iterate {@link Batch} mini-batches with feature maps stacked on dim 0.
     * Does not use the C++ DataLoader — preferred when models take named features.
     */
    public static Iterable<Batch> batches(RecommendDataset dataset, int batchSize,
                                          boolean shuffle, boolean dropLast) {
        return batches(dataset, batchSize, shuffle, dropLast, DeviceSupport.backend());
    }

    public static Iterable<Batch> batches(RecommendDataset dataset, int batchSize,
                                          boolean shuffle, boolean dropLast, String device) {
        return () -> new BatchIterator(dataset, batchSize, shuffle, dropLast, device);
    }

    private static final class BatchIterator implements Iterator<Batch> {
        private final RecommendDataset dataset;
        private final int batchSize;
        private final boolean dropLast;
        private final String device;
        private final int[] indices;
        private int pos;

        BatchIterator(RecommendDataset dataset, int batchSize, boolean shuffle,
                      boolean dropLast, String device) {
            this.dataset = dataset;
            this.batchSize = Math.max(1, batchSize);
            this.dropLast = dropLast;
            this.device = device;
            int n = (int) Math.min(dataset.sizeLong(), Integer.MAX_VALUE);
            this.indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            if (shuffle) {
                Random rng = new Random();
                for (int i = n - 1; i > 0; i--) {
                    int j = rng.nextInt(i + 1);
                    int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
                }
            }
            this.pos = 0;
        }

        @Override
        public boolean hasNext() {
            if (pos >= indices.length) return false;
            if (dropLast) return pos + batchSize <= indices.length;
            return true;
        }

        @Override
        public Batch next() {
            if (!hasNext()) throw new NoSuchElementException();
            int end = Math.min(pos + batchSize, indices.length);
            List<Batch> rows = new ArrayList<>(end - pos);
            for (int i = pos; i < end; i++) {
                rows.add(dataset.getBatch(indices[i]));
            }
            pos = end;
            return stackBatches(rows, device);
        }
    }

    /** Stack a list of single-row Batches into one mini-batch Batch. */
    public static Batch stackBatches(List<Batch> rows, String device) {
        if (rows == null || rows.isEmpty()) {
            throw new IllegalArgumentException("empty batch");
        }
        Map<String, Tensor> sparse = stackMap(collect(rows, b -> b.sparseFeatures));
        Map<String, Tensor> dense = stackMap(collect(rows, b -> b.denseFeatures));
        Map<String, Tensor> seq = stackMap(collect(rows, b -> b.sequenceFeatures));
        Map<String, Tensor> item = stackMap(collect(rows, b -> b.itemFeatures));
        Map<String, Tensor> negItem = stackMap(collect(rows, b -> b.negItemFeatures != null
                ? b.negItemFeatures : Collections.emptyMap()));
        Map<String, Tensor> taskLabels = stackMap(collect(rows, b -> b.taskLabels != null
                ? b.taskLabels : Collections.emptyMap()));
        Tensor labels = stackOptional(rows, b -> b.labels);
        Tensor tokens = stackOptional(rows, b -> b.tokens);
        Tensor positions = stackOptional(rows, b -> b.positions);
        Tensor timeDiffs = stackOptional(rows, b -> b.timeDiffs);
        Tensor targets = stackOptional(rows, b -> b.targets);

        Batch batch = new Batch(sparse, dense, seq, labels, tokens, positions, timeDiffs, targets,
                item,
                negItem.isEmpty() ? null : negItem,
                taskLabels.isEmpty() ? null : taskLabels);
        if (device != null && !"cpu".equals(device)) {
            return batch.to(device);
        }
        return batch;
    }

    private interface FieldGet {
        Map<String, Tensor> get(Batch b);
    }

    private interface TensorGet {
        Tensor get(Batch b);
    }

    private static List<Map<String, Tensor>> collect(List<Batch> rows, FieldGet g) {
        List<Map<String, Tensor>> out = new ArrayList<>(rows.size());
        for (Batch b : rows) out.add(g.get(b));
        return out;
    }

    private static Map<String, Tensor> stackMap(List<Map<String, Tensor>> maps) {
        Map<String, List<Tensor>> acc = new LinkedHashMap<>();
        for (Map<String, Tensor> m : maps) {
            if (m == null) continue;
            for (Map.Entry<String, Tensor> e : m.entrySet()) {
                acc.computeIfAbsent(e.getKey(), k -> new ArrayList<>()).add(e.getValue());
            }
        }
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (Map.Entry<String, List<Tensor>> e : acc.entrySet()) {
            out.put(e.getKey(), stackTensors(e.getValue()));
        }
        return out;
    }

    private static Tensor stackOptional(List<Batch> rows, TensorGet g) {
        List<Tensor> ts = new ArrayList<>();
        for (Batch b : rows) {
            Tensor t = g.get(b);
            if (t != null) ts.add(t);
        }
        if (ts.isEmpty()) return null;
        return stackTensors(ts);
    }

    private static Tensor stackTensors(List<Tensor> ts) {
        if (ts.size() == 1) {
            Tensor t = ts.get(0);
            // ensure batch dim
            return t.dim() == 0 ? t.unsqueeze(0) : t;
        }
        TensorVector vec = new TensorVector(ts.size());
        for (int i = 0; i < ts.size(); i++) {
            Tensor t = ts.get(i);
            if (t.dim() == 0) t = t.unsqueeze(0);
            vec.put(i, t);
        }
        return torch.stack(vec, 0);
    }
}
