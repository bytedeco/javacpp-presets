package org.bytedeco.pytorch.dataframe.dataset;

import java.util.*;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.dataloader.RandomDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialDataLoader;
import org.bytedeco.pytorch.global.torch;

/**
 * Pure-Java mini-batch DataLoader over {@link DataFrameDataset}.
 *
 * <p>Parameters mirror PyTorch DataLoader: batch_size, shuffle, drop_last, seed.
 * Each {@link Batch} exposes stacked feature tensors and labels ready for training.
 *
 * <p><b>Memory / PointerScope (critical for long training runs):</b>
 * {@link #collate(int[])} allocates several native {@link Tensor}s per batch
 * ({@code torch.tensor(...)} for each feature + labels). JavaCPP attaches newly
 * created Pointers to the <em>current</em> {@link org.bytedeco.javacpp.PointerScope}.
 * If you call {@code iterator().next()} <em>outside</em> a per-batch scope, those
 * tensors never get released → progressive native/MPS memory growth and training
 * that gets slower every step. Always do:
 * <pre>
 *   try (PointerScope scope = new PointerScope()) {
 *       Batch batch = it.next(); // collate tensors attach here
 *       // forward / backward ...
 *   } // all collate + intermediate tensors freed
 * </pre>
 * Model weights / optimizer state must be allocated <em>outside</em> the per-batch
 * scope so {@code scope.close()} does not free them.
 *
 * <p>For native C++ DataLoader interop use {@link #toRandomDataLoader()} /
 * {@link #toSequentialDataLoader()} (via {@link DataFrameNativeDataset}) or
 * {@link DataFrameDataset#nativeDataLoader()}.
 *
 * <pre>
 *   DataFrameDataLoader loader = ds.dataloader()
 *       .batchSize(256)
 *       .shuffle(true)
 *       .dropLast(false)
 *       .seed(42L)
 *       .build();
 *
 *   for (DataFrameDataLoader.Batch batch : loader) {
 *       Tensor x = batch.features();           // [B, n_feat]
 *       Tensor seq = batch.feature("item_seq"); // [B, 64]
 *       Tensor y = batch.labels();              // [B] or [B, n_label]
 *   }
 *
 *   // Symmetric native path
 *   SequentialDataLoader nloader = loader.toSequentialDataLoader();
 * </pre>
 */
public final class DataFrameDataLoader implements Iterable<DataFrameDataLoader.Batch> {

    public static final class Options {
        public int batchSize = 32;
        public boolean shuffle = false;
        public boolean dropLast = false;
        public Long seed = null;

        public Options batchSize(int v) { this.batchSize = Math.max(1, v); return this; }
        public Options shuffle(boolean v) { this.shuffle = v; return this; }
        public Options dropLast(boolean v) { this.dropLast = v; return this; }
        public Options seed(Long v) { this.seed = v; return this; }
    }

    public static final class Builder {
        private final DataFrameDataset dataset;
        private final Options opts = new Options();

        Builder(DataFrameDataset dataset) {
            this.dataset = Objects.requireNonNull(dataset);
        }

        public Builder batchSize(int v) { opts.batchSize(v); return this; }
        public Builder shuffle(boolean v) { opts.shuffle(v); return this; }
        public Builder dropLast(boolean v) { opts.dropLast(v); return this; }
        public Builder seed(long v) { opts.seed(v); return this; }
        public Builder seed(Long v) { opts.seed(v); return this; }

        public DataFrameDataLoader build() {
            return new DataFrameDataLoader(dataset, opts);
        }
    }

    public static Builder builder(DataFrameDataset ds) {
        return new Builder(ds);
    }

    /** One mini-batch. */
    public static final class Batch {
        private final int[] indices;
        private final Map<String, Tensor> features;
        private final Tensor stackedFeatures;
        private final Tensor labels;

        Batch(int[] indices, Map<String, Tensor> features, Tensor stackedFeatures, Tensor labels) {
            this.indices = indices;
            this.features = features;
            this.stackedFeatures = stackedFeatures;
            this.labels = labels;
        }

        public int size() { return indices.length; }
        public int[] indices() { return indices.clone(); }
        public Map<String, Tensor> featuresMap() { return features; }
        public Tensor feature(String name) {
            Tensor t = features.get(name);
            if (t == null) throw new IllegalArgumentException("No feature in batch: " + name);
            return t;
        }
        /** Stacked scalar features {@code [B, n_feat]}, or first sequence feature. */
        public Tensor features() {
            if (stackedFeatures != null) return stackedFeatures;
            if (!features.isEmpty()) return features.values().iterator().next();
            return torch.empty(new long[]{indices.length, 0});
        }
        public Tensor data() { return features(); }
        public Tensor featuresStacked() { return stackedFeatures; }
        public Tensor labels() { return labels; }
        public Tensor target() { return labels; }
    }

    private final DataFrameDataset dataset;
    private final Options options;
    private int[] order;
    private int epoch;

    private DataFrameDataLoader(DataFrameDataset dataset, Options options) {
        this.dataset = dataset;
        this.options = options;
        this.epoch = 0;
        reshuffle();
    }

    public DataFrameDataset dataset() { return dataset; }
    public Options options() { return options; }
    public int batchSize() { return options.batchSize; }
    public int epoch() { return epoch; }

    public int numBatches() {
        int n = dataset.size();
        if (n == 0) return 0;
        if (options.dropLast) return n / options.batchSize;
        return (n + options.batchSize - 1) / options.batchSize;
    }

    /** Reshuffle indices (call each epoch when shuffle=true). */
    public void reshuffle() {
        int n = dataset.size();
        order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        if (options.shuffle && n > 1) {
            long s = options.seed == null ? System.nanoTime() : options.seed + epoch;
            Random rng = new Random(s);
            for (int i = n - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
        }
    }

    @Override
    public Iterator<Batch> iterator() {
        if (options.shuffle) {
            reshuffle();
            epoch++;
        }
        return new Iterator<>() {
            int pos = 0;
            @Override public boolean hasNext() {
                int remaining = order.length - pos;
                if (remaining <= 0) return false;
                if (options.dropLast && remaining < options.batchSize) return false;
                return true;
            }
            @Override public Batch next() {
                if (!hasNext()) throw new NoSuchElementException();
                int end = Math.min(pos + options.batchSize, order.length);
                int[] idx = Arrays.copyOfRange(order, pos, end);
                pos = end;
                return collate(idx);
            }
        };
    }

    private Batch collate(int[] idx) {
        int B = idx.length;
        Map<String, Tensor> feats = new LinkedHashMap<>();
        Tensor stacked = null;

        int nFeat = dataset.scalarFeatureCount();
        String[] scalars = dataset.scalarFeatureNames();
        if (nFeat > 0) {
            float[] batch = dataset.gatherScalars(idx);
            stacked = torch.tensor(batch).reshape(new long[]{B, nFeat});
            feats.put("__stacked__", stacked);
            for (int j = 0; j < nFeat; j++) {
                float[] col = new float[B];
                for (int b = 0; b < B; b++) col[b] = batch[b * nFeat + j];
                feats.put(scalars[j], torch.tensor(col));
            }
        }

        for (String name : dataset.sequenceFeatureNames()) {
            int dim = dataset.sequenceDim(name);
            Object packed = dataset.gatherSequence(name, idx);
            if (packed instanceof long[]) {
                feats.put(name, torch.tensor((long[]) packed).reshape(new long[]{B, dim}));
            } else {
                feats.put(name, torch.tensor((float[]) packed).reshape(new long[]{B, dim}));
            }
        }

        Tensor labels = null;
        int nLab = dataset.labelCount();
        if (nLab > 0) {
            Object packed = dataset.gatherLabels(idx);
            if (dataset.labelsAsLong()) {
                long[] data = (long[]) packed;
                labels = nLab == 1
                    ? torch.tensor(data)
                    : torch.tensor(data).reshape(new long[]{B, nLab});
            } else {
                float[] data = (float[]) packed;
                labels = nLab == 1
                    ? torch.tensor(data)
                    : torch.tensor(data).reshape(new long[]{B, nLab});
            }
        }

        return new Batch(idx, feats, stacked, labels);
    }

    // ---- native DataLoader interop ------------------------------------------

    /**
     * Build a native {@link RandomDataLoader} over {@link DataFrameNativeDataset}
     * with the same batch size / drop_last as this loader.
     */
    public RandomDataLoader toRandomDataLoader() {
        return dataset.nativeDataLoader()
            .batchSize(options.batchSize)
            .shuffle(true)
            .dropLast(options.dropLast)
            .workers(0)
            .buildRandom();
    }

    /**
     * Build a native {@link SequentialDataLoader} over {@link DataFrameNativeDataset}
     * with the same batch size / drop_last as this loader.
     */
    public SequentialDataLoader toSequentialDataLoader() {
        return dataset.nativeDataLoader()
            .batchSize(options.batchSize)
            .shuffle(false)
            .dropLast(options.dropLast)
            .workers(0)
            .buildSequential();
    }

    /**
     * Choose random or sequential native loader based on this loader's
     * {@link Options#shuffle} flag.
     */
    public Object toNativeDataLoader() {
        return options.shuffle ? toRandomDataLoader() : toSequentialDataLoader();
    }

    /**
     * Symmetry factory: pure-Java loader from a native dataset adapter
     * (re-wraps {@link DataFrameNativeDataset#source()}).
     */
    public static DataFrameDataLoader fromNativeDataset(DataFrameNativeDataset ds) {
        return fromNativeDataset(ds, new Options());
    }

    public static DataFrameDataLoader fromNativeDataset(DataFrameNativeDataset ds, Options opts) {
        Objects.requireNonNull(ds, "ds");
        return new DataFrameDataLoader(ds.source(), opts == null ? new Options() : opts);
    }

    public static DataFrameDataLoader fromNativeDataset(DataFrameNativeDataset ds,
                                                        int batchSize, boolean shuffle) {
        return fromNativeDataset(ds, new Options().batchSize(batchSize).shuffle(shuffle));
    }
}
