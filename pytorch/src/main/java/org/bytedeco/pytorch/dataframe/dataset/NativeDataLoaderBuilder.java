package org.bytedeco.pytorch.dataframe.dataset;

import java.util.Objects;

import org.bytedeco.pytorch.data.dataloader.RandomDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialDataLoader;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.data.sampler.RandomSampler;
import org.bytedeco.pytorch.data.sampler.SequentialSampler;

/**
 * Fluent builder for native {@link RandomDataLoader} / {@link SequentialDataLoader}
 * over a {@link DataFrameNativeDataset}.
 *
 * <p>Mirrors common PyTorch DataLoader knobs: batch_size, shuffle, drop_last, workers.
 * Default {@code workers=0} (main-thread) is recommended when the dataset is a
 * Java virtual subclass.
 */
public final class NativeDataLoaderBuilder {

    private final DataFrameNativeDataset dataset;
    private int batchSize = 32;
    private boolean shuffle = false;
    private boolean dropLast = false;
    private int workers = 0;

    NativeDataLoaderBuilder(DataFrameNativeDataset dataset) {
        this.dataset = Objects.requireNonNull(dataset, "dataset");
    }

    public NativeDataLoaderBuilder batchSize(int v) {
        this.batchSize = Math.max(1, v);
        return this;
    }

    public NativeDataLoaderBuilder shuffle(boolean v) {
        this.shuffle = v;
        return this;
    }

    public NativeDataLoaderBuilder dropLast(boolean v) {
        this.dropLast = v;
        return this;
    }

    /**
     * Number of native worker threads. Prefer {@code 0} for Java virtual datasets
     * unless you have verified multi-thread virtual dispatch.
     */
    public NativeDataLoaderBuilder workers(int v) {
        this.workers = Math.max(0, v);
        return this;
    }

    public DataLoaderOptions options() {
        DataLoaderOptions opts = new DataLoaderOptions(batchSize);
        opts.drop_last(dropLast);
        opts.workers(workers);
        return opts;
    }

    public long datasetSize() {
        return dataset.source().sizeLong();
    }

    /** Build random or sequential loader based on {@link #shuffle(boolean)}. */
    public Object build() {
        return shuffle ? buildRandom() : buildSequential();
    }

    public RandomDataLoader buildRandom() {
        long n = datasetSize();
        RandomSampler sampler = new RandomSampler(n);
        return new RandomDataLoader(dataset, sampler, options());
    }

    public SequentialDataLoader buildSequential() {
        long n = datasetSize();
        SequentialSampler sampler = new SequentialSampler(n);
        return new SequentialDataLoader(dataset, sampler, options());
    }

    public DataFrameNativeDataset dataset() { return dataset; }
    public int batchSize() { return batchSize; }
    public boolean shuffle() { return shuffle; }
    public boolean dropLast() { return dropLast; }
    public int workers() { return workers; }
}
