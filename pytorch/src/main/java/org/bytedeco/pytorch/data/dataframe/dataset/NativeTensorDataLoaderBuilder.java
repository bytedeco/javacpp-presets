package org.bytedeco.pytorch.data.dataframe.dataset;

import java.util.Objects;

import org.bytedeco.pytorch.data.dataloader.RandomTensorDataLoader;
import org.bytedeco.pytorch.data.dataloader.SequentialTensorDataLoader;
import org.bytedeco.pytorch.data.options.DataLoaderOptions;
import org.bytedeco.pytorch.data.sampler.RandomSampler;
import org.bytedeco.pytorch.data.sampler.SequentialSampler;

/**
 * Fluent builder for native {@link RandomTensorDataLoader} /
 * {@link SequentialTensorDataLoader} over a {@link DataFrameJavaTensorDataset}.
 */
public final class NativeTensorDataLoaderBuilder {

    private final DataFrameJavaTensorDataset dataset;
    private int batchSize = 32;
    private boolean shuffle = false;
    private boolean dropLast = false;
    private int workers = 0;

    NativeTensorDataLoaderBuilder(DataFrameJavaTensorDataset dataset) {
        this.dataset = Objects.requireNonNull(dataset, "dataset");
    }

    public NativeTensorDataLoaderBuilder batchSize(int v) {
        this.batchSize = Math.max(1, v);
        return this;
    }

    public NativeTensorDataLoaderBuilder shuffle(boolean v) {
        this.shuffle = v;
        return this;
    }

    public NativeTensorDataLoaderBuilder dropLast(boolean v) {
        this.dropLast = v;
        return this;
    }

    public NativeTensorDataLoaderBuilder workers(int v) {
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

    public Object build() {
        return shuffle ? buildRandom() : buildSequential();
    }

    public RandomTensorDataLoader buildRandom() {
        long n = datasetSize();
        RandomSampler sampler = new RandomSampler(n);
        return new RandomTensorDataLoader(dataset, sampler, options());
    }

    public SequentialTensorDataLoader buildSequential() {
        long n = datasetSize();
        SequentialSampler sampler = new SequentialSampler(n);
        return new SequentialTensorDataLoader(dataset, sampler, options());
    }

    public DataFrameJavaTensorDataset dataset() { return dataset; }
    public int batchSize() { return batchSize; }
    public boolean shuffle() { return shuffle; }
    public boolean dropLast() { return dropLast; }
    public int workers() { return workers; }
}
