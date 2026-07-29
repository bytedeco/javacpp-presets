/*
 * Ported from torch-rechub-scala: torchrec/data/Dataset.scala (Dataset trait)
 *
 * @deprecated Use {@link RecommendDataset} which extends the native
 * {@link org.bytedeco.pytorch.data.Dataset} and is required for RandomDataLoader /
 * SequentialDataLoader integration.
 *
 * Kept as a thin adapter interface for any legacy call sites that only need
 * size + named Batch access.
 */
package org.bytedeco.pytorch.utils.recommend.data;

/**
 * @deprecated Prefer {@link RecommendDataset}.
 */
@Deprecated
public interface Dataset {
    long size();
    Batch get(long index);

    /** Adapt a RecommendDataset to this legacy interface. */
    static Dataset of(RecommendDataset ds) {
        return new Dataset() {
            @Override public long size() { return ds.sizeLong(); }
            @Override public Batch get(long index) { return ds.getBatch(index); }
        };
    }
}
