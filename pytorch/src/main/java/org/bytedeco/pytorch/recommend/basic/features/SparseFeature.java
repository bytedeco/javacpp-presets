/*
 * Ported from torch-rechub-scala: torchrec/basic/features/Feature.scala (SparseFeature)
 */
package org.bytedeco.pytorch.recommend.basic.features;

import java.util.Objects;

/**
 * Sparse (categorical) feature with embedding table.
 */
public final class SparseFeature implements Feature {
    private final String name;
    private final long vocabSize;
    private final int embedDim;
    /** Shared embedding table name, or null if none. */
    private final String sharedWith;
    /** Padding index, or null if none. */
    private final Long paddingIdx;

    public SparseFeature(String name, long vocabSize) {
        this(name, vocabSize, 8, null, null);
    }

    public SparseFeature(String name, long vocabSize, int embedDim) {
        this(name, vocabSize, embedDim, null, null);
    }

    public SparseFeature(String name, long vocabSize, int embedDim, String sharedWith, Long paddingIdx) {
        this.name = Objects.requireNonNull(name, "name");
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.sharedWith = sharedWith;
        this.paddingIdx = paddingIdx;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public long vocabSize() {
        return vocabSize;
    }

    @Override
    public int embedDim() {
        return embedDim;
    }

    public String sharedWith() {
        return sharedWith;
    }

    public Long paddingIdx() {
        return paddingIdx;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SparseFeature)) return false;
        SparseFeature that = (SparseFeature) o;
        return vocabSize == that.vocabSize
                && embedDim == that.embedDim
                && Objects.equals(name, that.name)
                && Objects.equals(sharedWith, that.sharedWith)
                && Objects.equals(paddingIdx, that.paddingIdx);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, vocabSize, embedDim, sharedWith, paddingIdx);
    }

    @Override
    public String toString() {
        return "SparseFeature{name='" + name + "', vocabSize=" + vocabSize
                + ", embedDim=" + embedDim + ", sharedWith=" + sharedWith
                + ", paddingIdx=" + paddingIdx + '}';
    }
}
