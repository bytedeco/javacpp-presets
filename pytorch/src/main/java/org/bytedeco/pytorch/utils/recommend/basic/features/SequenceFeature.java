/*
 * Ported from torch-rechub-scala: torchrec/basic/features/Feature.scala (SequenceFeature)
 */
package org.bytedeco.pytorch.utils.recommend.basic.features;

import java.util.Objects;

/**
 * Sequence feature for behavior history (e.g., clicked items).
 */
public final class SequenceFeature implements Feature {
    private final String name;
    private final long vocabSize;
    private final int embedDim;
    /** Pooling strategy: "mean", "sum", "concat", "last". */
    private final String pooling;
    /** Shared embedding table name, or null if none. */
    private final String sharedWith;
    private final int maxLen;
    private final long paddingIdx;

    public SequenceFeature(String name, long vocabSize) {
        this(name, vocabSize, 8, "mean", null, 50, 0L);
    }

    public SequenceFeature(String name, long vocabSize, int embedDim) {
        this(name, vocabSize, embedDim, "mean", null, 50, 0L);
    }

    public SequenceFeature(String name, long vocabSize, int embedDim, String pooling) {
        this(name, vocabSize, embedDim, pooling, null, 50, 0L);
    }

    public SequenceFeature(String name, long vocabSize, int embedDim, String pooling,
                           String sharedWith, int maxLen, long paddingIdx) {
        this.name = Objects.requireNonNull(name, "name");
        this.vocabSize = vocabSize;
        this.embedDim = embedDim;
        this.pooling = pooling != null ? pooling : "mean";
        this.sharedWith = sharedWith;
        this.maxLen = maxLen;
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

    public String pooling() {
        return pooling;
    }

    public String sharedWith() {
        return sharedWith;
    }

    public int maxLen() {
        return maxLen;
    }

    public long paddingIdx() {
        return paddingIdx;
    }

    @Override
    public boolean isSequence() {
        return true;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SequenceFeature)) return false;
        SequenceFeature that = (SequenceFeature) o;
        return vocabSize == that.vocabSize
                && embedDim == that.embedDim
                && maxLen == that.maxLen
                && paddingIdx == that.paddingIdx
                && Objects.equals(name, that.name)
                && Objects.equals(pooling, that.pooling)
                && Objects.equals(sharedWith, that.sharedWith);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, vocabSize, embedDim, pooling, sharedWith, maxLen, paddingIdx);
    }

    @Override
    public String toString() {
        return "SequenceFeature{name='" + name + "', vocabSize=" + vocabSize
                + ", embedDim=" + embedDim + ", pooling='" + pooling
                + "', sharedWith=" + sharedWith + ", maxLen=" + maxLen
                + ", paddingIdx=" + paddingIdx + '}';
    }
}
