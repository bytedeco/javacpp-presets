/*
 * Ported from torch-rechub-scala: torchrec/basic/features/Feature.scala (DenseFeature)
 */
package org.bytedeco.pytorch.utils.recommend.basic.features;

import java.util.Objects;

/**
 * Dense (numeric) feature - passes through without embedding.
 */
public final class DenseFeature implements Feature {
    private final String name;
    private final int embedDim;

    public DenseFeature(String name) {
        this(name, 1);
    }

    public DenseFeature(String name, int embedDim) {
        this.name = Objects.requireNonNull(name, "name");
        this.embedDim = embedDim;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public int embedDim() {
        return embedDim;
    }

    @Override
    public long vocabSize() {
        return 1L;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof DenseFeature)) return false;
        DenseFeature that = (DenseFeature) o;
        return embedDim == that.embedDim && Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name, embedDim);
    }

    @Override
    public String toString() {
        return "DenseFeature{name='" + name + "', embedDim=" + embedDim + '}';
    }
}
