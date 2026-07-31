/*
 * Ported from torch-rechub-scala: torchrec/basic/features/Feature.scala (LabelFeature)
 */
package org.bytedeco.pytorch.recommend.basic.features;

import java.util.Objects;

/**
 * Label feature for supervised learning.
 */
public final class LabelFeature implements Feature {
    private final String name;

    public LabelFeature() {
        this("label");
    }

    public LabelFeature(String name) {
        this.name = Objects.requireNonNull(name, "name");
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public int embedDim() {
        return 1;
    }

    @Override
    public long vocabSize() {
        return 2L;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof LabelFeature)) return false;
        LabelFeature that = (LabelFeature) o;
        return Objects.equals(name, that.name);
    }

    @Override
    public int hashCode() {
        return Objects.hash(name);
    }

    @Override
    public String toString() {
        return "LabelFeature{name='" + name + "'}";
    }
}
