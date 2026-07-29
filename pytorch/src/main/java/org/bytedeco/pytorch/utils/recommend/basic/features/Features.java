/*
 * Ported from torch-rechub-scala: torchrec/basic/features/Feature.scala (object Features)
 */
package org.bytedeco.pytorch.utils.recommend.basic.features;

import java.util.ArrayList;
import java.util.List;

/**
 * Helper for feature creation and manipulation.
 */
public final class Features {

    private Features() {}

    public static SparseFeature sparse(String name, long vocabSize) {
        return new SparseFeature(name, vocabSize, 8);
    }

    public static SparseFeature sparse(String name, long vocabSize, int embedDim) {
        return new SparseFeature(name, vocabSize, embedDim);
    }

    public static DenseFeature dense(String name) {
        return new DenseFeature(name, 1);
    }

    public static DenseFeature dense(String name, int embedDim) {
        return new DenseFeature(name, embedDim);
    }

    public static SequenceFeature sequence(String name, long vocabSize) {
        return new SequenceFeature(name, vocabSize, 8, "mean");
    }

    public static SequenceFeature sequence(String name, long vocabSize, int embedDim) {
        return new SequenceFeature(name, vocabSize, embedDim, "mean");
    }

    public static SequenceFeature sequence(String name, long vocabSize, int embedDim, String pooling) {
        return new SequenceFeature(name, vocabSize, embedDim, pooling);
    }

    /** Get all sparse features from a list */
    public static List<SparseFeature> getSparseFeatures(List<? extends Feature> features) {
        List<SparseFeature> out = new ArrayList<>();
        for (Feature f : features) {
            if (f instanceof SparseFeature) {
                out.add((SparseFeature) f);
            }
        }
        return out;
    }

    /** Get all dense features from a list */
    public static List<DenseFeature> getDenseFeatures(List<? extends Feature> features) {
        List<DenseFeature> out = new ArrayList<>();
        for (Feature f : features) {
            if (f instanceof DenseFeature) {
                out.add((DenseFeature) f);
            }
        }
        return out;
    }

    /** Get all sequence features from a list */
    public static List<SequenceFeature> getSequenceFeatures(List<? extends Feature> features) {
        List<SequenceFeature> out = new ArrayList<>();
        for (Feature f : features) {
            if (f instanceof SequenceFeature) {
                out.add((SequenceFeature) f);
            }
        }
        return out;
    }

    /**
     * Calculate total embedding dimension for sparse features.
     * All SparseFeatures must have the same embedDim; uses per-feature f.embedDim.
     */
    public static long calcSparseDim(List<? extends Feature> features) {
        long sum = 0L;
        for (SparseFeature f : getSparseFeatures(features)) {
            sum += f.embedDim();
        }
        return sum;
    }

    /**
     * Calculate total embedding dimension for sequence features with pooling.
     * Uses per-feature f.embedDim for concat pooling.
     */
    public static long calcSequenceDim(List<SequenceFeature> features, String pooling) {
        List<SequenceFeature> seq = features;
        if ("concat".equals(pooling)) {
            long sum = 0L;
            for (SequenceFeature f : seq) {
                sum += f.embedDim();
            }
            return sum;
        }
        if (seq.isEmpty()) {
            return 0L;
        }
        return seq.get(0).embedDim();
    }

    public static long calcSequenceDim(List<SequenceFeature> features) {
        return calcSequenceDim(features, "mean");
    }

    /** Calculate total embedding dimension for sequence features (from mixed Feature list). */
    public static long calcSequenceDimFromFeatures(List<? extends Feature> features, String pooling) {
        return calcSequenceDim(getSequenceFeatures(features), pooling);
    }

    public static long calcSequenceDimFromFeatures(List<? extends Feature> features) {
        return calcSequenceDimFromFeatures(features, "mean");
    }
}
