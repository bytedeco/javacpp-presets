package org.bytedeco.pytorch.dataframe.ml.math;

import java.util.Random;

/**
 * Lightweight index sampler for ML clustering / mini-batch algorithms.
 * Not related to torch::data::samplers::RandomSampler.
 */
public final class RandomSampler {
    private RandomSampler() {}

    public static int[] sampleIndices(int n, int k, Random rng) {
        if (n <= 0) throw new IllegalArgumentException("n must be > 0");
        if (k < 0) throw new IllegalArgumentException("k must be >= 0");
        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = rng.nextInt(n);
        return out;
    }
}
