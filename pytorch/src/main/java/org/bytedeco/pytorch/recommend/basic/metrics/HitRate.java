/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala (HitRate, NDCG, MRR)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

import java.util.Arrays;

/** Top-K Hit Rate. */
public class HitRate implements Metric {
    private int hits = 0;
    private int total = 0;
    private final int k;

    public HitRate(int k) {
        this.k = k;
    }

    @Override
    public String name() {
        return "Hit@" + k;
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        int n = predictions.length;
        if (n == 0) return;
        Integer[] sortedIndices = new Integer[n];
        for (int i = 0; i < n; i++) sortedIndices[i] = i;
        Arrays.sort(sortedIndices, (a, b) -> Float.compare(predictions[b], predictions[a]));
        int top = Math.min(k, n);
        boolean found = false;
        for (int i = 0; i < top && !found; i++) {
            if (labels[sortedIndices[i]] > 0.5f) found = true;
        }
        total += 1;
        if (found) hits += 1;
    }

    @Override
    public float compute() {
        return total == 0 ? 0.0f : (float) hits / total;
    }

    @Override
    public void reset() {
        hits = 0;
        total = 0;
    }
}
