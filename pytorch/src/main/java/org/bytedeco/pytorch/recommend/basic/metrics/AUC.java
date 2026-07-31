/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala
 * (AUC, LogLoss, Accuracy, HitRate, NDCG, MRR, MSE, RMSE, MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

import java.util.Arrays;

/** Area Under the ROC Curve. */
public class AUC implements Metric {
    private double pairs = 0.0;
    private int posCount = 0;
    private int negCount = 0;
    private final int posLabel;

    public AUC() {
        this(1);
    }

    public AUC(int posLabel) {
        this.posLabel = posLabel;
    }

    @Override
    public String name() {
        return "AUC";
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        int n = predictions.length;
        Integer[] sortedIndices = new Integer[n];
        for (int i = 0; i < n; i++) sortedIndices[i] = i;
        Arrays.sort(sortedIndices, (a, b) -> Float.compare(predictions[b], predictions[a]));

        int negAfter = 0;
        int batchPos = 0;
        int batchNeg = 0;
        for (int i = n - 1; i >= 0; i--) {
            int idx = sortedIndices[i];
            if (labels[idx] <= 0.5f) {
                negAfter += 1;
                batchNeg += 1;
            } else {
                pairs += negAfter;
                batchPos += 1;
            }
        }
        posCount += batchPos;
        negCount += batchNeg;
    }

    @Override
    public float compute() {
        if (posCount == 0 || negCount == 0) return 0.5f;
        return (float) (pairs / ((double) posCount * negCount));
    }

    @Override
    public void reset() {
        pairs = 0.0;
        posCount = 0;
        negCount = 0;
    }

    public static float calculate(float[] predictions, float[] labels) {
        AUC metric = new AUC();
        metric.update(predictions, labels);
        return metric.compute();
    }
}
