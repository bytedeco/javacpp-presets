/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala
 * (Accuracy, HitRate, NDCG, MRR, MSE, RMSE, MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

/** Accuracy metric. */
public class Accuracy implements Metric {
    private int correct = 0;
    private int total = 0;
    private final float threshold;

    public Accuracy() {
        this(0.5f);
    }

    public Accuracy(float threshold) {
        this.threshold = threshold;
    }

    @Override
    public String name() {
        return "Accuracy";
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        if (predictions.length != labels.length) {
            throw new IllegalArgumentException("predictions and labels length mismatch");
        }
        for (int i = 0; i < predictions.length; i++) {
            float pred = predictions[i] > threshold ? 1.0f : 0.0f;
            if (pred == labels[i]) correct += 1;
            total += 1;
        }
    }

    @Override
    public float compute() {
        return total == 0 ? 0.0f : (float) correct / total;
    }

    @Override
    public void reset() {
        correct = 0;
        total = 0;
    }
}
