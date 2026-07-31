/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala
 * (LogLoss, Accuracy, HitRate, NDCG, MRR, MSE, RMSE, MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

/** Log Loss (Binary Cross Entropy). */
public class LogLoss implements Metric {
    private double totalLoss = 0.0;
    private int count = 0;

    @Override
    public String name() {
        return "LogLoss";
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        if (predictions.length != labels.length) {
            throw new IllegalArgumentException("predictions and labels length mismatch");
        }
        for (int i = 0; i < predictions.length; i++) {
            float raw = predictions[i];
            float p;
            if (Float.isNaN(raw) || Float.isInfinite(raw)) {
                p = 0.5f;
            } else {
                p = (float) Math.max(1e-7, Math.min(1 - 1e-7, raw));
            }
            float y = labels[i];
            totalLoss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
            count += 1;
        }
    }

    @Override
    public float compute() {
        return count == 0 ? 0.0f : (float) (totalLoss / count);
    }

    @Override
    public void reset() {
        totalLoss = 0.0;
        count = 0;
    }

    public static float calculate(float[] predictions, float[] labels) {
        LogLoss metric = new LogLoss();
        metric.update(predictions, labels);
        return metric.compute();
    }
}
