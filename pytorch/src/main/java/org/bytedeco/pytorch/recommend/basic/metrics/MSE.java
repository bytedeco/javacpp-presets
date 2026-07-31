/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala
 * (MSE, RMSE, MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

/** MSE (Mean Squared Error). */
public class MSE implements Metric {
    protected double totalSqError = 0.0;
    protected int count = 0;

    @Override
    public String name() {
        return "MSE";
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        if (predictions.length != labels.length) {
            throw new IllegalArgumentException("predictions and labels length mismatch");
        }
        for (int i = 0; i < predictions.length; i++) {
            double diff = predictions[i] - labels[i];
            totalSqError += diff * diff;
            count += 1;
        }
    }

    @Override
    public float compute() {
        return count == 0 ? 0.0f : (float) (totalSqError / count);
    }

    @Override
    public void reset() {
        totalSqError = 0.0;
        count = 0;
    }
}
