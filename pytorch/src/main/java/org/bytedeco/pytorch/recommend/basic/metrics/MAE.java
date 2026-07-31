/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala (MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

/** MAE (Mean Absolute Error). */
public class MAE implements Metric {
    private double totalAbsError = 0.0;
    private int count = 0;

    @Override
    public String name() {
        return "MAE";
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        if (predictions.length != labels.length) {
            throw new IllegalArgumentException("predictions and labels length mismatch");
        }
        for (int i = 0; i < predictions.length; i++) {
            totalAbsError += Math.abs(predictions[i] - labels[i]);
            count += 1;
        }
    }

    @Override
    public float compute() {
        return count == 0 ? 0.0f : (float) (totalAbsError / count);
    }

    @Override
    public void reset() {
        totalAbsError = 0.0;
        count = 0;
    }
}
