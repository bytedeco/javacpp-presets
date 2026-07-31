/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala (RMSE, MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

/** RMSE (Root Mean Squared Error). */
public class RMSE extends MSE {
    @Override
    public String name() {
        return "RMSE";
    }

    @Override
    public float compute() {
        return (float) Math.sqrt(super.compute());
    }
}
