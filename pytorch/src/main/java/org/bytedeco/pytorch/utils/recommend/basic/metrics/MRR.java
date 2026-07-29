/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala
 * (MRR, MSE, RMSE, MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.utils.recommend.basic.metrics;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/** MRR (Mean Reciprocal Rank). */
public class MRR implements Metric {
    private double totalRR = 0.0;
    private int count = 0;

    @Override
    public String name() {
        return "MRR";
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        count += 1;
        int n = predictions.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(predictions[b], predictions[a]));
        for (int i = 0; i < n; i++) {
            if (labels[order[i]] > 0) {
                totalRR += 1.0 / (i + 1);
                break;
            }
        }
    }

    @Override
    public float compute() {
        return count == 0 ? 0.0f : (float) (totalRR / count);
    }

    @Override
    public void reset() {
        totalRR = 0.0;
        count = 0;
    }
}
