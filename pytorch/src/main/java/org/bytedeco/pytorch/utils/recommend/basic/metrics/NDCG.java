/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala
 * (NDCG, MRR, MSE, RMSE, MAE, MetricRegistry)
 */
package org.bytedeco.pytorch.utils.recommend.basic.metrics;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/** NDCG@K (Normalized Discounted Cumulative Gain). */
public class NDCG implements Metric {
    private double totalNDCG = 0.0;
    private int count = 0;
    private final int k;

    public NDCG(int k) {
        this.k = k;
    }

    @Override
    public String name() {
        return "NDCG@" + k;
    }

    @Override
    public void update(float[] predictions, float[] labels) {
        count += 1;
        int n = predictions.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Float.compare(predictions[b], predictions[a]));

        double dcg = 0.0;
        int take = Math.min(k, n);
        for (int idx = 0; idx < take; idx++) {
            float rel = labels[order[idx]];
            if (rel > 0) {
                dcg += 1.0 / (Math.log(idx + 2) / Math.log(2));
            }
        }

        float[] ideal = labels.clone();
        Arrays.sort(ideal);
        // reverse (descending)
        for (int i = 0, j = ideal.length - 1; i < j; i++, j--) {
            float tmp = ideal[i];
            ideal[i] = ideal[j];
            ideal[j] = tmp;
        }
        double idcg = 0.0;
        int idealTake = Math.min(k, ideal.length);
        for (int idx = 0; idx < idealTake; idx++) {
            if (ideal[idx] > 0) {
                idcg += 1.0 / (Math.log(idx + 2) / Math.log(2));
            }
        }

        totalNDCG += (idcg > 0) ? dcg / idcg : 0.0;
    }

    @Override
    public float compute() {
        return count == 0 ? 0.0f : (float) (totalNDCG / count);
    }

    @Override
    public void reset() {
        totalNDCG = 0.0;
        count = 0;
    }
}
