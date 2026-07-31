/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala
 *
 * Streaming metric interfaces: AUC, LogLoss, Accuracy, HitRate, NDCG, MRR, MSE, RMSE, MAE,
 * and MetricRegistry.
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Base interface for evaluation metrics. */
public interface Metric {
    String name();
    void update(float[] predictions, float[] labels);
    float compute();
    void reset();
}
