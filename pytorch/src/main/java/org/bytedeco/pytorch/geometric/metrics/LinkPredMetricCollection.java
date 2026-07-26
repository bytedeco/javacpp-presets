package org.bytedeco.pytorch.geometric.metrics;

import org.bytedeco.pytorch.Tensor;

import java.util.HashMap;
import java.util.Map;

public class LinkPredMetricCollection {
    private final Map<String, LinkPredMetric> metrics = new HashMap<>();

    public void addMetric(String name, LinkPredMetric metric) {
        metrics.put(name, metric);
    }

    public Map<String, Double> computeAll(Tensor yPred, Tensor yTrue) {
        Map<String, Double> results = new HashMap<>();
        for (Map.Entry<String, LinkPredMetric> entry : metrics.entrySet()) {
            Tensor res = entry.getValue().compute(yPred, yTrue);
            results.put(entry.getKey(), res.mean().item().toDouble());
        }
        return results;
    }
}