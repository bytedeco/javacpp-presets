package org.bytedeco.pytorch.geometric.metrics;

import java.util.HashMap;
import java.util.Map;

public class MetricManager {
    private final Map<String, Double> runningSums = new HashMap<>();
    private long totalSamples = 0;

    public void update(Map<String, Double> batchResults, long batchSize) {
        for (Map.Entry<String, Double> entry : batchResults.entrySet()) {
            runningSums.merge(entry.getKey(), entry.getValue() * batchSize, Double::sum);
        }
        totalSamples += batchSize;
    }

    public Map<String, Double> finalizeMetrics() {
        Map<String, Double> finalResults = new HashMap<>();
        for (Map.Entry<String, Double> entry : runningSums.entrySet()) {
            finalResults.put(entry.getKey(), entry.getValue() / totalSamples);
        }
        return finalResults;
    }
}