/*
 * Ported from torch-rechub-scala: torchrec/basic/metrics/Metric.scala (MetricRegistry)
 */
package org.bytedeco.pytorch.recommend.basic.metrics;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/** Metric registry for managing multiple metrics. */
public class MetricRegistry {
    private final Map<String, Metric> metrics = new HashMap<>();

    public MetricRegistry register(String name, Metric metric) {
        metrics.put(name, metric);
        return this;
    }

    public void update(float[] predictions, float[] labels) {
        for (Metric m : metrics.values()) {
            m.update(predictions, labels);
        }
    }

    public Map<String, Float> compute() {
        Map<String, Float> result = new LinkedHashMap<>();
        for (Map.Entry<String, Metric> e : metrics.entrySet()) {
            result.put(e.getKey(), e.getValue().compute());
        }
        return result;
    }

    public void reset() {
        for (Metric m : metrics.values()) {
            m.reset();
        }
    }

    public Metric get(String name) {
        return metrics.get(name);
    }

    public Set<String> names() {
        return metrics.keySet();
    }
}
