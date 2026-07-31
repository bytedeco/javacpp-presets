/*
 * Optional Kafka ingest adapter — wraps patterns from utils.kafka.KafkaFeatureBridge
 * without hard-depending on a live cluster (row-map oriented for offline demos).
 */
package org.bytedeco.pytorch.feature.bridge;

import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.offline.OfflineStore;
import org.bytedeco.pytorch.feature.offline.PointInTimeJoin;
import org.bytedeco.pytorch.feature.transform.AggregationSpec;
import org.bytedeco.pytorch.feature.transform.FeatureTransform;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Ingest event maps into offline store, optionally applying window aggregations
 * (stream → feature table path used at ByteDance/Alibaba/Uber).
 */
public final class KafkaIngestAdapter {

    private final OfflineStore offlineStore;

    public KafkaIngestAdapter(OfflineStore offlineStore) {
        this.offlineStore = Objects.requireNonNull(offlineStore, "offlineStore");
    }

    /**
     * Append raw events as feature rows for a view (pass-through).
     */
    public int ingestRaw(String project, String viewName, List<Map<String, Object>> events) {
        if (events == null || events.isEmpty()) return 0;
        offlineStore.put(project, viewName, events);
        return events.size();
    }

    /**
     * Apply transform then write results for the view.
     */
    public int ingestTransformed(String project, String viewName,
                                 List<Map<String, Object>> events,
                                 FeatureTransform transform) {
        Objects.requireNonNull(transform, "transform");
        if (events == null || events.isEmpty()) return 0;
        List<Map<String, Object>> out = transform.apply(events);
        offlineStore.put(project, viewName, out);
        return out.size();
    }

    /**
     * Window-aggregate events then write (Feathub-style stream feature materialize sim).
     */
    public int ingestAggregated(FeatureView view, List<Map<String, Object>> events, AggregationSpec agg) {
        Objects.requireNonNull(view, "view");
        Objects.requireNonNull(agg, "agg");
        if (events == null || events.isEmpty()) return 0;
        List<Map<String, Object>> out = agg.apply(events);
        // ensure join keys present
        List<String> keys = view.joinKeys().isEmpty() ? view.entityNames() : view.joinKeys();
        List<Map<String, Object>> normalized = new ArrayList<>(out.size());
        for (Map<String, Object> row : out) {
            Map<String, Object> n = new LinkedHashMap<>(row);
            // drop internal window markers from becoming "features" if not in schema — keep them ok
            for (String k : keys) {
                n.putIfAbsent(k, row.get(k));
            }
            normalized.add(n);
        }
        offlineStore.put(view.project(), view.name(), normalized);
        return normalized.size();
    }

    /**
     * Build synthetic Kafka-like event maps (event_type, entity keys, ts, payload).
     */
    public static List<Map<String, Object>> syntheticClickEvents(int n, long nowMs,
                                                                 String userKey, String itemKey) {
        List<Map<String, Object>> events = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Map<String, Object> e = new LinkedHashMap<>();
            e.put("event_type", i % 5 == 0 ? "expose" : "click");
            e.put(userKey != null ? userKey : "user_id", (long) (i % 100 + 1));
            e.put(itemKey != null ? itemKey : "item_id", (long) (i % 50 + 1));
            e.put("event_timestamp", nowMs - (n - i) * 1000L);
            e.put("value", 1.0);
            events.add(e);
        }
        return events;
    }

    /**
     * Group raw events by entity key string for debugging.
     */
    public static Map<String, List<Map<String, Object>>> groupByEntity(
            List<Map<String, Object>> events, List<String> joinKeys) {
        Map<String, List<Map<String, Object>>> out = new LinkedHashMap<>();
        if (events == null) return out;
        for (Map<String, Object> e : events) {
            String k = PointInTimeJoin.entityKey(e, joinKeys);
            out.computeIfAbsent(k, x -> new ArrayList<>()).add(e);
        }
        return out;
    }
}
