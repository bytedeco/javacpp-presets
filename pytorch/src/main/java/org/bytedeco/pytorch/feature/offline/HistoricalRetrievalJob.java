/*
 * Historical feature retrieval job — Feast get_historical_features.
 * Loads offline rows for each view in a FeatureService and runs PIT join.
 */
package org.bytedeco.pytorch.feature.offline;

import org.bytedeco.pytorch.feature.core.FeatureService;
import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;

import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Offline historical retrieval orchestrator. */
public final class HistoricalRetrievalJob {

    private final FeatureRegistry registry;
    private final OfflineStore offlineStore;
    private final PointInTimeJoin.Options joinOptions;

    public HistoricalRetrievalJob(FeatureRegistry registry, OfflineStore offlineStore) {
        this(registry, offlineStore, new PointInTimeJoin.Options());
    }

    public HistoricalRetrievalJob(FeatureRegistry registry, OfflineStore offlineStore,
                                  PointInTimeJoin.Options joinOptions) {
        this.registry = Objects.requireNonNull(registry, "registry");
        this.offlineStore = Objects.requireNonNull(offlineStore, "offlineStore");
        this.joinOptions = joinOptions != null ? joinOptions : new PointInTimeJoin.Options();
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 FeatureService service) {
        return getHistoricalFeatures(entityRows, service, null, null);
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 FeatureService service,
                                                 String labelColumn,
                                                 String schemaVersion) {
        Objects.requireNonNull(entityRows, "entityRows");
        Objects.requireNonNull(service, "service");
        List<FeatureView> views = registry.resolveViews(service);
        return getHistoricalFeatures(entityRows, views, labelColumn, schemaVersion);
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 List<FeatureView> views) {
        return getHistoricalFeatures(entityRows, views, null, null);
    }

    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 List<FeatureView> views,
                                                 String labelColumn,
                                                 String schemaVersion) {
        Objects.requireNonNull(entityRows, "entityRows");
        Objects.requireNonNull(views, "views");

        // Determine time range from entity event timestamps to bound offline reads
        long minTs = Long.MAX_VALUE;
        long maxTs = Long.MIN_VALUE;
        String tsCol = joinOptions.eventTimestampColumn;
        for (Map<String, Object> e : entityRows) {
            long t = FileOfflineStore.toEpochMillis(e.get(tsCol));
            if (t < minTs) minTs = t;
            if (t > maxTs) maxTs = t;
        }
        if (minTs == Long.MAX_VALUE) {
            minTs = 0L;
            maxTs = System.currentTimeMillis();
        }
        // Expand lower bound by max TTL so as-of rows are available
        long maxTtl = 0L;
        for (FeatureView v : views) {
            maxTtl = Math.max(maxTtl, v.ttlMillis());
        }
        // If TTL is 0 (infinite), look back 10 years for practicality in file stores
        long lookback = maxTtl > 0 ? maxTtl : 3650L * 24 * 3600 * 1000;
        Instant start = Instant.ofEpochMilli(Math.max(0, minTs - lookback));
        Instant end = Instant.ofEpochMilli(maxTs);

        Map<String, List<Map<String, Object>>> byView = new LinkedHashMap<>();
        for (FeatureView view : views) {
            List<Map<String, Object>> rows = offlineStore.readRange(view, start, end);
            byView.put(view.name(), rows);
        }

        PointInTimeJoin.Result result = PointInTimeJoin.joinMany(entityRows, byView, views, joinOptions);

        Set<String> entityKeys = new LinkedHashSet<>();
        for (FeatureView v : views) {
            entityKeys.addAll(v.joinKeys().isEmpty() ? v.entityNames() : v.joinKeys());
        }
        // Also keep keys present on entity rows that look like ids
        if (!entityRows.isEmpty()) {
            for (String k : entityRows.get(0).keySet()) {
                if (k.endsWith("_id") || k.equals("user_id") || k.equals("item_id")) {
                    entityKeys.add(k);
                }
            }
        }

        String schema = schemaVersion != null ? schemaVersion : ("hist-" + views.size() + "v");
        return TrainingDataset.fromJoinResult(result, new ArrayList<>(entityKeys), labelColumn, schema);
    }

    /**
     * Convenience: retrieve for explicit view names in a project.
     */
    public TrainingDataset getHistoricalFeatures(List<Map<String, Object>> entityRows,
                                                 String project,
                                                 List<String> viewNames,
                                                 String labelColumn) {
        List<FeatureView> views = new ArrayList<>();
        for (String vn : viewNames) {
            views.add(registry.requireFeatureView(project, vn));
        }
        return getHistoricalFeatures(entityRows, views, labelColumn, null);
    }
}
