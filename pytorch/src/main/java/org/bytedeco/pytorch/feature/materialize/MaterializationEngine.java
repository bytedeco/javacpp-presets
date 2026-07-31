/*
 * Materialization engine — offline latest-per-entity → online write.
 *
 * Feast materialize / Databricks feature table sync pattern used at
 * Meta, Google, Alibaba, ByteDance for online feature freshness.
 */
package org.bytedeco.pytorch.feature.materialize;

import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.offline.FileOfflineStore;
import org.bytedeco.pytorch.feature.offline.OfflineStore;
import org.bytedeco.pytorch.feature.offline.PointInTimeJoin;
import org.bytedeco.pytorch.feature.online.OnlineFeatureRow;
import org.bytedeco.pytorch.feature.online.OnlineStore;
import org.bytedeco.pytorch.feature.online.OnlineWriteBatch;
import org.bytedeco.pytorch.feature.registry.FeatureRegistry;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Orchestrates offline → online materialization. */
public final class MaterializationEngine {

    private final FeatureRegistry registry;
    private final OfflineStore offlineStore;
    private final OnlineStore onlineStore;
    private final IncrementalCursor cursor;

    public MaterializationEngine(FeatureRegistry registry,
                                 OfflineStore offlineStore,
                                 OnlineStore onlineStore) {
        this(registry, offlineStore, onlineStore, new IncrementalCursor());
    }

    public MaterializationEngine(FeatureRegistry registry,
                                 OfflineStore offlineStore,
                                 OnlineStore onlineStore,
                                 IncrementalCursor cursor) {
        this.registry = Objects.requireNonNull(registry, "registry");
        this.offlineStore = Objects.requireNonNull(offlineStore, "offlineStore");
        this.onlineStore = Objects.requireNonNull(onlineStore, "onlineStore");
        this.cursor = cursor != null ? cursor : new IncrementalCursor();
    }

    public IncrementalCursor cursor() {
        return cursor;
    }

    public MaterializationResult materialize(MaterializationJob job) {
        Objects.requireNonNull(job, "job");
        long t0 = System.nanoTime();
        long startWall = System.currentTimeMillis();
        MaterializationResult.Builder result = MaterializationResult.builder()
                .jobId(job.jobId())
                .startMs(startWall);

        List<String> viewNames = new ArrayList<>();
        long rowsRead = 0;
        long rowsWritten = 0;
        long entitiesTouched = 0;
        long globalWatermark = 0L;

        try {
            for (FeatureView view : job.views()) {
                if (!view.online()) continue;
                viewNames.add(view.name());

                Instant start = job.start();
                if (job.incremental()) {
                    long wm = cursor.get(view.project(), view.name());
                    if (wm > 0) {
                        // exclusive of last watermark → start just after
                        start = Instant.ofEpochMilli(wm + 1);
                    }
                }
                if (start == null) start = Instant.EPOCH;
                Instant end = job.end() != null ? job.end() : Instant.now();

                List<Map<String, Object>> rows = offlineStore.readRange(view, start, end);
                rowsRead += rows.size();

                String tsCol = view.source() != null ? view.source().timestampColumn()
                        : PointInTimeJoin.DEFAULT_EVENT_TS;
                List<String> joinKeys = view.joinKeys().isEmpty() ? view.entityNames() : view.joinKeys();

                // latest row per entity key within the window
                Map<String, Map<String, Object>> latest = new HashMap<>();
                Map<String, Long> latestTs = new HashMap<>();
                long viewMaxTs = 0L;
                for (Map<String, Object> row : rows) {
                    String ek = PointInTimeJoin.entityKey(row, joinKeys);
                    long ts = FileOfflineStore.toEpochMillis(row.get(tsCol));
                    if (ts > viewMaxTs) viewMaxTs = ts;
                    Long prev = latestTs.get(ek);
                    if (prev == null || ts >= prev) {
                        latestTs.put(ek, ts);
                        latest.put(ek, row);
                    }
                }

                List<OnlineFeatureRow> onlineRows = new ArrayList<>(latest.size());
                for (Map.Entry<String, Map<String, Object>> e : latest.entrySet()) {
                    Map<String, Object> src = e.getValue();
                    Map<String, Object> values = new LinkedHashMap<>();
                    for (String fn : view.featureNames()) {
                        values.put(fn, src.get(fn));
                    }
                    long ts = latestTs.getOrDefault(e.getKey(), 0L);
                    onlineRows.add(OnlineFeatureRow.builder(view.name(), e.getKey())
                            .project(view.project())
                            .values(values)
                            .eventTimestampMs(ts)
                            .ttlMs(view.ttlMillis())
                            .build());
                }

                if (!onlineRows.isEmpty()) {
                    onlineStore.onlineWrite(OnlineWriteBatch.of(onlineRows));
                }
                rowsWritten += onlineRows.size();
                entitiesTouched += onlineRows.size();
                result.perViewWritten(view.name(), onlineRows.size());

                if (viewMaxTs > 0) {
                    cursor.advance(view.project(), view.name(), viewMaxTs);
                    globalWatermark = Math.max(globalWatermark, viewMaxTs);
                }
            }

            long t1 = System.nanoTime();
            return result
                    .viewNames(viewNames)
                    .rowsRead(rowsRead)
                    .rowsWritten(rowsWritten)
                    .entitiesTouched(entitiesTouched)
                    .endMs(System.currentTimeMillis())
                    .elapsedNanos(t1 - t0)
                    .success(true)
                    .watermarkMs(globalWatermark)
                    .build();
        } catch (RuntimeException ex) {
            long t1 = System.nanoTime();
            return result
                    .viewNames(viewNames)
                    .rowsRead(rowsRead)
                    .rowsWritten(rowsWritten)
                    .entitiesTouched(entitiesTouched)
                    .endMs(System.currentTimeMillis())
                    .elapsedNanos(t1 - t0)
                    .success(false)
                    .error(ex.getMessage() != null ? ex.getMessage() : ex.getClass().getSimpleName())
                    .watermarkMs(globalWatermark)
                    .build();
        }
    }

    /** Materialize all online views in a project for [start, end]. */
    public MaterializationResult materializeProject(String project, Instant start, Instant end) {
        List<FeatureView> views = registry.listFeatureViews(project);
        List<FeatureView> online = new ArrayList<>();
        for (FeatureView v : views) {
            if (v.online()) online.add(v);
        }
        // also stream views as batch
        registry.listStreamFeatureViews(project).forEach(sfv -> {
            if (sfv.online()) online.add(sfv.asBatchView());
        });
        MaterializationJob job = MaterializationJob.builder()
                .project(project)
                .views(online)
                .start(start != null ? start : Instant.EPOCH)
                .end(end != null ? end : Instant.now())
                .incremental(false)
                .build();
        return materialize(job);
    }

    public MaterializationResult materializeViews(List<FeatureView> views, Instant start, Instant end) {
        MaterializationJob job = MaterializationJob.builder()
                .views(views)
                .start(start != null ? start : Instant.EPOCH)
                .end(end != null ? end : Instant.now())
                .build();
        return materialize(job);
    }

    public MaterializationResult materializeIncremental(List<FeatureView> views, Instant end) {
        MaterializationJob job = MaterializationJob.builder()
                .views(views)
                .end(end != null ? end : Instant.now())
                .incremental(true)
                .build();
        return materialize(job);
    }
}
