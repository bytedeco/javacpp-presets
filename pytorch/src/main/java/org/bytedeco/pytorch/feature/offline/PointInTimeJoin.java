/*
 * Point-in-time (as-of) join — the correctness core of every industrial feature store.
 *
 * Algorithm (Feast / Uber Michelangelo / Databricks / Meta training pipelines):
 *   For each entity row with event_timestamp T and join keys K:
 *     For each FeatureView V:
 *       Select feature rows matching K where feature_ts <= T
 *         and (TTL==0 or T - feature_ts <= TTL)
 *       Pick the row with max feature_ts (latest as-of T)
 *       Attach V's feature columns (optionally prefixed)
 *
 * MUST NOT leak future features (feature_ts > event_ts) — top production incident class.
 */
package org.bytedeco.pytorch.feature.offline;

import org.bytedeco.pytorch.feature.core.FeatureView;
import org.bytedeco.pytorch.feature.core.Field;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Pure point-in-time join engine over in-memory row tables. */
public final class PointInTimeJoin {

    public static final String DEFAULT_EVENT_TS = "event_timestamp";
    public static final String DEFAULT_PREFIX_SEP = "__";

    public static final class Options {
        public String eventTimestampColumn = DEFAULT_EVENT_TS;
        public boolean prefixWithViewName = true;
        public String prefixSeparator = DEFAULT_PREFIX_SEP;
        public boolean includeEventTimestamp = true;
        /** When true, missing features yield null columns rather than omitting keys. */
        public boolean emitNullsForMissing = true;

        public Options eventTimestampColumn(String c) {
            this.eventTimestampColumn = c;
            return this;
        }

        public Options prefixWithViewName(boolean v) {
            this.prefixWithViewName = v;
            return this;
        }

        public Options emitNullsForMissing(boolean v) {
            this.emitNullsForMissing = v;
            return this;
        }
    }

    public static final class JoinStats {
        public long entityRows;
        public long featureRowsScanned;
        public long joinsHit;
        public long joinsMiss;
        public long futureRowsRejected;
        public long ttlExpiredRejected;
        public long elapsedNanos;

        @Override
        public String toString() {
            return "JoinStats{entities=" + entityRows
                    + ", scanned=" + featureRowsScanned
                    + ", hit=" + joinsHit
                    + ", miss=" + joinsMiss
                    + ", futureRejected=" + futureRowsRejected
                    + ", ttlRejected=" + ttlExpiredRejected
                    + ", ms=" + (elapsedNanos / 1_000_000.0)
                    + "}";
        }
    }

    public static final class Result {
        public final List<Map<String, Object>> rows;
        public final JoinStats stats;
        public final List<String> outputColumns;

        public Result(List<Map<String, Object>> rows, JoinStats stats, List<String> outputColumns) {
            this.rows = rows;
            this.stats = stats;
            this.outputColumns = outputColumns;
        }
    }

    private PointInTimeJoin() {}

    /**
     * Join entity dataframe-like rows with one feature view's historical rows.
     *
     * @param entityRows      rows with join keys + event timestamp (+ labels)
     * @param featureRows     historical feature rows with join keys + feature timestamp + features
     * @param view            feature view (entities, schema, ttl, timestamp column name via source)
     * @param options         join options
     */
    public static Result joinOne(List<Map<String, Object>> entityRows,
                                 List<Map<String, Object>> featureRows,
                                 FeatureView view,
                                 Options options) {
        Objects.requireNonNull(entityRows, "entityRows");
        Objects.requireNonNull(featureRows, "featureRows");
        Objects.requireNonNull(view, "view");
        Options opt = options != null ? options : new Options();
        JoinStats stats = new JoinStats();
        long t0 = System.nanoTime();

        String featureTsCol = view.source() != null && view.source().timestampColumn() != null
                ? view.source().timestampColumn()
                : DEFAULT_EVENT_TS;
        List<String> joinKeys = view.joinKeys();
        if (joinKeys.isEmpty()) {
            // fall back to entity names
            joinKeys = view.entityNames();
        }
        long ttlMs = view.ttlMillis();
        List<String> featureNames = view.featureNames();

        // Index feature rows by entity key string for O(F + E*bucket) instead of O(E*F)
        Map<String, List<Map<String, Object>>> byKey = new HashMap<>();
        for (Map<String, Object> fr : featureRows) {
            String k = entityKey(fr, joinKeys);
            byKey.computeIfAbsent(k, x -> new ArrayList<>()).add(fr);
        }
        // Sort each bucket by feature ts ascending for binary-search style latest-as-of
        for (List<Map<String, Object>> bucket : byKey.values()) {
            bucket.sort(Comparator.comparingLong(r -> FileOfflineStore.toEpochMillis(r.get(featureTsCol))));
        }

        List<String> outputColumns = new ArrayList<>();
        if (!entityRows.isEmpty()) {
            outputColumns.addAll(entityRows.get(0).keySet());
        }
        for (String fn : featureNames) {
            String col = opt.prefixWithViewName ? view.name() + opt.prefixSeparator + fn : fn;
            if (!outputColumns.contains(col)) outputColumns.add(col);
        }

        List<Map<String, Object>> out = new ArrayList<>(entityRows.size());
        stats.entityRows = entityRows.size();

        for (Map<String, Object> entity : entityRows) {
            Map<String, Object> row = new LinkedHashMap<>(entity);
            long eventTs = FileOfflineStore.toEpochMillis(entity.get(opt.eventTimestampColumn));
            String k = entityKey(entity, joinKeys);
            List<Map<String, Object>> bucket = byKey.getOrDefault(k, List.of());
            stats.featureRowsScanned += bucket.size();

            Map<String, Object> best = null;
            long bestTs = Long.MIN_VALUE;
            for (Map<String, Object> fr : bucket) {
                long fts = FileOfflineStore.toEpochMillis(fr.get(featureTsCol));
                if (fts > eventTs) {
                    stats.futureRowsRejected++;
                    continue; // NO FUTURE LEAKAGE
                }
                if (ttlMs > 0 && (eventTs - fts) > ttlMs) {
                    stats.ttlExpiredRejected++;
                    continue;
                }
                if (fts >= bestTs) {
                    bestTs = fts;
                    best = fr;
                }
            }

            if (best != null) {
                stats.joinsHit++;
                for (String fn : featureNames) {
                    String col = opt.prefixWithViewName ? view.name() + opt.prefixSeparator + fn : fn;
                    row.put(col, best.get(fn));
                }
            } else {
                stats.joinsMiss++;
                if (opt.emitNullsForMissing) {
                    for (String fn : featureNames) {
                        String col = opt.prefixWithViewName ? view.name() + opt.prefixSeparator + fn : fn;
                        row.putIfAbsent(col, null);
                    }
                }
            }
            out.add(Collections.unmodifiableMap(row));
        }

        stats.elapsedNanos = System.nanoTime() - t0;
        return new Result(out, stats, outputColumns);
    }

    /**
     * Multi-view PIT join: sequentially left-join each view onto the entity rows.
     */
    public static Result joinMany(List<Map<String, Object>> entityRows,
                                  Map<String, List<Map<String, Object>>> featuresByViewName,
                                  List<FeatureView> views,
                                  Options options) {
        Objects.requireNonNull(entityRows, "entityRows");
        Objects.requireNonNull(featuresByViewName, "featuresByViewName");
        Objects.requireNonNull(views, "views");
        Options opt = options != null ? options : new Options();

        List<Map<String, Object>> current = entityRows;
        JoinStats total = new JoinStats();
        total.entityRows = entityRows.size();
        List<String> cols = new ArrayList<>();
        if (!entityRows.isEmpty()) cols.addAll(entityRows.get(0).keySet());

        long t0 = System.nanoTime();
        for (FeatureView view : views) {
            List<Map<String, Object>> frows = featuresByViewName.getOrDefault(view.name(), List.of());
            Result r = joinOne(current, frows, view, opt);
            current = mutableCopy(r.rows);
            total.featureRowsScanned += r.stats.featureRowsScanned;
            total.joinsHit += r.stats.joinsHit;
            total.joinsMiss += r.stats.joinsMiss;
            total.futureRowsRejected += r.stats.futureRowsRejected;
            total.ttlExpiredRejected += r.stats.ttlExpiredRejected;
            for (String c : r.outputColumns) {
                if (!cols.contains(c)) cols.add(c);
            }
        }
        total.elapsedNanos = System.nanoTime() - t0;
        List<Map<String, Object>> frozen = new ArrayList<>(current.size());
        for (Map<String, Object> r : current) {
            frozen.add(Collections.unmodifiableMap(new LinkedHashMap<>(r)));
        }
        return new Result(frozen, total, cols);
    }

    /**
     * Correctness probe: count how many joined feature timestamps are strictly greater
     * than the entity event timestamp (must be 0).
     */
    public static long countFutureLeaks(List<Map<String, Object>> joinedRows,
                                        List<Map<String, Object>> featureRows,
                                        FeatureView view,
                                        Options options) {
        // Re-validate by checking that for every entity, selected feature values
        // could only come from rows with fts <= event_ts. We verify by ensuring
        // no output feature equals a value that ONLY appears in future rows for that key.
        // Simpler direct check used in benchmarks: re-run joinOne and assert
        // stats.futureRowsRejected accounting; this helper checks raw feature rows vs entity.
        Options opt = options != null ? options : new Options();
        String featureTsCol = view.source() != null ? view.source().timestampColumn() : DEFAULT_EVENT_TS;
        List<String> joinKeys = view.joinKeys().isEmpty() ? view.entityNames() : view.joinKeys();
        long leaks = 0;
        for (Map<String, Object> entity : joinedRows) {
            long eventTs = FileOfflineStore.toEpochMillis(entity.get(opt.eventTimestampColumn));
            String k = entityKey(entity, joinKeys);
            for (Map<String, Object> fr : featureRows) {
                if (!entityKey(fr, joinKeys).equals(k)) continue;
                long fts = FileOfflineStore.toEpochMillis(fr.get(featureTsCol));
                if (fts <= eventTs) continue;
                // If joined row carries values matching this future-only row for all features,
                // count as potential leak signal (benchmark uses join stats primarily).
                boolean allMatch = !view.featureNames().isEmpty();
                for (String fn : view.featureNames()) {
                    String col = opt.prefixWithViewName ? view.name() + opt.prefixSeparator + fn : fn;
                    Object jv = entity.get(col);
                    Object fv = fr.get(fn);
                    if (!Objects.equals(jv, fv)) {
                        allMatch = false;
                        break;
                    }
                }
                // Only count if the same values could NOT come from any non-future row
                if (allMatch) {
                    boolean explainedByPast = false;
                    for (Map<String, Object> past : featureRows) {
                        if (!entityKey(past, joinKeys).equals(k)) continue;
                        long pts = FileOfflineStore.toEpochMillis(past.get(featureTsCol));
                        if (pts > eventTs) continue;
                        boolean same = true;
                        for (String fn : view.featureNames()) {
                            if (!Objects.equals(past.get(fn), fr.get(fn))) {
                                same = false;
                                break;
                            }
                        }
                        if (same) {
                            explainedByPast = true;
                            break;
                        }
                    }
                    if (!explainedByPast) leaks++;
                }
            }
        }
        return leaks;
    }

    public static String entityKey(Map<String, Object> row, List<String> joinKeys) {
        if (joinKeys == null || joinKeys.isEmpty()) return "";
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < joinKeys.size(); i++) {
            if (i > 0) sb.append('|');
            Object v = row.get(joinKeys.get(i));
            sb.append(v == null ? "" : String.valueOf(v));
        }
        return sb.toString();
    }

    private static List<Map<String, Object>> mutableCopy(List<Map<String, Object>> rows) {
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> r : rows) {
            out.add(new LinkedHashMap<>(r));
        }
        return out;
    }

    /** Build a wide column name for a view field. */
    public static String columnName(FeatureView view, Field field, Options options) {
        Options opt = options != null ? options : new Options();
        if (opt.prefixWithViewName) {
            return view.name() + opt.prefixSeparator + field.name();
        }
        return field.name();
    }
}
