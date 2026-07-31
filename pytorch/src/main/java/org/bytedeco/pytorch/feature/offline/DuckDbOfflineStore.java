/*
 * DuckDB-backed offline store — prefers SQL range scans / ASOF when JDBC present.
 * Falls back to in-memory FileOfflineStore semantics if DuckDB driver is unavailable.
 *
 * DuckDB ASOF JOIN docs: https://duckdb.org/docs/sql/query_syntax/from#asof-joins
 */
package org.bytedeco.pytorch.feature.offline;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.duckdb.DuckDB;
import org.bytedeco.pytorch.feature.core.FeatureView;

import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Offline store that keeps a FileOfflineStore mirror and optionally syncs tables into DuckDB.
 */
public final class DuckDbOfflineStore implements OfflineStore {

    private final FileOfflineStore mirror;
    private final boolean duckAvailable;
    private final ConcurrentHashMap<String, Boolean> registered = new ConcurrentHashMap<>();
    private DuckDB db; // nullable

    public DuckDbOfflineStore() {
        this(null);
    }

    public DuckDbOfflineStore(Path fileRoot) {
        this.mirror = fileRoot != null ? new FileOfflineStore(fileRoot) : FileOfflineStore.inMemory();
        boolean ok = false;
        DuckDB local = null;
        try {
            local = DuckDB.inMemory();
            ok = true;
        } catch (Throwable t) {
            ok = false;
            local = null;
        }
        this.duckAvailable = ok;
        this.db = local;
    }

    public boolean duckAvailable() {
        return duckAvailable;
    }

    public FileOfflineStore mirror() {
        return mirror;
    }

    private static String key(String project, String viewName) {
        return (project == null || project.isEmpty() ? "default" : project) + "/" + viewName;
    }

    private static String tableName(String project, String viewName) {
        String p = (project == null || project.isEmpty() ? "default" : project)
                .replaceAll("[^A-Za-z0-9_]", "_");
        String v = viewName.replaceAll("[^A-Za-z0-9_]", "_");
        return "fv_" + p + "_" + v;
    }

    @Override
    public void put(String project, String viewName, List<Map<String, Object>> rows) {
        mirror.put(project, viewName, rows);
        syncToDuck(project, viewName);
    }

    @Override
    public void replace(String project, String viewName, List<Map<String, Object>> rows) {
        mirror.replace(project, viewName, rows);
        registered.remove(key(project, viewName));
        syncToDuck(project, viewName);
    }

    private void syncToDuck(String project, String viewName) {
        if (!duckAvailable || db == null) return;
        try {
            List<Map<String, Object>> all = mirror.readAll(project, viewName);
            DataFrame df = rowsToDataFrame(all);
            String tname = tableName(project, viewName);
            db.register(tname, df);
            registered.put(key(project, viewName), Boolean.TRUE);
        } catch (Throwable t) {
            // keep mirror as source of truth
        }
    }

    static DataFrame rowsToDataFrame(List<Map<String, Object>> rows) {
        DataFrame df = DataFrame.create();
        if (rows == null || rows.isEmpty()) return df;
        // union of keys preserving first-row order then extras
        List<String> cols = new ArrayList<>(rows.get(0).keySet());
        for (Map<String, Object> r : rows) {
            for (String k : r.keySet()) {
                if (!cols.contains(k)) cols.add(k);
            }
        }
        for (String c : cols) {
            df.addColumn(c, inferDtype(rows, c));
        }
        for (Map<String, Object> r : rows) {
            int idx = df.addRow();
            for (String c : cols) {
                df.set(idx, c, r.get(c));
            }
        }
        return df;
    }

    private static org.bytedeco.pytorch.dataframe.Column.DType inferDtype(List<Map<String, Object>> rows, String col) {
        for (Map<String, Object> r : rows) {
            Object v = r.get(col);
            if (v == null) continue;
            if (v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte) {
                return org.bytedeco.pytorch.dataframe.Column.DType.INT64;
            }
            if (v instanceof Double || v instanceof Float) {
                return org.bytedeco.pytorch.dataframe.Column.DType.FLOAT64;
            }
            if (v instanceof Boolean) {
                return org.bytedeco.pytorch.dataframe.Column.DType.BOOLEAN;
            }
            if (v instanceof float[] || v instanceof double[]) {
                return org.bytedeco.pytorch.dataframe.Column.DType.VECTOR;
            }
            if (v instanceof long[] || v instanceof int[] || v instanceof List) {
                return org.bytedeco.pytorch.dataframe.Column.DType.LIST;
            }
            return org.bytedeco.pytorch.dataframe.Column.DType.STRING;
        }
        return org.bytedeco.pytorch.dataframe.Column.DType.STRING;
    }

    @Override
    public List<Map<String, Object>> readAll(String project, String viewName) {
        return mirror.readAll(project, viewName);
    }

    @Override
    public List<Map<String, Object>> readRange(String project, String viewName,
                                               Instant start, Instant end,
                                               String timestampColumn) {
        // Prefer mirror filter for correctness parity; DuckDB path available for SQL users via export
        return mirror.readRange(project, viewName, start, end, timestampColumn);
    }

    /**
     * Attempt DuckDB SQL ASOF-style retrieval helper.
     * Returns empty if DuckDB unavailable — callers should use {@link PointInTimeJoin}.
     */
    public List<Map<String, Object>> trySqlAsOf(FeatureView view,
                                                List<Map<String, Object>> entityRows,
                                                String eventTsCol) {
        if (!duckAvailable || db == null || entityRows == null || entityRows.isEmpty()) {
            return List.of();
        }
        try {
            syncToDuck(view.project(), view.name());
            String tFeature = tableName(view.project(), view.name());
            DataFrame entityDf = rowsToDataFrame(entityRows);
            db.register("entity_df", entityDf);
            String joinKey = view.joinKeys().isEmpty() ? view.entityNames().get(0) : view.joinKeys().get(0);
            String fts = view.source() != null ? view.source().timestampColumn() : "event_timestamp";
            // ASOF JOIN: entity asof feature on key and timestamp
            String sql = "SELECT e.*, f.* EXCLUDE (" + joinKey + ", " + fts + ") "
                    + "FROM entity_df e ASOF LEFT JOIN " + tFeature + " f "
                    + "ON e." + joinKey + " = f." + joinKey + " AND e." + eventTsCol + " >= f." + fts;
            DataFrame out = db.query(sql);
            return dataFrameToRows(out);
        } catch (Throwable t) {
            return List.of();
        }
    }

    static List<Map<String, Object>> dataFrameToRows(DataFrame df) {
        List<Map<String, Object>> rows = new ArrayList<>();
        if (df == null) return rows;
        List<String> names = df.getColumnNames();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (String name : names) {
                row.put(name, df.get(i, name));
            }
            rows.add(row);
        }
        return rows;
    }

    @Override
    public Optional<Long> latestTimestamp(String project, String viewName, String timestampColumn) {
        return mirror.latestTimestamp(project, viewName, timestampColumn);
    }

    @Override
    public long rowCount(String project, String viewName) {
        return mirror.rowCount(project, viewName);
    }

    @Override
    public void close() {
        mirror.close();
        if (db != null) {
            try {
                db.close();
            } catch (Exception ignored) {
            }
            db = null;
        }
    }
}
