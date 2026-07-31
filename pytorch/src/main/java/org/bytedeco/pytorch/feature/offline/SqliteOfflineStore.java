/*
 * SQLite offline store — durable row tables for historical features / PIT source.
 * Complements DuckDB (OLAP) for lighter single-node offline when parquet not needed.
 */
package org.bytedeco.pytorch.feature.offline;

import org.bytedeco.pytorch.feature.online.SqliteOnlineStore;
import org.bytedeco.pytorch.feature.store.FeatureValueCodec;
import org.bytedeco.pytorch.utils.sqlite.SQLite;
import org.bytedeco.pytorch.utils.sqlite.SQLiteConfig;

import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** SQLite-backed {@link OfflineStore}. */
public final class SqliteOfflineStore implements OfflineStore {

    public static final String TABLE = "offline_features";

    private final SQLite db;
    private final boolean ownsDb;
    /** Optional memory mirror for fast PIT within process after load. */
    private final FileOfflineStore mirror = FileOfflineStore.inMemory();

    public SqliteOfflineStore(SQLite db, boolean ownsDb) {
        this.db = Objects.requireNonNull(db, "db");
        this.ownsDb = ownsDb;
        try {
            ensureSchema();
            loadAllIntoMirror();
        } catch (SQLException e) {
            throw new IllegalStateException("SqliteOfflineStore init failed", e);
        }
    }

    public static SqliteOfflineStore open(Path dbFile) {
        try {
            SQLite db = SQLite.open(dbFile, SQLiteConfig.onlineFeatureCache());
            return new SqliteOfflineStore(db, true);
        } catch (Exception e) {
            throw new IllegalStateException("cannot open SqliteOfflineStore at " + dbFile, e);
        }
    }

    public static SqliteOfflineStore inMemory() {
        try {
            return new SqliteOfflineStore(SQLite.inMemory(SQLiteConfig.onlineFeatureCache()), true);
        } catch (SQLException e) {
            throw new IllegalStateException("cannot open in-memory SqliteOfflineStore", e);
        }
    }

    public SQLite db() {
        return db;
    }

    public FileOfflineStore mirror() {
        return mirror;
    }

    private void ensureSchema() throws SQLException {
        db.execute("CREATE TABLE IF NOT EXISTS " + TABLE + " ("
                + " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                + " project TEXT NOT NULL,"
                + " view_name TEXT NOT NULL,"
                + " event_ts INTEGER NOT NULL,"
                + " row_json TEXT NOT NULL"
                + ")");
        db.execute("CREATE INDEX IF NOT EXISTS idx_offline_view_ts "
                + "ON " + TABLE + " (project, view_name, event_ts)");
    }

    private static String proj(String project) {
        return project == null || project.isEmpty() ? "default" : project;
    }

    private void loadAllIntoMirror() throws SQLException {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT project, view_name, row_json FROM " + TABLE);
             ResultSet rs = ps.executeQuery()) {
            Map<String, List<Map<String, Object>>> grouped = new LinkedHashMap<>();
            while (rs.next()) {
                String p = rs.getString(1);
                String v = rs.getString(2);
                Map<String, Object> row = decodeRow(rs.getString(3));
                grouped.computeIfAbsent(p + "\0" + v, x -> new ArrayList<>()).add(row);
            }
            for (Map.Entry<String, List<Map<String, Object>>> e : grouped.entrySet()) {
                String[] parts = e.getKey().split("\0", 2);
                mirror.replace(parts[0], parts[1], e.getValue());
            }
        }
    }

    @Override
    public void put(String project, String viewName, List<Map<String, Object>> rows) {
        if (rows == null || rows.isEmpty()) return;
        mirror.put(project, viewName, rows);
        try {
            db.execute("BEGIN");
            try (PreparedStatement ps = db.connection().prepareStatement(
                    "INSERT INTO " + TABLE + " (project, view_name, event_ts, row_json) VALUES (?,?,?,?)")) {
                for (Map<String, Object> row : rows) {
                    long ts = FileOfflineStore.toEpochMillis(row.get("event_timestamp"));
                    if (ts == 0L) {
                        // try common aliases
                        ts = FileOfflineStore.toEpochMillis(row.get("event_ts"));
                    }
                    ps.setString(1, proj(project));
                    ps.setString(2, viewName);
                    ps.setLong(3, ts);
                    ps.setString(4, encodeRow(row));
                    ps.addBatch();
                }
                ps.executeBatch();
            }
            db.execute("COMMIT");
        } catch (SQLException e) {
            try { db.execute("ROLLBACK"); } catch (SQLException ignored) {}
            throw new IllegalStateException("SqliteOfflineStore.put failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void replace(String project, String viewName, List<Map<String, Object>> rows) {
        mirror.replace(project, viewName, rows);
        try {
            db.execute("BEGIN");
            try (PreparedStatement del = db.connection().prepareStatement(
                    "DELETE FROM " + TABLE + " WHERE project=? AND view_name=?")) {
                del.setString(1, proj(project));
                del.setString(2, viewName);
                del.executeUpdate();
            }
            if (rows != null && !rows.isEmpty()) {
                try (PreparedStatement ps = db.connection().prepareStatement(
                        "INSERT INTO " + TABLE + " (project, view_name, event_ts, row_json) VALUES (?,?,?,?)")) {
                    for (Map<String, Object> row : rows) {
                        long ts = FileOfflineStore.toEpochMillis(row.get("event_timestamp"));
                        ps.setString(1, proj(project));
                        ps.setString(2, viewName);
                        ps.setLong(3, ts);
                        ps.setString(4, encodeRow(row));
                        ps.addBatch();
                    }
                    ps.executeBatch();
                }
            }
            db.execute("COMMIT");
        } catch (SQLException e) {
            try { db.execute("ROLLBACK"); } catch (SQLException ignored) {}
            throw new IllegalStateException("SqliteOfflineStore.replace failed: " + e.getMessage(), e);
        }
    }

    @Override
    public List<Map<String, Object>> readAll(String project, String viewName) {
        return mirror.readAll(project, viewName);
    }

    @Override
    public List<Map<String, Object>> readRange(String project, String viewName,
                                               Instant start, Instant end,
                                               String timestampColumn) {
        // Prefer SQL range for correctness with large durable sets, then fall back mirror
        try {
            long s = start != null ? start.toEpochMilli() : Long.MIN_VALUE;
            long e = end != null ? end.toEpochMilli() : Long.MAX_VALUE;
            List<Map<String, Object>> out = new ArrayList<>();
            try (PreparedStatement ps = db.connection().prepareStatement(
                    "SELECT row_json FROM " + TABLE
                            + " WHERE project=? AND view_name=? AND event_ts>=? AND event_ts<=?"
                            + " ORDER BY event_ts")) {
                ps.setString(1, proj(project));
                ps.setString(2, viewName);
                ps.setLong(3, s);
                ps.setLong(4, e);
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        out.add(decodeRow(rs.getString(1)));
                    }
                }
            }
            return out;
        } catch (SQLException ex) {
            return mirror.readRange(project, viewName, start, end, timestampColumn);
        }
    }

    @Override
    public Optional<Long> latestTimestamp(String project, String viewName, String timestampColumn) {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT MAX(event_ts) FROM " + TABLE + " WHERE project=? AND view_name=?")) {
            ps.setString(1, proj(project));
            ps.setString(2, viewName);
            try (ResultSet rs = ps.executeQuery()) {
                if (rs.next()) {
                    long v = rs.getLong(1);
                    if (!rs.wasNull()) return Optional.of(v);
                }
            }
        } catch (SQLException ignored) {
        }
        return mirror.latestTimestamp(project, viewName, timestampColumn);
    }

    @Override
    public long rowCount(String project, String viewName) {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT COUNT(*) FROM " + TABLE + " WHERE project=? AND view_name=?")) {
            ps.setString(1, proj(project));
            ps.setString(2, viewName);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getLong(1) : 0L;
            }
        } catch (SQLException e) {
            return mirror.rowCount(project, viewName);
        }
    }

    @Override
    public void close() {
        mirror.close();
        if (ownsDb) {
            try {
                db.close();
            } catch (Exception ignored) {
            }
        }
    }

    static String encodeRow(Map<String, Object> row) {
        Map<String, String> enc = FeatureValueCodec.encodeMap(row);
        return SqliteOnlineStore.mapToJsonObject(enc);
    }

    static Map<String, Object> decodeRow(String json) {
        Map<String, String> enc = SqliteOnlineStore.jsonObjectToMap(json);
        return FeatureValueCodec.decodeMap(enc);
    }
}
