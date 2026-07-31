/*
 * SQLite online feature store — process-local / edge WAL cache.
 *
 * Wraps utils.sqlite.SQLite with a generic (project, view, entity_key) schema
 * compatible with OnlineStore SPI. Complements SQLiteFeatureCache (typed
 * user/item rows) with full FeatureView column maps.
 *
 * Industry: Meta/ByteDance nearline feature mirrors, Apple on-device stores,
 * Tencent edge rankers — single-writer multi-reader WAL.
 */
package org.bytedeco.pytorch.feature.online;

import org.bytedeco.pytorch.feature.store.FeatureValueCodec;
import org.bytedeco.pytorch.utils.sqlite.SQLite;
import org.bytedeco.pytorch.utils.sqlite.SQLiteConfig;

import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** SQLite-backed {@link OnlineStore}. */
public final class SqliteOnlineStore implements OnlineStore {

    public static final String TABLE = "online_features";

    private final SQLite db;
    private final boolean ownsDb;
    private final boolean evictExpiredOnRead;

    public SqliteOnlineStore(SQLite db, boolean ownsDb) {
        this.db = Objects.requireNonNull(db, "db");
        this.ownsDb = ownsDb;
        this.evictExpiredOnRead = true;
        try {
            ensureSchema();
        } catch (SQLException e) {
            throw new IllegalStateException("SqliteOnlineStore schema init failed", e);
        }
    }

    public static SqliteOnlineStore open(Path dbFile) {
        try {
            SQLite db = SQLite.open(dbFile, SQLiteConfig.onlineFeatureCache());
            return new SqliteOnlineStore(db, true);
        } catch (Exception e) {
            throw new IllegalStateException("cannot open SqliteOnlineStore at " + dbFile, e);
        }
    }

    public static SqliteOnlineStore inMemory() {
        try {
            SQLite db = SQLite.inMemory(SQLiteConfig.onlineFeatureCache());
            return new SqliteOnlineStore(db, true);
        } catch (SQLException e) {
            throw new IllegalStateException("cannot open in-memory SqliteOnlineStore", e);
        }
    }

    public SQLite db() {
        return db;
    }

    private void ensureSchema() throws SQLException {
        db.execute("CREATE TABLE IF NOT EXISTS " + TABLE + " ("
                + " project TEXT NOT NULL,"
                + " view_name TEXT NOT NULL,"
                + " entity_key TEXT NOT NULL,"
                + " values_json TEXT NOT NULL,"
                + " event_ts INTEGER NOT NULL DEFAULT 0,"
                + " written_at INTEGER NOT NULL,"
                + " ttl_ms INTEGER NOT NULL DEFAULT 0,"
                + " PRIMARY KEY (project, view_name, entity_key)"
                + ")");
        db.execute("CREATE INDEX IF NOT EXISTS idx_online_view "
                + "ON " + TABLE + " (project, view_name)");
    }

    private static String proj(String project) {
        return project == null || project.isEmpty() ? "default" : project;
    }

    @Override
    public void onlineWrite(OnlineWriteBatch batch) {
        if (batch == null || batch.size() == 0) return;
        try {
            db.execute("BEGIN");
            try (PreparedStatement ps = db.connection().prepareStatement(
                    "INSERT INTO " + TABLE
                            + " (project, view_name, entity_key, values_json, event_ts, written_at, ttl_ms)"
                            + " VALUES (?,?,?,?,?,?,?)"
                            + " ON CONFLICT(project, view_name, entity_key) DO UPDATE SET"
                            + " values_json=excluded.values_json,"
                            + " event_ts=excluded.event_ts,"
                            + " written_at=excluded.written_at,"
                            + " ttl_ms=excluded.ttl_ms")) {
                for (OnlineFeatureRow row : batch.rows()) {
                    Map<String, String> enc = FeatureValueCodec.encodeMap(row.values());
                    String json = mapToJsonObject(enc);
                    ps.setString(1, proj(row.project()));
                    ps.setString(2, row.viewName());
                    ps.setString(3, row.entityKey());
                    ps.setString(4, json);
                    ps.setLong(5, row.eventTimestampMs());
                    ps.setLong(6, row.writtenAtMs());
                    ps.setLong(7, row.ttlMs());
                    ps.addBatch();
                }
                ps.executeBatch();
            }
            db.execute("COMMIT");
        } catch (SQLException e) {
            try { db.execute("ROLLBACK"); } catch (SQLException ignored) {}
            throw new IllegalStateException("SqliteOnlineStore.write failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Optional<OnlineFeatureRow> onlineRead(String project, String viewName, String entityKey) {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT values_json, event_ts, written_at, ttl_ms FROM " + TABLE
                        + " WHERE project=? AND view_name=? AND entity_key=?")) {
            ps.setString(1, proj(project));
            ps.setString(2, viewName);
            ps.setString(3, entityKey);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return Optional.empty();
                OnlineFeatureRow row = fromRs(project, viewName, entityKey, rs);
                if (evictExpiredOnRead && row.isExpired(System.currentTimeMillis())) {
                    delete(project, viewName, entityKey);
                    return Optional.empty();
                }
                return Optional.of(row);
            }
        } catch (SQLException e) {
            throw new IllegalStateException("SqliteOnlineStore.read failed: " + e.getMessage(), e);
        }
    }

    @Override
    public Map<String, OnlineFeatureRow> onlineReadBatch(String project, String viewName,
                                                         Collection<String> entityKeys) {
        Map<String, OnlineFeatureRow> out = new LinkedHashMap<>();
        if (entityKeys == null || entityKeys.isEmpty()) return out;
        // Simple loop — SQLite handles this well for moderate fanout; can IN-optimize later
        for (String ek : entityKeys) {
            onlineRead(project, viewName, ek).ifPresent(r -> out.put(ek, r));
        }
        return out;
    }

    @Override
    public long size(String project, String viewName) {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT COUNT(*) FROM " + TABLE + " WHERE project=? AND view_name=?")) {
            ps.setString(1, proj(project));
            ps.setString(2, viewName);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next() ? rs.getLong(1) : 0L;
            }
        } catch (SQLException e) {
            return -1L;
        }
    }

    @Override
    public void delete(String project, String viewName, String entityKey) {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "DELETE FROM " + TABLE + " WHERE project=? AND view_name=? AND entity_key=?")) {
            ps.setString(1, proj(project));
            ps.setString(2, viewName);
            ps.setString(3, entityKey);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("SqliteOnlineStore.delete failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void clearView(String project, String viewName) {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "DELETE FROM " + TABLE + " WHERE project=? AND view_name=?")) {
            ps.setString(1, proj(project));
            ps.setString(2, viewName);
            ps.executeUpdate();
        } catch (SQLException e) {
            throw new IllegalStateException("SqliteOnlineStore.clearView failed: " + e.getMessage(), e);
        }
    }

    @Override
    public void close() {
        if (ownsDb) {
            try {
                db.close();
            } catch (Exception ignored) {
            }
        }
    }

    private static OnlineFeatureRow fromRs(String project, String viewName, String entityKey,
                                           ResultSet rs) throws SQLException {
        Map<String, String> enc = jsonObjectToMap(rs.getString("values_json"));
        Map<String, Object> values = FeatureValueCodec.decodeMap(enc);
        return OnlineFeatureRow.builder(viewName, entityKey)
                .project(proj(project))
                .values(values)
                .eventTimestampMs(rs.getLong("event_ts"))
                .writtenAtMs(rs.getLong("written_at"))
                .ttlMs(rs.getLong("ttl_ms"))
                .build();
    }

    /** Encode map as flat JSON object of string values. */
    public static String mapToJsonObject(Map<String, String> map) {
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, String> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(esc(e.getKey())).append("\":");
            if (e.getValue() == null) sb.append("null");
            else sb.append('"').append(esc(e.getValue())).append('"');
        }
        return sb.append('}').toString();
    }

    public static Map<String, String> jsonObjectToMap(String raw) {
        Map<String, String> out = new LinkedHashMap<>();
        if (raw == null || raw.isBlank()) return out;
        String s = raw.trim();
        if (s.startsWith("{")) s = s.substring(1);
        if (s.endsWith("}")) s = s.substring(0, s.length() - 1);
        boolean inQ = false;
        StringBuilder cur = new StringBuilder();
        java.util.List<String> parts = new java.util.ArrayList<>();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' && (i == 0 || s.charAt(i - 1) != '\\')) inQ = !inQ;
            if (c == ',' && !inQ) {
                parts.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(c);
            }
        }
        if (cur.length() > 0) parts.add(cur.toString());
        for (String part : parts) {
            int colon = -1;
            inQ = false;
            for (int i = 0; i < part.length(); i++) {
                char c = part.charAt(i);
                if (c == '"' && (i == 0 || part.charAt(i - 1) != '\\')) inQ = !inQ;
                if (c == ':' && !inQ) { colon = i; break; }
            }
            if (colon < 0) continue;
            String k = unq(part.substring(0, colon).trim());
            String v = unq(part.substring(colon + 1).trim());
            if ("null".equals(v)) v = null;
            out.put(k, v);
        }
        return out;
    }

    private static String esc(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String unq(String s) {
        s = s.trim();
        if (s.startsWith("\"") && s.endsWith("\"") && s.length() >= 2) {
            s = s.substring(1, s.length() - 1);
        }
        return s.replace("\\\"", "\"").replace("\\\\", "\\");
    }
}
