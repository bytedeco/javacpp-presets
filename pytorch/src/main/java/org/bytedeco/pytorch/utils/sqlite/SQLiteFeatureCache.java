/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet, mullerhai
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.bytedeco.pytorch.utils.sqlite;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.nio.file.Path;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Process-local / edge <b>online feature cache</b> on SQLite (WAL).
 *
 * <p>Complements DuckDB offline feature store:
 * <ul>
 *   <li>DuckDB materializes batch features → Parquet / snapshot</li>
 *   <li>This cache loads entity-keyed rows for low-latency point lookups
 *       during ranking / retrieval (Meta/ByteDance nearline mirrors,
 *       Apple on-device personalization stores, Tencent edge rankers)</li>
 * </ul>
 *
 * <p>Storage model (honest, industry-standard):
 * <ul>
 *   <li>{@code entity_features(entity_type, entity_id, version, dense BLOB,
 *       sparse_json TEXT, emb BLOB, updated_at)} — PRIMARY KEY (entity_type, entity_id)</li>
 *   <li>{@code kv_features(ns, key, value, updated_at)} — generic string KV</li>
 *   <li>Optional TTL via {@code updated_at} filter on read</li>
 * </ul>
 *
 * <pre>{@code
 * try (SQLiteFeatureCache cache = SQLiteFeatureCache.open(Path.of("feat.db"))) {
 *     cache.putUser(42L, new float[]{0.1f, 0.2f}, Map.of("city", "SZ"), emb);
 *     FeatureRow row = cache.getUser(42L);
 *     Map<Long, FeatureRow> batch = cache.getUsers(List.of(1L, 2L, 42L));
 * }
 * }</pre>
 */
public final class SQLiteFeatureCache implements AutoCloseable {

    public static final String ENTITY_TABLE = "entity_features";
    public static final String KV_TABLE = "kv_features";
    public static final String TYPE_USER = "user";
    public static final String TYPE_ITEM = "item";
    public static final String TYPE_QUERY = "query";
    public static final String TYPE_AUTHOR = "author";

    private final SQLite db;
    private final boolean ownsDb;

    public SQLiteFeatureCache(SQLite db) {
        this(db, false);
    }

    public SQLiteFeatureCache(SQLite db, boolean ownsDb) {
        this.db = Objects.requireNonNull(db, "db");
        this.ownsDb = ownsDb;
    }

    public static SQLiteFeatureCache open(Path dbFile) throws Exception {
        SQLite db = SQLite.open(dbFile, SQLiteConfig.onlineFeatureCache());
        SQLiteFeatureCache c = new SQLiteFeatureCache(db, true);
        c.ensureSchema();
        return c;
    }

    public static SQLiteFeatureCache inMemory() throws SQLException {
        SQLite db = SQLite.inMemory(SQLiteConfig.onlineFeatureCache());
        SQLiteFeatureCache c = new SQLiteFeatureCache(db, true);
        c.ensureSchema();
        return c;
    }

    public SQLite db() {
        return db;
    }

    public void ensureSchema() throws SQLException {
        db.execute("CREATE TABLE IF NOT EXISTS " + ENTITY_TABLE + " ("
                + " entity_type TEXT NOT NULL,"
                + " entity_id   INTEGER NOT NULL,"
                + " version     INTEGER NOT NULL DEFAULT 1,"
                + " dense       BLOB,"
                + " sparse_json TEXT,"
                + " emb         BLOB,"
                + " meta_json   TEXT,"
                + " updated_at  INTEGER NOT NULL,"
                + " PRIMARY KEY (entity_type, entity_id)"
                + ")");
        db.execute("CREATE INDEX IF NOT EXISTS idx_entity_updated "
                + "ON " + ENTITY_TABLE + " (updated_at)");
        db.execute("CREATE TABLE IF NOT EXISTS " + KV_TABLE + " ("
                + " ns         TEXT NOT NULL,"
                + " key        TEXT NOT NULL,"
                + " value      TEXT,"
                + " value_blob BLOB,"
                + " updated_at INTEGER NOT NULL,"
                + " PRIMARY KEY (ns, key)"
                + ")");
    }

    // ---- entity put / get --------------------------------------------------

    public void put(String entityType, long entityId, float[] dense,
                    String sparseJson, float[] emb, String metaJson)
            throws SQLException {
        Objects.requireNonNull(entityType, "entityType");
        long now = System.currentTimeMillis();
        db.executeUpdate(
                "INSERT INTO " + ENTITY_TABLE
                        + " (entity_type, entity_id, version, dense, sparse_json, emb, meta_json, updated_at) "
                        + "VALUES (?, ?, 1, ?, ?, ?, ?, ?) "
                        + "ON CONFLICT (entity_type, entity_id) DO UPDATE SET "
                        + " version=entity_features.version+1, "
                        + " dense=excluded.dense, sparse_json=excluded.sparse_json, "
                        + " emb=excluded.emb, meta_json=excluded.meta_json, "
                        + " updated_at=excluded.updated_at",
                entityType, entityId,
                SQLite.floatsToBlob(dense), sparseJson,
                SQLite.floatsToBlob(emb), metaJson, now);
    }

    public void putUser(long userId, float[] dense, Map<String, ?> sparse, float[] emb)
            throws SQLException {
        put(TYPE_USER, userId, dense, toJson(sparse), emb, null);
    }

    public void putItem(long itemId, float[] dense, Map<String, ?> sparse, float[] emb)
            throws SQLException {
        put(TYPE_ITEM, itemId, dense, toJson(sparse), emb, null);
    }

    public FeatureRow get(String entityType, long entityId) throws SQLException {
        return get(entityType, entityId, 0L);
    }

    /**
     * @param maxAgeMs 0 = no TTL; otherwise require {@code updated_at >= now - maxAgeMs}
     */
    public FeatureRow get(String entityType, long entityId, long maxAgeMs) throws SQLException {
        String sql = "SELECT entity_type, entity_id, version, dense, sparse_json, emb, meta_json, updated_at "
                + "FROM " + ENTITY_TABLE + " WHERE entity_type=? AND entity_id=?";
        if (maxAgeMs > 0) {
            sql += " AND updated_at >= ?";
        }
        try (PreparedStatement ps = db.connection().prepareStatement(sql)) {
            ps.setString(1, entityType);
            ps.setLong(2, entityId);
            if (maxAgeMs > 0) {
                ps.setLong(3, System.currentTimeMillis() - maxAgeMs);
            }
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return null;
                return FeatureRow.fromResultSet(rs);
            }
        }
    }

    public FeatureRow getUser(long userId) throws SQLException {
        return get(TYPE_USER, userId);
    }

    public FeatureRow getItem(long itemId) throws SQLException {
        return get(TYPE_ITEM, itemId);
    }

    /** Batched point lookup — single query with {@code IN (...)} (serving hot path). */
    public Map<Long, FeatureRow> getBatch(String entityType, List<Long> ids)
            throws SQLException {
        Map<Long, FeatureRow> out = new LinkedHashMap<>();
        if (ids == null || ids.isEmpty()) return out;
        // chunk to stay under SQLite variable limits
        final int chunk = 500;
        for (int i = 0; i < ids.size(); i += chunk) {
            List<Long> part = ids.subList(i, Math.min(ids.size(), i + chunk));
            StringBuilder ph = new StringBuilder();
            for (int j = 0; j < part.size(); j++) {
                if (j > 0) ph.append(',');
                ph.append('?');
            }
            String sql = "SELECT entity_type, entity_id, version, dense, sparse_json, emb, meta_json, updated_at "
                    + "FROM " + ENTITY_TABLE + " WHERE entity_type=? AND entity_id IN (" + ph + ")";
            try (PreparedStatement ps = db.connection().prepareStatement(sql)) {
                ps.setString(1, entityType);
                for (int j = 0; j < part.size(); j++) {
                    ps.setLong(j + 2, part.get(j));
                }
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        FeatureRow row = FeatureRow.fromResultSet(rs);
                        out.put(row.entityId, row);
                    }
                }
            }
        }
        return out;
    }

    public Map<Long, FeatureRow> getUsers(List<Long> ids) throws SQLException {
        return getBatch(TYPE_USER, ids);
    }

    public Map<Long, FeatureRow> getItems(List<Long> ids) throws SQLException {
        return getBatch(TYPE_ITEM, ids);
    }

    public int delete(String entityType, long entityId) throws SQLException {
        return db.executeUpdate(
                "DELETE FROM " + ENTITY_TABLE + " WHERE entity_type=? AND entity_id=?",
                entityType, entityId);
    }

    public long count(String entityType) throws Exception {
        DataFrame df = entityType == null
                ? db.query("SELECT count(*) AS c FROM " + ENTITY_TABLE)
                : db.query("SELECT count(*) AS c FROM " + ENTITY_TABLE + " WHERE entity_type=?",
                entityType);
        if (df.rowCount() == 0) return 0L;
        Object v = df.get(0, "c");
        return v instanceof Number ? ((Number) v).longValue() : 0L;
    }

    // ---- bulk warm from DataFrame / DuckDB export --------------------------

    /**
     * Warm cache from a DataFrame with columns
     * {@code entity_id} (required), optional {@code dense}/{@code emb} as float[],
     * optional {@code sparse_json}.
     */
    public long warmFromDataFrame(String entityType, DataFrame df) throws Exception {
        Objects.requireNonNull(entityType, "entityType");
        Objects.requireNonNull(df, "df");
        if (!df.hasColumn("entity_id")) {
            throw new IllegalArgumentException("DataFrame must have entity_id column");
        }
        boolean hasDense = df.hasColumn("dense");
        boolean hasEmb = df.hasColumn("emb") || df.hasColumn("embedding");
        String embCol = df.hasColumn("emb") ? "emb" : "embedding";
        boolean hasSparse = df.hasColumn("sparse_json");
        boolean hasMeta = df.hasColumn("meta_json");
        long n = 0;
        db.begin();
        try {
            for (int r = 0; r < df.rowCount(); r++) {
                Object idObj = df.get(r, "entity_id");
                if (idObj == null) continue;
                long id = ((Number) idObj).longValue();
                float[] dense = hasDense ? asFloats(df.get(r, "dense")) : null;
                float[] emb = hasEmb ? asFloats(df.get(r, embCol)) : null;
                String sparse = hasSparse && df.get(r, "sparse_json") != null
                        ? df.get(r, "sparse_json").toString() : null;
                String meta = hasMeta && df.get(r, "meta_json") != null
                        ? df.get(r, "meta_json").toString() : null;
                put(entityType, id, dense, sparse, emb, meta);
                n++;
            }
            db.commit();
        } catch (Exception e) {
            db.rollback();
            throw e;
        }
        return n;
    }

    // ---- generic KV --------------------------------------------------------

    public void kvPut(String ns, String key, String value) throws SQLException {
        db.executeUpdate(
                "INSERT INTO " + KV_TABLE + " (ns, key, value, value_blob, updated_at) "
                        + "VALUES (?, ?, ?, NULL, ?) "
                        + "ON CONFLICT (ns, key) DO UPDATE SET value=excluded.value, "
                        + " value_blob=NULL, updated_at=excluded.updated_at",
                ns, key, value, System.currentTimeMillis());
    }

    public void kvPutBlob(String ns, String key, byte[] value) throws SQLException {
        db.executeUpdate(
                "INSERT INTO " + KV_TABLE + " (ns, key, value, value_blob, updated_at) "
                        + "VALUES (?, ?, NULL, ?, ?) "
                        + "ON CONFLICT (ns, key) DO UPDATE SET value=NULL, "
                        + " value_blob=excluded.value_blob, updated_at=excluded.updated_at",
                ns, key, value, System.currentTimeMillis());
    }

    public String kvGet(String ns, String key) throws SQLException {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT value FROM " + KV_TABLE + " WHERE ns=? AND key=?")) {
            ps.setString(1, ns);
            ps.setString(2, key);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return null;
                return rs.getString(1);
            }
        }
    }

    public byte[] kvGetBlob(String ns, String key) throws SQLException {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT value_blob FROM " + KV_TABLE + " WHERE ns=? AND key=?")) {
            ps.setString(1, ns);
            ps.setString(2, key);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return null;
                return rs.getBytes(1);
            }
        }
    }

    public int purgeOlderThan(long epochMs) throws SQLException {
        int a = db.executeUpdate("DELETE FROM " + ENTITY_TABLE + " WHERE updated_at < ?", epochMs);
        int b = db.executeUpdate("DELETE FROM " + KV_TABLE + " WHERE updated_at < ?", epochMs);
        return a + b;
    }

    public void checkpoint() throws SQLException {
        db.walCheckpoint("TRUNCATE");
        db.optimize();
    }

    @Override
    public void close() {
        if (ownsDb) db.close();
    }

    // ---- row type ----------------------------------------------------------

    public static final class FeatureRow {
        public final String entityType;
        public final long entityId;
        public final int version;
        public final float[] dense;
        public final String sparseJson;
        public final float[] emb;
        public final String metaJson;
        public final long updatedAt;

        public FeatureRow(String entityType, long entityId, int version,
                          float[] dense, String sparseJson, float[] emb,
                          String metaJson, long updatedAt) {
            this.entityType = entityType;
            this.entityId = entityId;
            this.version = version;
            this.dense = dense;
            this.sparseJson = sparseJson;
            this.emb = emb;
            this.metaJson = metaJson;
            this.updatedAt = updatedAt;
        }

        static FeatureRow fromResultSet(ResultSet rs) throws SQLException {
            return new FeatureRow(
                    rs.getString("entity_type"),
                    rs.getLong("entity_id"),
                    rs.getInt("version"),
                    SQLite.blobToFloats(rs.getBytes("dense")),
                    rs.getString("sparse_json"),
                    SQLite.blobToFloats(rs.getBytes("emb")),
                    rs.getString("meta_json"),
                    rs.getLong("updated_at"));
        }
    }

    private static float[] asFloats(Object v) {
        if (v == null) return null;
        if (v instanceof float[]) return (float[]) v;
        if (v instanceof double[]) {
            double[] d = (double[]) v;
            float[] f = new float[d.length];
            for (int i = 0; i < d.length; i++) f[i] = (float) d[i];
            return f;
        }
        if (v instanceof byte[]) return SQLite.blobToFloats((byte[]) v);
        if (v instanceof List) {
            List<?> list = (List<?>) v;
            float[] f = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object x = list.get(i);
                f[i] = x instanceof Number ? ((Number) x).floatValue() : 0f;
            }
            return f;
        }
        return null;
    }

    private static String toJson(Map<String, ?> map) {
        if (map == null || map.isEmpty()) return null;
        // minimal JSON object — avoid pulling extra deps; good enough for sparse side-info
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, ?> e : map.entrySet()) {
            if (!first) sb.append(',');
            first = false;
            sb.append('"').append(escapeJson(e.getKey())).append("\":");
            Object v = e.getValue();
            if (v == null) sb.append("null");
            else if (v instanceof Number || v instanceof Boolean) sb.append(v);
            else sb.append('"').append(escapeJson(v.toString())).append('"');
        }
        sb.append('}');
        return sb.toString();
    }

    private static String escapeJson(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
