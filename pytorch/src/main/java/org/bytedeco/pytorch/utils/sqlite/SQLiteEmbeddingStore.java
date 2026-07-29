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
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;

/**
 * Local embedding / vector KV store on SQLite for on-device and process-local
 * multimodal + recsys retrieval sidecars.
 *
 * <p>Honest scope (what big-tech public systems actually do at this layer):
 * <ul>
 *   <li><b>Apple</b> on-device embedding tables + brute-force / small-index top-K</li>
 *   <li><b>Meta / Google</b> process-local ID→embedding maps feeding larger ANN
 *       (Faiss/ScaNN/Lance) — SQLite holds the authoritative float payloads</li>
 *   <li><b>ByteDance / Tencent</b> item tower embedding mirrors for edge rankers</li>
 * </ul>
 *
 * <p>This is <b>not</b> a replacement for Faiss/Lance at million-scale ANN.
 * It provides:
 * <ul>
 *   <li>ID → float[] get/put with WAL durability</li>
 *   <li>Batch get for two-tower / multimodal scoring</li>
 *   <li>Brute-force cosine / L2 top-K over a namespace (tens–hundreds of thousands
 *       rows is practical; beyond that export to Lance/Faiss)</li>
 *   <li>Namespace isolation (user_tower / item_tower / clip_image / clap_audio …)</li>
 * </ul>
 *
 * <pre>{@code
 * try (SQLiteEmbeddingStore store = SQLiteEmbeddingStore.open(Path.of("emb.db"), 128)) {
 *     store.put("item_tower", 1001L, emb);
 *     float[] v = store.get("item_tower", 1001L);
 *     List<Hit> hits = store.topKCosine("item_tower", query, 10);
 * }
 * }</pre>
 */
public final class SQLiteEmbeddingStore implements AutoCloseable {

    public static final String TABLE = "embeddings";

    private final SQLite db;
    private final boolean ownsDb;
    private final int defaultDim;

    public SQLiteEmbeddingStore(SQLite db, int defaultDim) {
        this(db, defaultDim, false);
    }

    public SQLiteEmbeddingStore(SQLite db, int defaultDim, boolean ownsDb) {
        this.db = Objects.requireNonNull(db, "db");
        this.defaultDim = Math.max(1, defaultDim);
        this.ownsDb = ownsDb;
    }

    public static SQLiteEmbeddingStore open(Path dbFile, int dim) throws Exception {
        SQLite db = SQLite.open(dbFile, SQLiteConfig.embeddingStore());
        SQLiteEmbeddingStore s = new SQLiteEmbeddingStore(db, dim, true);
        s.ensureSchema();
        return s;
    }

    public static SQLiteEmbeddingStore inMemory(int dim) throws SQLException {
        SQLite db = SQLite.inMemory(SQLiteConfig.embeddingStore());
        SQLiteEmbeddingStore s = new SQLiteEmbeddingStore(db, dim, true);
        s.ensureSchema();
        return s;
    }

    public SQLite db() {
        return db;
    }

    public int defaultDim() {
        return defaultDim;
    }

    public void ensureSchema() throws SQLException {
        db.execute("CREATE TABLE IF NOT EXISTS " + TABLE + " ("
                + " ns         TEXT NOT NULL,"
                + " id         INTEGER NOT NULL,"
                + " dim        INTEGER NOT NULL,"
                + " vector     BLOB NOT NULL,"
                + " norm       REAL,"
                + " meta_json  TEXT,"
                + " updated_at INTEGER NOT NULL,"
                + " PRIMARY KEY (ns, id)"
                + ")");
        db.execute("CREATE INDEX IF NOT EXISTS idx_emb_ns ON " + TABLE + " (ns)");
    }

    // ---- put / get ---------------------------------------------------------

    public void put(String ns, long id, float[] vector) throws SQLException {
        put(ns, id, vector, null);
    }

    public void put(String ns, long id, float[] vector, String metaJson) throws SQLException {
        Objects.requireNonNull(ns, "ns");
        Objects.requireNonNull(vector, "vector");
        if (vector.length == 0) throw new IllegalArgumentException("empty vector");
        float norm = l2Norm(vector);
        db.executeUpdate(
                "INSERT INTO " + TABLE
                        + " (ns, id, dim, vector, norm, meta_json, updated_at) "
                        + "VALUES (?, ?, ?, ?, ?, ?, ?) "
                        + "ON CONFLICT (ns, id) DO UPDATE SET "
                        + " dim=excluded.dim, vector=excluded.vector, norm=excluded.norm, "
                        + " meta_json=excluded.meta_json, updated_at=excluded.updated_at",
                ns, id, vector.length, SQLite.floatsToBlob(vector), (double) norm,
                metaJson, System.currentTimeMillis());
    }

    public void putBatch(String ns, Map<Long, float[]> vectors) throws SQLException {
        Objects.requireNonNull(ns, "ns");
        if (vectors == null || vectors.isEmpty()) return;
        db.begin();
        try {
            for (Map.Entry<Long, float[]> e : vectors.entrySet()) {
                put(ns, e.getKey(), e.getValue());
            }
            db.commit();
        } catch (SQLException ex) {
            db.rollback();
            throw ex;
        }
    }

    public float[] get(String ns, long id) throws SQLException {
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT vector FROM " + TABLE + " WHERE ns=? AND id=?")) {
            ps.setString(1, ns);
            ps.setLong(2, id);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) return null;
                return SQLite.blobToFloats(rs.getBytes(1));
            }
        }
    }

    public Map<Long, float[]> getBatch(String ns, List<Long> ids) throws SQLException {
        Map<Long, float[]> out = new LinkedHashMap<>();
        if (ids == null || ids.isEmpty()) return out;
        final int chunk = 500;
        for (int i = 0; i < ids.size(); i += chunk) {
            List<Long> part = ids.subList(i, Math.min(ids.size(), i + chunk));
            StringBuilder ph = new StringBuilder();
            for (int j = 0; j < part.size(); j++) {
                if (j > 0) ph.append(',');
                ph.append('?');
            }
            String sql = "SELECT id, vector FROM " + TABLE + " WHERE ns=? AND id IN (" + ph + ")";
            try (PreparedStatement ps = db.connection().prepareStatement(sql)) {
                ps.setString(1, ns);
                for (int j = 0; j < part.size(); j++) ps.setLong(j + 2, part.get(j));
                try (ResultSet rs = ps.executeQuery()) {
                    while (rs.next()) {
                        out.put(rs.getLong(1), SQLite.blobToFloats(rs.getBytes(2)));
                    }
                }
            }
        }
        return out;
    }

    public int delete(String ns, long id) throws SQLException {
        return db.executeUpdate("DELETE FROM " + TABLE + " WHERE ns=? AND id=?", ns, id);
    }

    public int deleteNamespace(String ns) throws SQLException {
        return db.executeUpdate("DELETE FROM " + TABLE + " WHERE ns=?", ns);
    }

    public long count(String ns) throws Exception {
        DataFrame df = ns == null
                ? db.query("SELECT count(*) AS c FROM " + TABLE)
                : db.query("SELECT count(*) AS c FROM " + TABLE + " WHERE ns=?", ns);
        if (df.rowCount() == 0) return 0L;
        Object v = df.get(0, "c");
        return v instanceof Number ? ((Number) v).longValue() : 0L;
    }

    public List<String> namespaces() throws Exception {
        DataFrame df = db.query("SELECT DISTINCT ns FROM " + TABLE + " ORDER BY ns");
        List<String> out = new ArrayList<>();
        for (int i = 0; i < df.rowCount(); i++) {
            Object v = df.get(i, "ns");
            if (v != null) out.add(v.toString());
        }
        return out;
    }

    // ---- brute-force top-K -------------------------------------------------

    public List<Hit> topKCosine(String ns, float[] query, int k) throws SQLException {
        return topK(ns, query, k, Metric.COSINE);
    }

    public List<Hit> topKL2(String ns, float[] query, int k) throws SQLException {
        return topK(ns, query, k, Metric.L2);
    }

    public List<Hit> topKDot(String ns, float[] query, int k) throws SQLException {
        return topK(ns, query, k, Metric.DOT);
    }

    public List<Hit> topK(String ns, float[] query, int k, Metric metric) throws SQLException {
        Objects.requireNonNull(ns, "ns");
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(metric, "metric");
        int topK = Math.max(1, k);
        float qNorm = l2Norm(query);

        // min-heap for cosine/dot (higher better); max-heap for L2 (lower better)
        Comparator<Hit> cmp = metric == Metric.L2
                ? Comparator.comparingDouble((Hit h) -> h.score).reversed()
                : Comparator.comparingDouble((Hit h) -> h.score);
        PriorityQueue<Hit> heap = new PriorityQueue<>(topK + 1, cmp);

        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT id, vector, norm FROM " + TABLE + " WHERE ns=?")) {
            ps.setString(1, ns);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    long id = rs.getLong(1);
                    float[] v = SQLite.blobToFloats(rs.getBytes(2));
                    if (v == null || v.length != query.length) continue;
                    double storedNorm = rs.getDouble(3);
                    if (rs.wasNull() || storedNorm <= 0) storedNorm = l2Norm(v);
                    double score;
                    switch (metric) {
                        case DOT:
                            score = dot(query, v);
                            break;
                        case L2:
                            score = l2Distance(query, v);
                            break;
                        case COSINE:
                        default:
                            double denom = (double) qNorm * storedNorm;
                            score = denom == 0.0 ? 0.0 : dot(query, v) / denom;
                            break;
                    }
                    Hit hit = new Hit(id, score);
                    if (heap.size() < topK) {
                        heap.offer(hit);
                    } else {
                        Hit worst = heap.peek();
                        boolean better = metric == Metric.L2
                                ? hit.score < worst.score
                                : hit.score > worst.score;
                        if (better) {
                            heap.poll();
                            heap.offer(hit);
                        }
                    }
                }
            }
        }
        List<Hit> out = new ArrayList<>(heap);
        if (metric == Metric.L2) {
            out.sort(Comparator.comparingDouble((Hit h) -> h.score));
        } else {
            out.sort(Comparator.comparingDouble((Hit h) -> h.score).reversed());
        }
        return out;
    }

    /**
     * Export a namespace to DataFrame ({@code id}, {@code embedding} float[]).
     * Useful before building a Lance/Faiss index.
     */
    public DataFrame toDataFrame(String ns) throws Exception {
        Objects.requireNonNull(ns, "ns");
        DataFrame df = DataFrame.create();
        df.addColumn("id", org.bytedeco.pytorch.dataframe.Column.DType.INT64);
        df.addColumn("embedding", org.bytedeco.pytorch.dataframe.Column.DType.VECTOR);
        try (PreparedStatement ps = db.connection().prepareStatement(
                "SELECT id, vector FROM " + TABLE + " WHERE ns=? ORDER BY id")) {
            ps.setString(1, ns);
            try (ResultSet rs = ps.executeQuery()) {
                while (rs.next()) {
                    int r = df.addEmptyRow();
                    df.set(r, "id", rs.getLong(1));
                    df.set(r, "embedding", SQLite.blobToFloats(rs.getBytes(2)));
                }
            }
        }
        return df;
    }

    /** Bulk load from DataFrame with {@code id} + {@code embedding}|{@code vector}. */
    public long loadFromDataFrame(String ns, DataFrame df) throws Exception {
        Objects.requireNonNull(ns, "ns");
        Objects.requireNonNull(df, "df");
        if (!df.hasColumn("id")) throw new IllegalArgumentException("id column required");
        String vcol = df.hasColumn("embedding") ? "embedding"
                : df.hasColumn("vector") ? "vector" : null;
        if (vcol == null) throw new IllegalArgumentException("embedding or vector column required");
        long n = 0;
        db.begin();
        try {
            for (int r = 0; r < df.rowCount(); r++) {
                Object idObj = df.get(r, "id");
                Object vecObj = df.get(r, vcol);
                if (idObj == null || vecObj == null) continue;
                float[] v;
                if (vecObj instanceof float[]) v = (float[]) vecObj;
                else if (vecObj instanceof byte[]) v = SQLite.blobToFloats((byte[]) vecObj);
                else continue;
                put(ns, ((Number) idObj).longValue(), v);
                n++;
            }
            db.commit();
        } catch (Exception e) {
            db.rollback();
            throw e;
        }
        return n;
    }

    public void checkpoint() throws SQLException {
        db.walCheckpoint("TRUNCATE");
        db.optimize();
    }

    @Override
    public void close() {
        if (ownsDb) db.close();
    }

    // ---- types / math ------------------------------------------------------

    public enum Metric {
        COSINE, L2, DOT
    }

    public static final class Hit {
        public final long id;
        public final double score;

        public Hit(long id, double score) {
            this.id = id;
            this.score = score;
        }

        @Override
        public String toString() {
            return "Hit{id=" + id + ", score=" + score + "}";
        }
    }

    static float l2Norm(float[] v) {
        double s = 0;
        for (float x : v) s += (double) x * x;
        return (float) Math.sqrt(s);
    }

    static double dot(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        double s = 0;
        for (int i = 0; i < n; i++) s += (double) a[i] * b[i];
        return s;
    }

    static double l2Distance(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        double s = 0;
        for (int i = 0; i < n; i++) {
            double d = (double) a[i] - b[i];
            s += d * d;
        }
        return Math.sqrt(s);
    }
}
