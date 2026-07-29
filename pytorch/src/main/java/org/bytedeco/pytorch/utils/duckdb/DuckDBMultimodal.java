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
package org.bytedeco.pytorch.utils.duckdb;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.sql.SQLException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * Multimodal media catalog on DuckDB — video / audio / image / text metadata
 * + embedding vectors for retrieval and recsys side-info.
 *
 * <p>Grounded in public patterns:
 * <ul>
 *   <li><b>Meta</b> — content understanding catalogs (Reels/IG media tables) with
 *       embedding columns for retrieval</li>
 *   <li><b>ByteDance</b> — short-video item profile (cover, ASR text, visual emb)</li>
 *   <li><b>Google</b> — YouTube / Photos style media index with duration, resolution</li>
 *   <li><b>Apple</b> — on-device media library metadata (local, privacy-preserving)</li>
 *   <li><b>Tencent</b> — WeSee multi-modal item features</li>
 * </ul>
 *
 * <p>Binary payloads stay on object storage / local FS; DuckDB stores paths,
 * technical metadata, labels, and {@code FLOAT[]} embeddings for ANN prefilter
 * / analytics. Pair with Lance / Faiss for large-scale vector search.
 *
 * <pre>{@code
 * try (DuckDB db = DuckDB.inMemory(DuckDBConfig.multimodalCatalog())) {
 *     DuckDBMultimodal mm = new DuckDBMultimodal(db);
 *     mm.ensureSchema();
 *     mm.upsertImage("img_1", "/data/a.jpg", 1024, 768, emb, new String[]{"cat"});
 *     mm.upsertVideo("vid_1", "/data/v.mp4", 60_000, 30.0, 1920, 1080, vEmb, null);
 *     DataFrame hits = mm.searchByModality("image", 100);
 *     DataFrame near = mm.bruteForceTopK(queryEmb, "image", 10);
 * }
 * }</pre>
 */
public final class DuckDBMultimodal {

    public static final String CATALOG = "media_catalog";
    public static final String FRAMES = "media_frames";
    public static final String TEXT_DOCS = "text_docs";
    public static final String AUDIO_SEGMENTS = "audio_segments";

    private final DuckDB db;

    public DuckDBMultimodal(DuckDB db) {
        this.db = Objects.requireNonNull(db, "db");
    }

    public DuckDB db() {
        return db;
    }

    public void ensureSchema() throws SQLException {
        db.ensureMediaCatalog();
        String td = DuckDB.sanitizeIdent(TEXT_DOCS);
        db.execute("CREATE TABLE IF NOT EXISTS " + td + " ("
                + " doc_id     VARCHAR PRIMARY KEY,"
                + " media_id   VARCHAR,"
                + " lang       VARCHAR,"
                + " title      VARCHAR,"
                + " body       VARCHAR,"
                + " tokens     VARCHAR[],"
                + " embedding  FLOAT[],"
                + " meta_json  JSON,"
                + " updated_at TIMESTAMP DEFAULT current_timestamp"
                + ")");
        String as = DuckDB.sanitizeIdent(AUDIO_SEGMENTS);
        db.execute("CREATE TABLE IF NOT EXISTS " + as + " ("
                + " media_id   VARCHAR,"
                + " seg_idx    INTEGER,"
                + " start_ms   BIGINT,"
                + " end_ms     BIGINT,"
                + " transcript VARCHAR,"
                + " embedding  FLOAT[],"
                + " PRIMARY KEY (media_id, seg_idx)"
                + ")");
    }

    // ---- upserts -----------------------------------------------------------

    public void upsertImage(String mediaId, String uri, Integer width, Integer height,
                            float[] embedding, String[] labels) throws SQLException {
        upsertMedia(mediaId, "image", uri, width, height, null, null, null, null,
                null, embedding, labels, null);
    }

    public void upsertAudio(String mediaId, String uri, Long durationMs,
                            Integer sampleRate, Integer channels,
                            float[] embedding, String[] labels) throws SQLException {
        upsertMedia(mediaId, "audio", uri, null, null, durationMs, sampleRate, channels,
                null, null, embedding, labels, null);
    }

    public void upsertVideo(String mediaId, String uri, Long durationMs, Double fps,
                            Integer width, Integer height,
                            float[] embedding, String[] labels) throws SQLException {
        // fps stored in meta_json lightly via codec field optional
        upsertMedia(mediaId, "video", uri, width, height, durationMs, null, null,
                fps == null ? null : ("fps=" + fps), null, embedding, labels, null);
    }

    public void upsertText(String docId, String mediaId, String lang, String title,
                           String body, float[] embedding) throws SQLException {
        ensureSchema();
        // Embed list literal in SQL — DuckDB JDBC binds FLOAT[] more reliably this way
        String embLit = embedding == null ? "NULL" : toFloatListLiteral(embedding) + "::FLOAT[]";
        String sql = "INSERT INTO " + DuckDB.sanitizeIdent(TEXT_DOCS)
                + " (doc_id, media_id, lang, title, body, embedding, updated_at) "
                + "VALUES (?, ?, ?, ?, ?, " + embLit + ", now()) "
                + "ON CONFLICT (doc_id) DO UPDATE SET "
                + " media_id=excluded.media_id, lang=excluded.lang, title=excluded.title, "
                + " body=excluded.body, embedding=excluded.embedding, "
                + " updated_at=now()";
        db.executeUpdate(sql, docId, mediaId, lang, title, body);
    }

    public void upsertMedia(String mediaId, String modality, String uri,
                            Integer width, Integer height, Long durationMs,
                            Integer sampleRate, Integer channels, String codec,
                            Long bytes, float[] embedding, String[] labels,
                            String metaJson) throws SQLException {
        ensureSchema();
        Objects.requireNonNull(mediaId, "mediaId");
        Objects.requireNonNull(modality, "modality");
        String mod = modality.toLowerCase(Locale.ROOT);
        String embLit = embedding == null ? "NULL" : toFloatListLiteral(embedding) + "::FLOAT[]";
        String labLit = labels == null ? "NULL" : toVarcharListLiteral(labels) + "::VARCHAR[]";
        String metaLit = metaJson == null ? "NULL" : ("'" + metaJson.replace("'", "''") + "'::JSON");
        // Keep complex types as SQL literals; scalars as bound params (JDBC-safe).
        String sql = "INSERT INTO " + DuckDB.sanitizeIdent(CATALOG)
                + " (media_id, modality, uri, width, height, duration_ms, sample_rate, "
                + "  channels, codec, bytes, embedding, labels, meta_json, updated_at) "
                + "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, " + embLit + ", " + labLit + ", "
                + metaLit + ", now()) "
                + "ON CONFLICT (media_id) DO UPDATE SET "
                + " modality=excluded.modality, uri=excluded.uri, width=excluded.width, "
                + " height=excluded.height, duration_ms=excluded.duration_ms, "
                + " sample_rate=excluded.sample_rate, channels=excluded.channels, "
                + " codec=excluded.codec, bytes=excluded.bytes, "
                + " embedding=excluded.embedding, labels=excluded.labels, "
                + " meta_json=excluded.meta_json, updated_at=now()";
        db.executeUpdate(sql, mediaId, mod, uri, width, height, durationMs, sampleRate, channels,
                codec, bytes);
    }

    public void addFrame(String mediaId, int frameIdx, Long ptsMs, String uri,
                         float[] embedding) throws SQLException {
        ensureSchema();
        String embLit = embedding == null ? "NULL" : toFloatListLiteral(embedding) + "::FLOAT[]";
        db.executeUpdate(
                "INSERT INTO " + DuckDB.sanitizeIdent(FRAMES)
                        + " (media_id, frame_idx, pts_ms, uri, embedding) VALUES (?, ?, ?, ?, "
                        + embLit + ") "
                        + "ON CONFLICT (media_id, frame_idx) DO UPDATE SET "
                        + " pts_ms=excluded.pts_ms, uri=excluded.uri, embedding=excluded.embedding",
                mediaId, frameIdx, ptsMs, uri);
    }

    public void addAudioSegment(String mediaId, int segIdx, long startMs, long endMs,
                                String transcript, float[] embedding) throws SQLException {
        ensureSchema();
        String embLit = embedding == null ? "NULL" : toFloatListLiteral(embedding) + "::FLOAT[]";
        db.executeUpdate(
                "INSERT INTO " + DuckDB.sanitizeIdent(AUDIO_SEGMENTS)
                        + " (media_id, seg_idx, start_ms, end_ms, transcript, embedding) "
                        + "VALUES (?, ?, ?, ?, ?, " + embLit + ") "
                        + "ON CONFLICT (media_id, seg_idx) DO UPDATE SET "
                        + " start_ms=excluded.start_ms, end_ms=excluded.end_ms, "
                        + " transcript=excluded.transcript, embedding=excluded.embedding",
                mediaId, segIdx, startMs, endMs, transcript);
    }

    // ---- queries -----------------------------------------------------------

    public DataFrame searchByModality(String modality, int limit) throws Exception {
        ensureSchema();
        return db.query(
                "SELECT media_id, modality, uri, width, height, duration_ms, "
                        + "sample_rate, channels, codec, bytes, labels, updated_at "
                        + "FROM " + DuckDB.sanitizeIdent(CATALOG)
                        + " WHERE modality = ? ORDER BY updated_at DESC LIMIT ?",
                modality.toLowerCase(Locale.ROOT), Math.max(1, limit));
    }

    public DataFrame getMedia(String mediaId) throws Exception {
        ensureSchema();
        return db.query("SELECT * FROM " + DuckDB.sanitizeIdent(CATALOG)
                + " WHERE media_id = ?", mediaId);
    }

    public DataFrame listFrames(String mediaId) throws Exception {
        ensureSchema();
        return db.query("SELECT * FROM " + DuckDB.sanitizeIdent(FRAMES)
                + " WHERE media_id = ? ORDER BY frame_idx", mediaId);
    }

    public DataFrame fullTextSearch(String query, int limit) throws Exception {
        ensureSchema();
        // ILIKE fallback — production may load fts extension
        String q = "%" + query + "%";
        return db.query(
                "SELECT doc_id, media_id, lang, title, "
                        + "substr(body, 1, 200) AS snippet, updated_at "
                        + "FROM " + DuckDB.sanitizeIdent(TEXT_DOCS)
                        + " WHERE title ILIKE ? OR body ILIKE ? "
                        + "ORDER BY updated_at DESC LIMIT ?",
                q, q, Math.max(1, limit));
    }

    /**
     * Brute-force cosine top-K over catalog embeddings (small/medium catalogs).
     * For large-scale ANN use Lance/Faiss; this is the honest SQL baseline.
     */
    public DataFrame bruteForceTopK(float[] query, String modality, int k) throws Exception {
        Objects.requireNonNull(query, "query");
        ensureSchema();
        int topK = Math.max(1, k);
        // Register query as temp single-row table via SQL list literal
        String lit = toFloatListLiteral(query);
        String modFilter = modality == null || modality.isBlank()
                ? "TRUE"
                : "modality = '" + modality.toLowerCase(Locale.ROOT).replace("'", "''") + "'";
        String sql = "WITH q AS (SELECT " + lit + "::FLOAT[] AS v) "
                + "SELECT c.media_id, c.modality, c.uri, c.labels, "
                + "  list_cosine_similarity(c.embedding, q.v) AS score "
                + "FROM " + DuckDB.sanitizeIdent(CATALOG) + " c, q "
                + "WHERE c.embedding IS NOT NULL AND " + modFilter + " "
                + "ORDER BY score DESC NULLS LAST LIMIT " + topK;
        try {
            return db.query(sql);
        } catch (Exception ex) {
            // Older DuckDB: manual cosine
            String sql2 = "WITH q AS (SELECT " + lit + "::FLOAT[] AS v) "
                    + "SELECT c.media_id, c.modality, c.uri, "
                    + "  (list_dot_product(c.embedding, q.v) / "
                    + "   (nullif(sqrt(list_dot_product(c.embedding, c.embedding)),0) "
                    + "  * nullif(sqrt(list_dot_product(q.v, q.v)),0))) AS score "
                    + "FROM " + DuckDB.sanitizeIdent(CATALOG) + " c, q "
                    + "WHERE c.embedding IS NOT NULL AND " + modFilter + " "
                    + "ORDER BY score DESC NULLS LAST LIMIT " + topK;
            return db.query(sql2);
        }
    }

    public DataFrame modalityStats() throws Exception {
        ensureSchema();
        return db.query(
                "SELECT modality, count(*) AS n, "
                        + "avg(duration_ms) AS avg_duration_ms, "
                        + "avg(width) AS avg_width, avg(height) AS avg_height, "
                        + "sum(CASE WHEN embedding IS NULL THEN 1 ELSE 0 END) AS missing_emb "
                        + "FROM " + DuckDB.sanitizeIdent(CATALOG) + " GROUP BY 1 ORDER BY n DESC");
    }

    public void ingestCatalogParquet(String path) throws SQLException {
        ensureSchema();
        db.execute("INSERT INTO " + DuckDB.sanitizeIdent(CATALOG)
                + " SELECT * FROM read_parquet('" + DuckDB.escapePath(path) + "')");
    }

    public void exportCatalogParquet(String path) throws SQLException {
        db.exportParquet(CATALOG, path);
    }

    public long count(String modality) throws Exception {
        DataFrame df = modality == null
                ? db.query("SELECT count(*) AS c FROM " + DuckDB.sanitizeIdent(CATALOG))
                : db.query("SELECT count(*) AS c FROM " + DuckDB.sanitizeIdent(CATALOG)
                + " WHERE modality = ?", modality.toLowerCase(Locale.ROOT));
        if (df.rowCount() == 0) return 0L;
        Object v = df.get(0, "c");
        return v instanceof Number ? ((Number) v).longValue() : 0L;
    }

    static String toFloatListLiteral(float[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(", ");
            float x = v[i];
            if (Float.isNaN(x) || Float.isInfinite(x)) sb.append("0");
            else sb.append(x);
        }
        sb.append(']');
        return sb.toString();
    }

    static String toVarcharListLiteral(String[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) sb.append(", ");
            if (v[i] == null) {
                sb.append("NULL");
            } else {
                sb.append('\'').append(v[i].replace("'", "''")).append('\'');
            }
        }
        sb.append(']');
        return sb.toString();
    }

    public static String[] commonImageLabelsExample() {
        return Arrays.asList("person", "product", "outdoor", "text_overlay").toArray(new String[0]);
    }
}
