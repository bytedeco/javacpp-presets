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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Offline / nearline <b>feature store</b> helpers on DuckDB for search / rec / ads.
 *
 * <p>Design mirrors patterns used at scale (public literature, not internal IP):
 * <ul>
 *   <li><b>Meta</b> Feature Store / FBLearner — entity keyed feature tables,
 *       point-in-time correct joins, training dump to Parquet</li>
 *   <li><b>ByteDance</b> lagrange / monoseq — event log scan, sequence
 *       aggregation, negative sampling for CTR/CVR</li>
 *   <li><b>Google</b> TFX / Feast-style — feature views, as-of joins,
 *       batch materialization</li>
 *   <li><b>Tencent</b> WeSee / ads — dense+sparse feature assemble,
 *       funnel windows for ranking eval</li>
 * </ul>
 *
 * <p>All heavy lifting is SQL on DuckDB (vectorized, parquet-native). This class
 * only generates safe SQL and thin DataFrame I/O — no fabricated proprietary
 * protocols.
 *
 * <pre>{@code
 * try (DuckDB db = DuckDB.open(path, DuckDBConfig.offlineFeatureEngineering())) {
 *     DuckDBFeatureStore fs = new DuckDBFeatureStore(db);
 *     fs.ensureEventLog("events");
 *     fs.ingestParquetEvents("events", "logs/dt=star/star.parquet", true);
 *     fs.materializeUserAgg("events", "user_feat", "user_id", "ts", 7);
 *     DataFrame train = fs.pointInTimeJoin(
 *         "events", "user_feat", "user_id", "ts", "updated_at",
 *         List.of("clk_7d", "ord_7d"));
 *     fs.exportTrainingParquet(train, "train/");
 * }
 * }</pre>
 */
public final class DuckDBFeatureStore {

    private final DuckDB db;

    public DuckDBFeatureStore(DuckDB db) {
        this.db = Objects.requireNonNull(db, "db");
    }

    public DuckDB db() {
        return db;
    }

    // ---- schema templates --------------------------------------------------

    /**
     * Standard interaction event log (CTR/CVR/ ranking impressions).
     * Columns cover Criteo / Avazu / AliExpress-style sparse + dense + label.
     */
    public void ensureEventLog(String table) throws SQLException {
        String t = DuckDB.sanitizeIdent(table);
        db.execute("CREATE TABLE IF NOT EXISTS " + t + " ("
                + " event_id    VARCHAR,"
                + " user_id     BIGINT,"
                + " item_id     BIGINT,"
                + " ts          TIMESTAMP,"
                + " dt          DATE,"
                + " label       INTEGER,"
                + " label_cvr   INTEGER,"
                + " scene       VARCHAR,"
                + " request_id  VARCHAR,"
                + " pos         INTEGER,"
                + " dense       FLOAT[],"
                + " sparse      BIGINT[],"
                + " seq_items   BIGINT[],"
                + " meta_json   JSON"
                + ")");
    }

    /** Entity feature table (user / item / author / query). */
    public void ensureEntityFeatures(String table, String entityCol) throws SQLException {
        String t = DuckDB.sanitizeIdent(table);
        String e = DuckDB.sanitizeIdent(entityCol == null ? "entity_id" : entityCol);
        db.execute("CREATE TABLE IF NOT EXISTS " + t + " ("
                + " " + e + " BIGINT,"
                + " updated_at TIMESTAMP,"
                + " dense      FLOAT[],"
                + " cat0       INTEGER, cat1 INTEGER, cat2 INTEGER, cat3 INTEGER,"
                + " emb        FLOAT[],"
                + " meta_json  JSON,"
                + " PRIMARY KEY (" + e + ")"
                + ")");
    }

    /** Sequence feature table (user recent items / search queries). */
    public void ensureSequenceTable(String table) throws SQLException {
        String t = DuckDB.sanitizeIdent(table);
        db.execute("CREATE TABLE IF NOT EXISTS " + t + " ("
                + " user_id    BIGINT,"
                + " updated_at TIMESTAMP,"
                + " item_ids   BIGINT[],"
                + " timestamps BIGINT[],"
                + " actions    INTEGER[],"
                + " PRIMARY KEY (user_id)"
                + ")");
    }

    // ---- ingest ------------------------------------------------------------

    public void ingestParquetEvents(String table, String parquetGlob, boolean hivePartitioning)
            throws SQLException {
        db.registerParquet("_src_events_" + table, parquetGlob, hivePartitioning);
        String t = DuckDB.sanitizeIdent(table);
        // INSERT by column name intersection is engine-side; here we select *
        // and rely on matching schemas — callers can customize via raw SQL.
        db.execute("INSERT INTO " + t + " SELECT * FROM "
                + DuckDB.sanitizeIdent("_src_events_" + table));
    }

    public long ingestDataFrame(String table, DataFrame df, boolean replace) throws Exception {
        if (replace) return db.replaceWithDataFrame(table, df);
        return db.appendDataFrame(table, df, true);
    }

    // ---- aggregations / materialization ------------------------------------

    /**
     * Rolling user aggregates over the last {@code windowDays} (count, sum label,
     * CTR proxy). Writes/replaces {@code outTable}.
     */
    public void materializeUserAgg(String eventTable, String outTable,
                                   String userCol, String tsCol, int windowDays)
            throws SQLException {
        String e = DuckDB.sanitizeIdent(eventTable);
        String o = DuckDB.sanitizeIdent(outTable);
        String u = DuckDB.sanitizeIdent(userCol);
        String ts = DuckDB.sanitizeIdent(tsCol);
        int w = Math.max(1, windowDays);
        db.execute("CREATE OR REPLACE TABLE " + o + " AS "
                + "SELECT " + u + " AS user_id, "
                + "  max(" + ts + ") AS updated_at, "
                + "  count(*)::BIGINT AS impress_" + w + "d, "
                + "  sum(COALESCE(label,0))::BIGINT AS click_" + w + "d, "
                + "  CASE WHEN count(*)=0 THEN 0.0 "
                + "       ELSE sum(COALESCE(label,0))::DOUBLE / count(*) END AS ctr_" + w + "d, "
                + "  count(DISTINCT item_id)::BIGINT AS uniq_item_" + w + "d "
                + "FROM " + e + " "
                + "WHERE " + ts + " >= (current_timestamp - INTERVAL '" + w + " days') "
                + "GROUP BY 1");
    }

    /**
     * Item-side aggregates (popularity, CTR) — classic cold-start / retrieval features.
     */
    public void materializeItemAgg(String eventTable, String outTable,
                                   String itemCol, String tsCol, int windowDays)
            throws SQLException {
        String e = DuckDB.sanitizeIdent(eventTable);
        String o = DuckDB.sanitizeIdent(outTable);
        String it = DuckDB.sanitizeIdent(itemCol);
        String ts = DuckDB.sanitizeIdent(tsCol);
        int w = Math.max(1, windowDays);
        db.execute("CREATE OR REPLACE TABLE " + o + " AS "
                + "SELECT " + it + " AS item_id, "
                + "  max(" + ts + ") AS updated_at, "
                + "  count(*)::BIGINT AS impress_" + w + "d, "
                + "  sum(COALESCE(label,0))::BIGINT AS click_" + w + "d, "
                + "  CASE WHEN count(*)=0 THEN 0.0 "
                + "       ELSE sum(COALESCE(label,0))::DOUBLE / count(*) END AS ctr_" + w + "d, "
                + "  count(DISTINCT user_id)::BIGINT AS uniq_user_" + w + "d "
                + "FROM " + e + " "
                + "WHERE " + ts + " >= (current_timestamp - INTERVAL '" + w + " days') "
                + "GROUP BY 1");
    }

    /**
     * Build padded user sequences (last {@code maxLen} items before each event is
     * typically done per-row; this materializes the latest sequence per user).
     */
    public void materializeUserSequence(String eventTable, String outTable,
                                        String userCol, String itemCol, String tsCol,
                                        int maxLen) throws SQLException {
        String e = DuckDB.sanitizeIdent(eventTable);
        String o = DuckDB.sanitizeIdent(outTable);
        String u = DuckDB.sanitizeIdent(userCol);
        String it = DuckDB.sanitizeIdent(itemCol);
        String ts = DuckDB.sanitizeIdent(tsCol);
        int m = Math.max(1, maxLen);
        // list aggregation + array_slice — DuckDB dialect
        db.execute("CREATE OR REPLACE TABLE " + o + " AS "
                + "SELECT " + u + " AS user_id, "
                + "  max(" + ts + ") AS updated_at, "
                + "  array_slice(list(" + it + " ORDER BY " + ts + " DESC), 1, " + m + ") AS item_ids, "
                + "  array_slice(list(epoch_ms(" + ts + ") ORDER BY " + ts + " DESC), 1, " + m + ") AS timestamps "
                + "FROM " + e + " "
                + "WHERE " + it + " IS NOT NULL "
                + "GROUP BY 1");
    }

    // ---- point-in-time join ------------------------------------------------

    /**
     * Point-in-time correct join: for each event row, attach the latest entity
     * feature row with {@code featureTsCol <= eventTsCol}.
     *
     * <p>Implements the classic Feast / Meta PIT join via ASOF JOIN (DuckDB).
     */
    public DataFrame pointInTimeJoin(String eventTable, String featureTable,
                                     String entityCol, String eventTsCol,
                                     String featureTsCol, List<String> featureCols)
            throws Exception {
        Objects.requireNonNull(featureCols, "featureCols");
        String e = DuckDB.sanitizeIdent(eventTable);
        String f = DuckDB.sanitizeIdent(featureTable);
        String ent = DuckDB.sanitizeIdent(entityCol);
        String ets = DuckDB.sanitizeIdent(eventTsCol);
        String fts = DuckDB.sanitizeIdent(featureTsCol);
        String feats = featureCols.stream()
                .map(c -> "f." + DuckDB.sanitizeIdent(c))
                .collect(Collectors.joining(", "));
        if (feats.isEmpty()) feats = "f.*";
        String sql = "SELECT e.*, " + feats + " FROM " + e + " e "
                + "ASOF LEFT JOIN " + f + " f "
                + "ON e." + ent + " = f." + ent + " AND e." + ets + " >= f." + fts;
        return db.query(sql);
    }

    /**
     * Simple broadcast join of pre-aggregated user + item features onto events
     * (no time travel — use when features are snapshot-static for the batch).
     */
    public DataFrame assembleBatch(String eventTable, String userFeatTable, String itemFeatTable,
                                   String userCol, String itemCol,
                                   List<String> userCols, List<String> itemCols)
            throws Exception {
        String e = DuckDB.sanitizeIdent(eventTable);
        String u = userFeatTable == null ? null : DuckDB.sanitizeIdent(userFeatTable);
        String it = itemFeatTable == null ? null : DuckDB.sanitizeIdent(itemFeatTable);
        String uc = DuckDB.sanitizeIdent(userCol);
        String ic = DuckDB.sanitizeIdent(itemCol);
        StringBuilder sql = new StringBuilder("SELECT e.*");
        if (u != null && userCols != null) {
            for (String c : userCols) {
                sql.append(", uf.").append(DuckDB.sanitizeIdent(c))
                        .append(" AS ").append(DuckDB.sanitizeIdent("u_" + c));
            }
        }
        if (it != null && itemCols != null) {
            for (String c : itemCols) {
                sql.append(", itf.").append(DuckDB.sanitizeIdent(c))
                        .append(" AS ").append(DuckDB.sanitizeIdent("i_" + c));
            }
        }
        sql.append(" FROM ").append(e).append(" e ");
        if (u != null) {
            sql.append("LEFT JOIN ").append(u).append(" uf ON e.").append(uc)
                    .append(" = uf.").append(uc).append(' ');
        }
        if (it != null) {
            sql.append("LEFT JOIN ").append(it).append(" itf ON e.").append(ic)
                    .append(" = itf.").append(ic).append(' ');
        }
        return db.query(sql.toString());
    }

    // ---- negative sampling -------------------------------------------------

    /**
     * In-batch uniform negative sampling for retrieval / two-tower training.
     * Produces {@code numNeg} random items per positive that are not the positive item.
     *
     * <p>Algorithm: cross join positives with random item sample, filter collisions.
     * Suitable for offline batch (ByteDance/Google two-tower training dumps).
     */
    public DataFrame uniformNegativeSample(String eventTable, String itemCatalogTable,
                                           String userCol, String itemCol,
                                           int numNeg, long seed) throws Exception {
        if (numNeg < 1) throw new IllegalArgumentException("numNeg >= 1");
        String e = DuckDB.sanitizeIdent(eventTable);
        String cat = DuckDB.sanitizeIdent(itemCatalogTable);
        String u = DuckDB.sanitizeIdent(userCol);
        String it = DuckDB.sanitizeIdent(itemCol);
        // Use hash-based pseudo random for reproducibility
        String sql = "WITH pos AS ("
                + "  SELECT " + u + " AS user_id, " + it + " AS pos_item, "
                + "         row_number() OVER () AS rid FROM " + e
                + "  WHERE COALESCE(label,1) > 0"
                + "), items AS ("
                + "  SELECT " + it + " AS item_id FROM " + cat
                + "), neg AS ("
                + "  SELECT p.user_id, p.pos_item, i.item_id AS neg_item, p.rid, "
                + "    row_number() OVER (PARTITION BY p.rid ORDER BY hash(p.rid, i.item_id, "
                + seed + ")) AS rn "
                + "  FROM pos p CROSS JOIN items i "
                + "  WHERE i.item_id IS DISTINCT FROM p.pos_item"
                + ") "
                + "SELECT user_id, pos_item, neg_item, 0 AS label FROM neg WHERE rn <= " + numNeg
                + " UNION ALL "
                + "SELECT user_id, pos_item, pos_item AS neg_item, 1 AS label FROM pos";
        return db.query(sql);
    }

    /**
     * Popularity-weighted negative sampling using precomputed item frequency table
     * with columns {@code item_id, weight}.
     */
    public DataFrame popularityNegativeSample(String eventTable, String itemWeightTable,
                                              String userCol, String itemCol,
                                              int numNeg, long seed) throws Exception {
        if (numNeg < 1) throw new IllegalArgumentException("numNeg >= 1");
        String e = DuckDB.sanitizeIdent(eventTable);
        String w = DuckDB.sanitizeIdent(itemWeightTable);
        String u = DuckDB.sanitizeIdent(userCol);
        String it = DuckDB.sanitizeIdent(itemCol);
        String sql = "WITH pos AS ("
                + "  SELECT " + u + " AS user_id, " + it + " AS pos_item, "
                + "         row_number() OVER () AS rid FROM " + e
                + "  WHERE COALESCE(label,1) > 0"
                + "), weighted AS ("
                + "  SELECT item_id, weight, "
                + "    sum(weight) OVER () AS z, "
                + "    sum(weight) OVER (ORDER BY item_id) AS cum "
                + "  FROM " + w
                + "), neg AS ("
                + "  SELECT p.user_id, p.pos_item, wt.item_id AS neg_item, p.rid, "
                + "    row_number() OVER (PARTITION BY p.rid ORDER BY "
                + "      hash(p.rid, wt.item_id, " + seed + ") % 1000000 / 1000000.0 "
                + "      * log(1 + wt.weight)) AS rn "
                + "  FROM pos p CROSS JOIN weighted wt "
                + "  WHERE wt.item_id IS DISTINCT FROM p.pos_item"
                + ") "
                + "SELECT user_id, pos_item, neg_item, 0 AS label FROM neg WHERE rn <= " + numNeg
                + " UNION ALL "
                + "SELECT user_id, pos_item, pos_item, 1 FROM pos";
        return db.query(sql);
    }

    // ---- train / eval splits -----------------------------------------------

    /** Time-based split: train where ts &lt; splitTs, eval otherwise. */
    public void timeSplit(String eventTable, String trainTable, String evalTable,
                          String tsCol, String splitTs) throws SQLException {
        String e = DuckDB.sanitizeIdent(eventTable);
        String tr = DuckDB.sanitizeIdent(trainTable);
        String ev = DuckDB.sanitizeIdent(evalTable);
        String ts = DuckDB.sanitizeIdent(tsCol);
        String boundary = splitTs.replace("'", "''");
        db.execute("CREATE OR REPLACE TABLE " + tr + " AS SELECT * FROM " + e
                + " WHERE " + ts + " < TIMESTAMP '" + boundary + "'");
        db.execute("CREATE OR REPLACE TABLE " + ev + " AS SELECT * FROM " + e
                + " WHERE " + ts + " >= TIMESTAMP '" + boundary + "'");
    }

    /** Random split by hash of entity (user-level holdout avoids leakage). */
    public void userHashSplit(String eventTable, String trainTable, String evalTable,
                              String userCol, double evalRatio, long seed)
            throws SQLException {
        if (evalRatio <= 0 || evalRatio >= 1) {
            throw new IllegalArgumentException("evalRatio in (0,1)");
        }
        String e = DuckDB.sanitizeIdent(eventTable);
        String tr = DuckDB.sanitizeIdent(trainTable);
        String ev = DuckDB.sanitizeIdent(evalTable);
        String u = DuckDB.sanitizeIdent(userCol);
        // hash to [0,1)
        String bucket = "(hash(" + u + ", " + seed + ") % 1000000) / 1000000.0";
        db.execute("CREATE OR REPLACE TABLE " + ev + " AS SELECT * FROM " + e
                + " WHERE " + bucket + " < " + evalRatio);
        db.execute("CREATE OR REPLACE TABLE " + tr + " AS SELECT * FROM " + e
                + " WHERE " + bucket + " >= " + evalRatio);
    }

    // ---- export ------------------------------------------------------------

    public void exportTrainingParquet(String tableOrSql, String path) throws SQLException {
        db.exportParquet(tableOrSql, path);
    }

    public void exportTrainingParquet(DataFrame df, String path) throws Exception {
        db.exportParquet(df, path);
    }

    public void exportTrainingPartitioned(String table, String dir, String... partitionCols)
            throws SQLException {
        db.exportParquetPartitioned(table, dir, partitionCols);
    }

    // ---- feature stats / quality -------------------------------------------

    /** Null rate + basic stats for numeric feature columns (data QA). */
    public DataFrame featureQualityReport(String table, List<String> cols) throws Exception {
        String t = DuckDB.sanitizeIdent(table);
        if (cols == null || cols.isEmpty()) {
            return db.summarize(table);
        }
        List<String> parts = new ArrayList<>();
        for (String c : cols) {
            String id = DuckDB.sanitizeIdent(c);
            parts.add("count(*) AS n");
            parts.add("sum(CASE WHEN " + id + " IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS "
                    + DuckDB.sanitizeIdent(c + "_null_rate"));
            parts.add("avg(try_cast(" + id + " AS DOUBLE)) AS " + DuckDB.sanitizeIdent(c + "_mean"));
            parts.add("stddev_samp(try_cast(" + id + " AS DOUBLE)) AS "
                    + DuckDB.sanitizeIdent(c + "_std"));
        }
        // dedupe count(*)
        List<String> uniq = new ArrayList<>();
        uniq.add("count(*) AS n");
        for (String c : cols) {
            String id = DuckDB.sanitizeIdent(c);
            uniq.add("sum(CASE WHEN " + id + " IS NULL THEN 1 ELSE 0 END)::DOUBLE / count(*) AS "
                    + DuckDB.sanitizeIdent(c + "_null_rate"));
            uniq.add("avg(try_cast(" + id + " AS DOUBLE)) AS " + DuckDB.sanitizeIdent(c + "_mean"));
            uniq.add("stddev_samp(try_cast(" + id + " AS DOUBLE)) AS "
                    + DuckDB.sanitizeIdent(c + "_std"));
        }
        return db.query("SELECT " + String.join(", ", uniq) + " FROM " + t);
    }

    /** Top-N sparse IDs by frequency (vocab mining for embedding tables). */
    public DataFrame topSparseIds(String table, String sparseCol, int topN) throws Exception {
        String t = DuckDB.sanitizeIdent(table);
        String c = DuckDB.sanitizeIdent(sparseCol);
        int n = Math.max(1, topN);
        // If column is a list, unnest; else treat as scalar id
        String sql = "SELECT id, count(*) AS cnt FROM ("
                + "  SELECT UNNEST(" + c + ") AS id FROM " + t
                + "  WHERE " + c + " IS NOT NULL"
                + "  UNION ALL "
                + "  SELECT " + c + " AS id FROM " + t
                + "  WHERE " + c + " IS NOT NULL AND typeof(" + c + ") NOT LIKE '%%[]%%'"
                + ") GROUP BY 1 ORDER BY cnt DESC LIMIT " + n;
        try {
            return db.query(sql);
        } catch (Exception ex) {
            // fallback scalar-only
            return db.query("SELECT " + c + " AS id, count(*) AS cnt FROM " + t
                    + " WHERE " + c + " IS NOT NULL GROUP BY 1 ORDER BY cnt DESC LIMIT " + n);
        }
    }

    /** Convenience: list of feature column names excluding keys/labels. */
    public static List<String> defaultExcludeFromFeatures() {
        return Arrays.asList("event_id", "user_id", "item_id", "ts", "dt", "label",
                "label_cvr", "request_id", "scene", "pos", "meta_json");
    }
}
