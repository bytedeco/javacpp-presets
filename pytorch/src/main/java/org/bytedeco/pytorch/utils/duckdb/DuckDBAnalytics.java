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

import java.util.Objects;

/**
 * Ranking / growth analytics SQL helpers on DuckDB for search–rec–ads evaluation.
 *
 * <p>Implements well-documented public metrics and funnels (not proprietary code):
 * <ul>
 *   <li>CTR / CVR / eCPM-style aggregates</li>
 *   <li>Funnel conversion (impression → click → convert)</li>
 *   <li>Cohort retention by registration / first-active week</li>
 *   <li>AUC-friendly score-label extracts; GAUC grouping keys</li>
 *   <li>NDCG/MAP input tables (relevance + rank position)</li>
 *   <li>A/B experiment slice aggregates (bucket × metric)</li>
 * </ul>
 *
 * <p>References (public): Google rank-eval practices, Meta ranking calibration
 * blogs, Tencent/ByteDance open talks on offline metrics — formulas only.
 */
public final class DuckDBAnalytics {

    private final DuckDB db;

    public DuckDBAnalytics(DuckDB db) {
        this.db = Objects.requireNonNull(db, "db");
    }

    public DuckDB db() {
        return db;
    }

    // ---- basic rate metrics ------------------------------------------------

    /**
     * Daily CTR / CVR from an event table with {@code dt}, {@code label}, optional {@code label_cvr}.
     */
    public DataFrame dailyCtrCvr(String eventTable) throws Exception {
        String t = DuckDB.sanitizeIdent(eventTable);
        return db.query(
                "SELECT dt, "
                        + "  count(*) AS impress, "
                        + "  sum(COALESCE(label,0)) AS clicks, "
                        + "  sum(COALESCE(label_cvr,0)) AS converts, "
                        + "  sum(COALESCE(label,0))::DOUBLE / nullif(count(*),0) AS ctr, "
                        + "  sum(COALESCE(label_cvr,0))::DOUBLE / nullif(sum(COALESCE(label,0)),0) AS cvr, "
                        + "  sum(COALESCE(label_cvr,0))::DOUBLE / nullif(count(*),0) AS ctcvr "
                        + "FROM " + t + " GROUP BY dt ORDER BY dt");
    }

    public DataFrame sceneCtr(String eventTable, String sceneCol) throws Exception {
        String t = DuckDB.sanitizeIdent(eventTable);
        String s = DuckDB.sanitizeIdent(sceneCol == null ? "scene" : sceneCol);
        return db.query(
                "SELECT " + s + " AS scene, count(*) AS impress, "
                        + "sum(COALESCE(label,0)) AS clicks, "
                        + "sum(COALESCE(label,0))::DOUBLE / nullif(count(*),0) AS ctr "
                        + "FROM " + t + " GROUP BY 1 ORDER BY impress DESC");
    }

    // ---- funnel ------------------------------------------------------------

    /**
     * Simple 3-step funnel counts by day from boolean/int flag columns.
     *
     * @param steps ordered step column names that are truthy when reached
     */
    public DataFrame funnelDaily(String eventTable, String dtCol, String... steps)
            throws Exception {
        if (steps == null || steps.length == 0) {
            throw new IllegalArgumentException("steps required");
        }
        String t = DuckDB.sanitizeIdent(eventTable);
        String dt = DuckDB.sanitizeIdent(dtCol == null ? "dt" : dtCol);
        StringBuilder sb = new StringBuilder("SELECT ").append(dt).append(" AS dt, count(*) AS n0");
        for (int i = 0; i < steps.length; i++) {
            String c = DuckDB.sanitizeIdent(steps[i]);
            sb.append(", sum(CASE WHEN COALESCE(").append(c)
                    .append(",0) > 0 THEN 1 ELSE 0 END) AS n").append(i + 1);
        }
        sb.append(" FROM ").append(t).append(" GROUP BY 1 ORDER BY 1");
        return db.query(sb.toString());
    }

    // ---- cohort retention --------------------------------------------------

    /**
     * Weekly retention: cohort = date_trunc('week', first_ts); measure active in +1..N weeks.
     */
    public DataFrame weeklyRetention(String activityTable, String userCol, String tsCol,
                                     int maxWeeks) throws Exception {
        String t = DuckDB.sanitizeIdent(activityTable);
        String u = DuckDB.sanitizeIdent(userCol);
        String ts = DuckDB.sanitizeIdent(tsCol);
        int w = Math.max(1, maxWeeks);
        String sql = "WITH firsts AS ("
                + "  SELECT " + u + " AS user_id, date_trunc('week', min(" + ts + ")) AS cohort "
                + "  FROM " + t + " GROUP BY 1"
                + "), acts AS ("
                + "  SELECT DISTINCT " + u + " AS user_id, date_trunc('week', " + ts + ") AS wk "
                + "  FROM " + t
                + "), joined AS ("
                + "  SELECT f.cohort, a.user_id, "
                + "    datediff('week', f.cohort, a.wk) AS week_n "
                + "  FROM firsts f JOIN acts a USING (user_id) "
                + "  WHERE datediff('week', f.cohort, a.wk) BETWEEN 0 AND " + w
                + ") "
                + "SELECT cohort, week_n, count(DISTINCT user_id) AS users "
                + "FROM joined GROUP BY 1, 2 ORDER BY 1, 2";
        return db.query(sql);
    }

    // ---- ranking eval extracts ---------------------------------------------

    /**
     * Build (group_id, label, score) for GAUC / grouped AUC computation offline.
     * Actual AUC is computed in Java/Python; this only extracts clean rows.
     */
    public DataFrame rankingScores(String table, String groupCol, String labelCol,
                                   String scoreCol) throws Exception {
        String t = DuckDB.sanitizeIdent(table);
        String g = DuckDB.sanitizeIdent(groupCol);
        String y = DuckDB.sanitizeIdent(labelCol);
        String s = DuckDB.sanitizeIdent(scoreCol);
        return db.query(
                "SELECT " + g + " AS group_id, " + y + " AS label, " + s + " AS score "
                        + "FROM " + t + " WHERE " + s + " IS NOT NULL AND " + y + " IS NOT NULL");
    }

    /**
     * Per-request ranked list with position for NDCG@K / MAP@K input.
     * Expects columns request_id, item_id, label (relevance), score; assigns rank.
     */
    public DataFrame rankedLists(String table, String requestCol, String labelCol,
                                 String scoreCol, int maxPos) throws Exception {
        String t = DuckDB.sanitizeIdent(table);
        String r = DuckDB.sanitizeIdent(requestCol);
        String y = DuckDB.sanitizeIdent(labelCol);
        String s = DuckDB.sanitizeIdent(scoreCol);
        int k = Math.max(1, maxPos);
        return db.query(
                "SELECT * FROM ("
                        + "  SELECT " + r + " AS request_id, item_id, " + y + " AS relevance, "
                        + "    " + s + " AS score, "
                        + "    row_number() OVER (PARTITION BY " + r + " ORDER BY " + s
                        + " DESC NULLS LAST) AS pos "
                        + "  FROM " + t
                        + ") x WHERE pos <= " + k + " ORDER BY request_id, pos");
    }

    /**
     * Approximate NDCG@K in pure SQL (binary relevance). Good for quick offline
     * dashboards; production metric libs remain authoritative for graded labels.
     */
    public DataFrame ndcgAtK(String rankedOrRawTable, String requestCol, String labelCol,
                             String scoreCol, int k) throws Exception {
        String t = DuckDB.sanitizeIdent(rankedOrRawTable);
        String r = DuckDB.sanitizeIdent(requestCol);
        String y = DuckDB.sanitizeIdent(labelCol);
        String s = DuckDB.sanitizeIdent(scoreCol);
        int kk = Math.max(1, k);
        String sql = "WITH ranked AS ("
                + "  SELECT " + r + " AS request_id, " + y + " AS rel, "
                + "    row_number() OVER (PARTITION BY " + r + " ORDER BY " + s
                + " DESC NULLS LAST) AS pos "
                + "  FROM " + t
                + "), dcg AS ("
                + "  SELECT request_id, "
                + "    sum(CASE WHEN pos <= " + kk + " THEN "
                + "      (CASE WHEN rel > 0 THEN 1.0 ELSE 0.0 END) "
                + "      / log2(pos + 1) ELSE 0 END) AS dcg "
                + "  FROM ranked GROUP BY 1"
                + "), ideal AS ("
                + "  SELECT request_id, "
                + "    sum(CASE WHEN ipos <= " + kk + " THEN 1.0 / log2(ipos + 1) ELSE 0 END) AS idcg "
                + "  FROM ("
                + "    SELECT request_id, "
                + "      row_number() OVER (PARTITION BY request_id ORDER BY rel DESC) AS ipos, rel "
                + "    FROM ranked WHERE rel > 0"
                + "  ) z GROUP BY 1"
                + ") "
                + "SELECT d.request_id, d.dcg, COALESCE(i.idcg, 0) AS idcg, "
                + "  CASE WHEN COALESCE(i.idcg,0)=0 THEN 0 "
                + "       ELSE d.dcg / i.idcg END AS ndcg "
                + "FROM dcg d LEFT JOIN ideal i USING (request_id)";
        return db.query(sql);
    }

    // ---- A/B ---------------------------------------------------------------

    /**
     * Experiment bucket aggregates: mean metric + counts + Wilson-friendly totals.
     */
    public DataFrame experimentSlice(String table, String bucketCol, String metricCol)
            throws Exception {
        String t = DuckDB.sanitizeIdent(table);
        String b = DuckDB.sanitizeIdent(bucketCol);
        String m = DuckDB.sanitizeIdent(metricCol);
        return db.query(
                "SELECT " + b + " AS bucket, count(*) AS n, "
                        + "avg(" + m + ") AS mean_metric, "
                        + "stddev_samp(" + m + ") AS std_metric, "
                        + "sum(" + m + ") AS sum_metric "
                        + "FROM " + t + " GROUP BY 1 ORDER BY 1");
    }

    public DataFrame experimentCtr(String table, String bucketCol, String labelCol)
            throws Exception {
        String t = DuckDB.sanitizeIdent(table);
        String b = DuckDB.sanitizeIdent(bucketCol);
        String y = DuckDB.sanitizeIdent(labelCol);
        return db.query(
                "SELECT " + b + " AS bucket, count(*) AS n, "
                        + "sum(COALESCE(" + y + ",0)) AS positives, "
                        + "sum(COALESCE(" + y + ",0))::DOUBLE / nullif(count(*),0) AS rate "
                        + "FROM " + t + " GROUP BY 1 ORDER BY 1");
    }

    // ---- calibration / score distribution ----------------------------------

    /** Score histogram + empirical positive rate per bucket (reliability diagram input). */
    public DataFrame calibrationBins(String table, String labelCol, String scoreCol, int bins)
            throws Exception {
        String t = DuckDB.sanitizeIdent(table);
        String y = DuckDB.sanitizeIdent(labelCol);
        String s = DuckDB.sanitizeIdent(scoreCol);
        int b = Math.max(2, bins);
        return db.query(
                "WITH x AS ("
                        + "  SELECT " + y + " AS label, " + s + " AS score, "
                        + "    least(" + b + " - 1, greatest(0, "
                        + "      cast(floor(" + s + " * " + b + ") AS INT))) AS bin "
                        + "  FROM " + t + " WHERE " + s + " IS NOT NULL"
                        + ") "
                        + "SELECT bin, count(*) AS n, avg(score) AS avg_score, "
                        + "  avg(label::DOUBLE) AS positive_rate "
                        + "FROM x GROUP BY 1 ORDER BY 1");
    }

    // ---- windowed user activity --------------------------------------------

    public DataFrame userActivityRolling(String eventTable, String userCol, String tsCol,
                                         int days) throws Exception {
        String t = DuckDB.sanitizeIdent(eventTable);
        String u = DuckDB.sanitizeIdent(userCol);
        String ts = DuckDB.sanitizeIdent(tsCol);
        int d = Math.max(1, days);
        return db.query(
                "SELECT " + u + " AS user_id, "
                        + "  count(*) AS events_" + d + "d, "
                        + "  count(DISTINCT date_trunc('day', " + ts + ")) AS active_days_" + d + "d, "
                        + "  min(" + ts + ") AS first_ts, max(" + ts + ") AS last_ts "
                        + "FROM " + t + " "
                        + "WHERE " + ts + " >= current_timestamp - INTERVAL '" + d + " days' "
                        + "GROUP BY 1");
    }
}
