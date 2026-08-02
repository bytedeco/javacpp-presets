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
package org.bytedeco.pytorch.utils.doris;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;

import java.util.Objects;

/**
 * Recsys sample / exposure-click join templates on Doris (public SQL patterns only).
 *
 * <p>Does not embed proprietary ranking SQL. Prefer DuckDB on exported Parquet for
 * heavy negative sampling — this class covers OLAP-side dump and point features.</p>
 */
public final class DorisRecsys {
    private DorisRecsys() {}

    /**
     * Export interaction samples for a partition day:
     * {@code SELECT user_id, item_id, label, ts FROM events WHERE dt = ?}.
     */
    public static DataFrame exportSamples(DorisOptions options, String eventsTable, String dt,
                                          String userCol, String itemCol, String labelCol, String tsCol) {
        Objects.requireNonNull(options, "options");
        Objects.requireNonNull(eventsTable, "eventsTable");
        Objects.requireNonNull(dt, "dt");
        String u = userCol == null ? "user_id" : userCol;
        String i = itemCol == null ? "item_id" : itemCol;
        String l = labelCol == null ? "label" : labelCol;
        String t = tsCol == null ? "event_timestamp" : tsCol;
        String db = options.database();
        String sql = "SELECT `" + esc(u) + "`, `" + esc(i) + "`, `" + esc(l) + "`, `" + esc(t) + "` FROM "
                + DorisCatalog.qualify(db, eventsTable)
                + " WHERE `dt` = '" + dt.replace("'", "''") + "'";
        return Doris.query(options, sql);
    }

    /**
     * Join impressions with clicks on (user_id, item_id, dt) — classic CTR sample skeleton.
     */
    public static DataFrame impressionClickJoin(DorisOptions options,
                                                String impressionTable, String clickTable, String dt) {
        Objects.requireNonNull(dt, "dt");
        String db = options.database();
        String sql = """
                SELECT
                  i.user_id,
                  i.item_id,
                  i.dt,
                  CASE WHEN c.user_id IS NULL THEN 0 ELSE 1 END AS label
                FROM %s i
                LEFT JOIN %s c
                  ON i.user_id = c.user_id AND i.item_id = c.item_id AND i.dt = c.dt
                WHERE i.dt = '%s'
                """.formatted(
                DorisCatalog.qualify(db, impressionTable),
                DorisCatalog.qualify(db, clickTable),
                dt.replace("'", "''"));
        return Doris.query(options, sql);
    }

    /** Partitioned scan of a behavior lake-style table via Doris. */
    public static DataFrame scanBehavior(DorisCatalog catalog, String table, PartitionFilter dtFilter,
                                         long limit) {
        var scan = catalog.scan(catalog.options().database(), table);
        if (dtFilter != null) scan.filter(dtFilter);
        if (limit >= 0) scan.limit(limit);
        return scan.collect();
    }

    private static String esc(String name) {
        return DorisCatalog.escapeIdent(name);
    }
}
