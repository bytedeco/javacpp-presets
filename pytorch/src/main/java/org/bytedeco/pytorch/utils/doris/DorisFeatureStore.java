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
import org.bytedeco.pytorch.utils.lake.LakeFeatureBridge;
import org.bytedeco.pytorch.utils.lake.LakeSchema;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Feature-store helpers on Doris wide / detail tables (Unique Key point lookup + Stream Load upsert).
 *
 * <p>Aligned with public Feature Store offline/online patterns (Feast / Meta style) using
 * Doris as the serving OLAP + primary-key store — not a private middleware clone.</p>
 */
public final class DorisFeatureStore implements AutoCloseable {

    private final DorisCatalog catalog;
    private final boolean ownCatalog;
    private final String database;
    private final String table;
    private final String[] entityKeys;
    private final String timestampColumn;

    public DorisFeatureStore(DorisCatalog catalog, boolean ownCatalog,
                             String database, String table,
                             String[] entityKeys, String timestampColumn) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.ownCatalog = ownCatalog;
        this.database = Objects.requireNonNull(database, "database");
        this.table = Objects.requireNonNull(table, "table");
        this.entityKeys = entityKeys == null ? new String[0] : entityKeys.clone();
        this.timestampColumn = timestampColumn;
    }

    public static DorisFeatureStore open(DorisOptions options, String... entityKeys) {
        Objects.requireNonNull(options.database(), "database");
        Objects.requireNonNull(options.table(), "table");
        DorisCatalog cat = Doris.open(options);
        return new DorisFeatureStore(cat, true, options.database(), options.table(),
                entityKeys, options.properties().getOrDefault("timestamp_column", "event_timestamp"));
    }

    public DorisCatalog catalog() {
        return catalog;
    }

    /** Point-get features for one entity key map. */
    public DataFrame get(Map<String, Object> keys, String... featureColumns) {
        return catalog.pointQuery(database, table, keys, featureColumns);
    }

    /** Batch get by repeating OR predicates (small fan-out; prefer temp table for huge key sets). */
    public DataFrame multiGet(List<Map<String, Object>> keyRows, String... featureColumns) {
        Objects.requireNonNull(keyRows, "keyRows");
        if (keyRows.isEmpty()) return DataFrame.create();
        if (entityKeys.length == 0) {
            throw new IllegalStateException("entityKeys required for multiGet");
        }
        String colList = (featureColumns == null || featureColumns.length == 0)
                ? "*"
                : String.join(", ", DorisCatalog.quoteAll(featureColumns));
        StringBuilder sb = new StringBuilder("SELECT ").append(colList)
                .append(" FROM ").append(DorisCatalog.qualify(database, table))
                .append(" WHERE ");
        for (int i = 0; i < keyRows.size(); i++) {
            if (i > 0) sb.append(" OR ");
            sb.append('(');
            Map<String, Object> row = keyRows.get(i);
            for (int k = 0; k < entityKeys.length; k++) {
                if (k > 0) sb.append(" AND ");
                String key = entityKeys[k];
                Object v = row.get(key);
                sb.append('`').append(DorisCatalog.escapeIdent(key)).append("` = ");
                sb.append(sqlLiteral(v));
            }
            sb.append(')');
        }
        return catalog.query(sb.toString());
    }

    /** Upsert feature rows via Stream Load (Unique Key table recommended). */
    public DorisStreamLoad.Result put(DataFrame features) {
        DorisOptions opts = catalog.options().toBuilder()
                .database(database)
                .table(table)
                .tableModel(DorisOptions.TableModel.UNIQUE)
                .keys(entityKeys.length == 0 ? null : entityKeys)
                .build();
        return Doris.streamLoad(opts, features);
    }

    public DorisStreamLoad.Result putRows(List<Map<String, Object>> rows, LakeSchema schema) {
        DataFrame df = LakeFeatureBridge.fromOfflineRows(rows, schema);
        return put(df);
    }

    /** Scan feature table with optional partition filter (offline materialize source). */
    public DataFrame readRange(PartitionFilter partitionFilter, String extraWhere, long limit) {
        var scan = catalog.scan(database, table);
        if (partitionFilter != null) scan.filter(partitionFilter);
        if (extraWhere != null && !extraWhere.isBlank()) scan.where(extraWhere);
        if (limit >= 0) scan.limit(limit);
        return scan.collect();
    }

    public List<Map<String, Object>> toOfflineRows(DataFrame df) {
        return LakeFeatureBridge.toOfflineRows(df, entityKeys, null, timestampColumn);
    }

    private static String sqlLiteral(Object v) {
        if (v == null) return "NULL";
        if (v instanceof Number || v instanceof Boolean) return String.valueOf(v);
        return "'" + String.valueOf(v).replace("'", "''") + "'";
    }

    @Override
    public void close() {
        if (ownCatalog) catalog.close();
    }
}
