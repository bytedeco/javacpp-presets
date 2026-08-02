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
package org.bytedeco.pytorch.utils.lake;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Bridge between lake {@link DataFrame} batches and FeaturePlatform offline row maps.
 *
 * <p>Entity keys + feature columns are projected into {@code Map&lt;String,Object&gt;}
 * rows suitable for {@code OfflineStore#put} / PIT join.</p>
 *
 * @see LakeTensorBridge
 * @see org.bytedeco.pytorch.feature.offline.OfflineStore
 */
public final class LakeFeatureBridge {
    private LakeFeatureBridge() {}

    /**
     * Convert a lake batch into offline feature rows (all columns).
     */
    public static List<Map<String, Object>> toOfflineRows(DataFrame df) {
        Objects.requireNonNull(df, "dataframe");
        return df.toRecords();
    }

    /**
     * Project entity keys + feature columns (+ optional timestamp) into offline rows.
     *
     * @param entityKeys entity join keys (e.g. user_id, item_id)
     * @param featureColumns feature value columns; empty → all non-entity/non-ts columns
     * @param timestampColumn optional event timestamp column name (may be null)
     */
    public static List<Map<String, Object>> toOfflineRows(DataFrame df,
                                                          String[] entityKeys,
                                                          String[] featureColumns,
                                                          String timestampColumn) {
        Objects.requireNonNull(df, "dataframe");
        List<String> keep = new ArrayList<>();
        if (entityKeys != null) {
            for (String k : entityKeys) {
                if (k != null && hasColumn(df, k)) keep.add(k);
            }
        }
        if (timestampColumn != null && hasColumn(df, timestampColumn) && !keep.contains(timestampColumn)) {
            keep.add(timestampColumn);
        }
        if (featureColumns == null || featureColumns.length == 0) {
            for (Column c : df.columns()) {
                if (!keep.contains(c.name())) keep.add(c.name());
            }
        } else {
            for (String f : featureColumns) {
                if (f != null && hasColumn(df, f) && !keep.contains(f)) keep.add(f);
            }
        }
        if (keep.isEmpty()) return List.of();
        DataFrame projected = df.select(keep.toArray(new String[0]));
        return projected.toRecords();
    }

    /**
     * Rebuild a typed DataFrame from offline store rows.
     */
    public static DataFrame fromOfflineRows(List<Map<String, Object>> rows, LakeSchema schema) {
        return LakeTensorBridge.fromFeatureRows(rows, schema);
    }

    /**
     * Extract entity key maps only (for online lookup / join keys).
     */
    public static List<Map<String, Object>> entityKeys(DataFrame df, String... keys) {
        Objects.requireNonNull(df, "dataframe");
        if (keys == null || keys.length == 0) return List.of();
        List<String> keep = new ArrayList<>();
        for (String k : keys) {
            if (k != null && hasColumn(df, k)) keep.add(k);
        }
        if (keep.isEmpty()) return List.of();
        return df.select(keep.toArray(new String[0])).toRecords();
    }

    /**
     * Normalize timestamp column to epoch millis Long when possible
     * (LocalDateTime / Instant / Number / ISO string left as-is if unparseable).
     */
    public static List<Map<String, Object>> withEpochMillis(List<Map<String, Object>> rows,
                                                            String timestampColumn) {
        Objects.requireNonNull(rows, "rows");
        if (timestampColumn == null || timestampColumn.isBlank()) return rows;
        List<Map<String, Object>> out = new ArrayList<>(rows.size());
        for (Map<String, Object> row : rows) {
            Map<String, Object> copy = new LinkedHashMap<>(row);
            Object v = copy.get(timestampColumn);
            Long ms = toEpochMillis(v);
            if (ms != null) copy.put(timestampColumn, ms);
            out.add(copy);
        }
        return out;
    }

    private static boolean hasColumn(DataFrame df, String name) {
        try {
            df.column(name);
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    private static Long toEpochMillis(Object v) {
        if (v == null) return null;
        if (v instanceof Long l) return l;
        if (v instanceof Integer i) return i.longValue();
        if (v instanceof java.time.Instant inst) return inst.toEpochMilli();
        if (v instanceof java.time.LocalDateTime ldt) {
            return ldt.atZone(java.time.ZoneOffset.UTC).toInstant().toEpochMilli();
        }
        if (v instanceof java.time.LocalDate ld) {
            return ld.atStartOfDay(java.time.ZoneOffset.UTC).toInstant().toEpochMilli();
        }
        if (v instanceof java.sql.Timestamp ts) return ts.getTime();
        if (v instanceof java.util.Date d) return d.getTime();
        if (v instanceof Number n) return n.longValue();
        return null;
    }
}
