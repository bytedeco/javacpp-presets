/*
 * Copyright (C) 2020-2026 Eduardo Gonzalez, Hervé Guillemet, Samuel Audet
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
package org.bytedeco.pytorch.utils.orm.dataframe;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.orm.mapping.BeanToMapMapper;
import org.bytedeco.pytorch.utils.orm.mapping.MapToBeanMapper;
import org.bytedeco.pytorch.utils.orm.mapping.TypeUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Bridge between JavaBeans / maps and {@link DataFrame}.
 *
 * <pre>{@code
 * List&lt;Person&gt; people = ...;
 * DataFrame df = DataFrameMapper.fromBeans(people);
 * List&lt;Person&gt; back = DataFrameMapper.toBeans(df, Person.class);
 * }</pre>
 */
public final class DataFrameMapper {
    private DataFrameMapper() {}

    // ---- beans → DataFrame ----

    public static <T> DataFrame fromBeans(List<T> beans) {
        return fromBeans(beans, MapToBeanMapper.NamingStrategy.IDENTITY);
    }

    public static <T> DataFrame fromBeans(List<T> beans, MapToBeanMapper.NamingStrategy naming) {
        if (beans == null || beans.isEmpty()) {
            return DataFrame.create();
        }
        MapToBeanMapper.NamingStrategy strategy =
                naming == null ? MapToBeanMapper.NamingStrategy.IDENTITY : naming;

        // infer schema from first non-null bean, merge property types
        Class<?> type = null;
        for (T b : beans) {
            if (b != null) {
                type = b.getClass();
                break;
            }
        }
        if (type == null) return DataFrame.create();

        List<BeanToMapMapper.PropertyAccess> props = BeanToMapMapper.propertiesOf(type);
        DataFrame df = DataFrame.create();
        for (BeanToMapMapper.PropertyAccess p : props) {
            if (!p.readable) continue;
            String col = strategy.toColumn(p.name);
            df.addColumn(col, columnDTypeOf(p.type));
        }

        for (T bean : beans) {
            if (bean == null) {
                df.addRow();
                continue;
            }
            Map<String, Object> map = BeanToMapMapper.toMap(bean);
            int ri = df.addEmptyRow();
            for (BeanToMapMapper.PropertyAccess p : props) {
                if (!p.readable) continue;
                String col = strategy.toColumn(p.name);
                Object v = map.get(p.name);
                df.set(ri, col, v);
            }
        }
        return df;
    }

    public static <T> DataFrame fromBean(T bean) {
        if (bean == null) return DataFrame.create();
        return fromBeans(Collections.singletonList(bean));
    }

    public static <T> DataFrame fromBean(T bean, MapToBeanMapper.NamingStrategy naming) {
        if (bean == null) return DataFrame.create();
        return fromBeans(Collections.singletonList(bean), naming);
    }

    // ---- maps → DataFrame ----

    public static DataFrame fromMaps(List<? extends Map<String, ?>> rows) {
        if (rows == null || rows.isEmpty()) return DataFrame.create();

        // union of keys in encounter order
        Map<String, Column.DType> schema = new LinkedHashMap<>();
        for (Map<String, ?> row : rows) {
            if (row == null) continue;
            for (Map.Entry<String, ?> e : row.entrySet()) {
                if (e.getKey() == null) continue;
                Column.DType inferred = inferDType(e.getValue());
                Column.DType existing = schema.get(e.getKey());
                if (existing == null) {
                    schema.put(e.getKey(), inferred);
                } else if (existing != inferred && inferred != Column.DType.STRING) {
                    // widen to STRING on conflict, or keep numeric preference
                    schema.put(e.getKey(), widen(existing, inferred));
                }
            }
        }

        DataFrame df = DataFrame.create();
        for (Map.Entry<String, Column.DType> e : schema.entrySet()) {
            df.addColumn(e.getKey(), e.getValue());
        }
        for (Map<String, ?> row : rows) {
            int ri = df.addEmptyRow();
            if (row == null) continue;
            for (String col : schema.keySet()) {
                if (row.containsKey(col)) {
                    df.set(ri, col, row.get(col));
                }
            }
        }
        return df;
    }

    public static DataFrame fromMap(Map<String, ?> row) {
        if (row == null) return DataFrame.create();
        return fromMaps(Collections.singletonList(row));
    }

    // ---- DataFrame → beans ----

    public static <T> List<T> toBeans(DataFrame df, Class<T> type) {
        return toBeans(df, type, MapToBeanMapper.NamingStrategy.IDENTITY);
    }

    public static <T> List<T> toBeans(DataFrame df, Class<T> type,
                                      MapToBeanMapper.NamingStrategy naming) {
        Objects.requireNonNull(type, "type");
        if (df == null || df.rowCount() == 0) return new ArrayList<>();
        MapToBeanMapper.NamingStrategy strategy =
                naming == null ? MapToBeanMapper.NamingStrategy.IDENTITY : naming;

        List<T> out = new ArrayList<>(df.rowCount());
        for (int i = 0; i < df.rowCount(); i++) {
            out.add(toBean(df, i, type, strategy));
        }
        return out;
    }

    public static <T> T toBean(DataFrame df, Class<T> type) {
        if (df == null || df.rowCount() == 0) return null;
        return toBean(df, 0, type, MapToBeanMapper.NamingStrategy.IDENTITY);
    }

    public static <T> T toBean(DataFrame df, int rowIndex, Class<T> type) {
        return toBean(df, rowIndex, type, MapToBeanMapper.NamingStrategy.IDENTITY);
    }

    public static <T> T toBean(DataFrame df, int rowIndex, Class<T> type,
                               MapToBeanMapper.NamingStrategy naming) {
        Objects.requireNonNull(df, "df");
        Objects.requireNonNull(type, "type");
        Map<String, Object> map = toMap(df, rowIndex);
        return MapToBeanMapper.toBean(map, type, naming);
    }

    // ---- DataFrame → maps ----

    public static List<Map<String, Object>> toMaps(DataFrame df) {
        if (df == null || df.rowCount() == 0) return new ArrayList<>();
        List<Map<String, Object>> out = new ArrayList<>(df.rowCount());
        for (int i = 0; i < df.rowCount(); i++) {
            out.add(toMap(df, i));
        }
        return out;
    }

    public static Map<String, Object> toMap(DataFrame df, int rowIndex) {
        Objects.requireNonNull(df, "df");
        Map<String, Object> row = new LinkedHashMap<>();
        for (Column c : df.columns()) {
            row.put(c.name(), df.get(rowIndex, c.name()));
        }
        return row;
    }

    // ---- helpers ----

    private static Column.DType inferDType(Object value) {
        if (value == null) return Column.DType.STRING;
        return columnDTypeOf(value.getClass());
    }

    /** Infer {@link Column.DType} from a Java property / value type. */
    public static Column.DType columnDTypeOf(Class<?> javaType) {
        if (javaType == null) return Column.DType.STRING;
        Class<?> t = TypeUtils.wrap(javaType);
        if (t == Boolean.class) return Column.DType.BOOLEAN;
        if (t == Byte.class || t == Short.class || t == Integer.class) return Column.DType.INT32;
        if (t == Long.class || t == java.math.BigInteger.class) return Column.DType.INT64;
        if (t == Float.class) return Column.DType.FLOAT32;
        if (t == Double.class || t == java.math.BigDecimal.class) return Column.DType.FLOAT64;
        if (t == java.time.LocalDate.class || t == java.sql.Date.class) return Column.DType.DATE;
        if (t == java.time.LocalDateTime.class || t == java.sql.Timestamp.class
                || t == java.time.Instant.class || t == java.util.Date.class
                || t == java.time.ZonedDateTime.class || t == java.time.OffsetDateTime.class) {
            return Column.DType.DATETIME;
        }
        if (t == java.time.LocalTime.class || t == java.sql.Time.class) return Column.DType.TIME;
        if (t == byte[].class) return Column.DType.BINARY;
        return Column.DType.STRING;
    }

    private static Column.DType widen(Column.DType a, Column.DType b) {
        if (a == b) return a;
        // numeric widen
        if (isInt(a) && isInt(b)) return Column.DType.INT64;
        if (isNumeric(a) && isNumeric(b)) return Column.DType.FLOAT64;
        return Column.DType.STRING;
    }

    private static boolean isInt(Column.DType t) {
        return t == Column.DType.INT32 || t == Column.DType.INT64;
    }

    private static boolean isNumeric(Column.DType t) {
        return t == Column.DType.INT32 || t == Column.DType.INT64
                || t == Column.DType.FLOAT32 || t == Column.DType.FLOAT64;
    }
}
