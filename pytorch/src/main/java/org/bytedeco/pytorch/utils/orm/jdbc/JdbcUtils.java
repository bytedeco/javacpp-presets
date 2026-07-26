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
package org.bytedeco.pytorch.utils.orm.jdbc;

import java.math.BigDecimal;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Timestamp;
import java.sql.Types;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Low-level JDBC helpers for column metadata and row extraction.
 */
public final class JdbcUtils {
    private JdbcUtils() {}

    public static String[] columnLabels(ResultSetMetaData meta) throws SQLException {
        int n = meta.getColumnCount();
        String[] names = new String[n];
        Set<String> seen = new HashSet<>();
        for (int i = 1; i <= n; i++) {
            String label = meta.getColumnLabel(i);
            if (label == null || label.isEmpty()) label = meta.getColumnName(i);
            if (label == null || label.isEmpty()) label = "col_" + (i - 1);
            String base = label;
            int k = 1;
            while (!seen.add(label)) {
                label = base + "_" + (k++);
            }
            names[i - 1] = label;
        }
        return names;
    }

    public static int columnCount(ResultSet rs) throws SQLException {
        return rs.getMetaData().getColumnCount();
    }

    /** Read the current row as Object[] (1-based JDBC → 0-based array). */
    public static Object[] readRow(ResultSet rs) throws SQLException {
        int n = columnCount(rs);
        Object[] row = new Object[n];
        for (int i = 1; i <= n; i++) {
            row[i - 1] = getObject(rs, i);
        }
        return row;
    }

    /**
     * Extract a cell value with reasonable Java types (unwrap SQL date/time,
     * prefer Integer/Long/Double over vendor wrappers where possible).
     */
    public static Object getObject(ResultSet rs, int oneBasedCol) throws SQLException {
        Object v = rs.getObject(oneBasedCol);
        if (v == null || rs.wasNull()) return null;
        if (v instanceof Timestamp) {
            return ((Timestamp) v).toLocalDateTime();
        }
        if (v instanceof java.sql.Date) {
            return ((java.sql.Date) v).toLocalDate();
        }
        if (v instanceof java.sql.Time) {
            return ((java.sql.Time) v).toLocalTime();
        }
        if (v instanceof BigDecimal) {
            BigDecimal bd = (BigDecimal) v;
            if (bd.scale() <= 0) {
                try {
                    return bd.longValueExact();
                } catch (ArithmeticException ignored) {
                    return bd;
                }
            }
            return bd.doubleValue();
        }
        return v;
    }

    public static void bind(PreparedStatement ps, int index, Object value) throws SQLException {
        if (value == null) {
            ps.setObject(index, null);
            return;
        }
        if (value instanceof Integer) {
            ps.setInt(index, (Integer) value);
        } else if (value instanceof Long) {
            ps.setLong(index, (Long) value);
        } else if (value instanceof Short) {
            ps.setShort(index, (Short) value);
        } else if (value instanceof Byte) {
            ps.setByte(index, (Byte) value);
        } else if (value instanceof Double) {
            ps.setDouble(index, (Double) value);
        } else if (value instanceof Float) {
            ps.setFloat(index, (Float) value);
        } else if (value instanceof Boolean) {
            ps.setBoolean(index, (Boolean) value);
        } else if (value instanceof String) {
            ps.setString(index, (String) value);
        } else if (value instanceof byte[]) {
            ps.setBytes(index, (byte[]) value);
        } else if (value instanceof BigDecimal) {
            ps.setBigDecimal(index, (BigDecimal) value);
        } else if (value instanceof Timestamp) {
            ps.setTimestamp(index, (Timestamp) value);
        } else if (value instanceof java.sql.Date) {
            ps.setDate(index, (java.sql.Date) value);
        } else if (value instanceof java.sql.Time) {
            ps.setTime(index, (java.sql.Time) value);
        } else if (value instanceof LocalDateTime) {
            ps.setString(index, value.toString());
        } else if (value instanceof LocalDate) {
            ps.setString(index, value.toString());
        } else if (value instanceof LocalTime) {
            ps.setString(index, value.toString());
        } else if (value instanceof Instant) {
            ps.setString(index, value.toString());
        } else if (value instanceof Date) {
            ps.setTimestamp(index, new Timestamp(((Date) value).getTime()));
        } else if (value instanceof Enum) {
            ps.setString(index, ((Enum<?>) value).name());
        } else {
            ps.setObject(index, value);
        }
    }

    public static void bindAll(PreparedStatement ps, Object... params) throws SQLException {
        if (params == null) return;
        for (int i = 0; i < params.length; i++) {
            bind(ps, i + 1, params[i]);
        }
    }

    public static List<String> columnLabelList(ResultSetMetaData meta) throws SQLException {
        String[] names = columnLabels(meta);
        List<String> list = new ArrayList<>(names.length);
        for (String n : names) list.add(n);
        return list;
    }

    public static int sqlTypeOf(Object value) {
        if (value == null) return Types.NULL;
        if (value instanceof Integer) return Types.INTEGER;
        if (value instanceof Long) return Types.BIGINT;
        if (value instanceof Short) return Types.SMALLINT;
        if (value instanceof Byte) return Types.TINYINT;
        if (value instanceof Double) return Types.DOUBLE;
        if (value instanceof Float) return Types.REAL;
        if (value instanceof Boolean) return Types.BOOLEAN;
        if (value instanceof byte[]) return Types.BLOB;
        if (value instanceof BigDecimal) return Types.DECIMAL;
        if (value instanceof Timestamp || value instanceof LocalDateTime || value instanceof Instant) {
            return Types.TIMESTAMP;
        }
        if (value instanceof java.sql.Date || value instanceof LocalDate) return Types.DATE;
        if (value instanceof java.sql.Time || value instanceof LocalTime) return Types.TIME;
        return Types.VARCHAR;
    }

    /** Quote an identifier for SQLite / generic SQL (double quotes). */
    public static String quoteIdent(String name) {
        if (name == null) return "\"\"";
        if ((name.startsWith("\"") && name.endsWith("\""))
                || (name.startsWith("`") && name.endsWith("`"))) {
            return name;
        }
        return "\"" + name.replace("\"", "\"\"") + "\"";
    }
}
