package org.bytedeco.pytorch.dataframe.sql;

import org.bytedeco.pytorch.dataframe.Column;

import java.math.BigDecimal;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Types;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * Maps JDBC SQL types ↔ {@link Column.DType} and extracts cell values.
 */
public final class JdbcTypeMap {
    private JdbcTypeMap() {}

    public static Column.DType fromJdbc(int sqlType, String typeName, int scale) {
        switch (sqlType) {
            case Types.BOOLEAN:
            case Types.BIT:
                return Column.DType.BOOLEAN;
            case Types.TINYINT:
            case Types.SMALLINT:
            case Types.INTEGER:
                return Column.DType.INT32;
            case Types.BIGINT:
                return Column.DType.INT64;
            case Types.REAL:
            case Types.FLOAT:
                return Column.DType.FLOAT32;
            case Types.DOUBLE:
            case Types.NUMERIC:
            case Types.DECIMAL:
                return Column.DType.FLOAT64;
            case Types.DATE:
                return Column.DType.DATE;
            case Types.TIMESTAMP:
            case Types.TIMESTAMP_WITH_TIMEZONE:
                return Column.DType.DATETIME;
            case Types.TIME:
            case Types.TIME_WITH_TIMEZONE:
                return Column.DType.TIME;
            case Types.BINARY:
            case Types.VARBINARY:
            case Types.LONGVARBINARY:
            case Types.BLOB:
                return Column.DType.BINARY;
            case Types.CLOB:
            case Types.NCLOB:
            case Types.VARCHAR:
            case Types.NVARCHAR:
            case Types.CHAR:
            case Types.NCHAR:
            case Types.LONGVARCHAR:
            case Types.LONGNVARCHAR:
            default:
                // SQLite often reports NUMERIC affinity loosely
                if (typeName != null) {
                    String t = typeName.toUpperCase(Locale.ROOT);
                    if (t.contains("INT")) return Column.DType.INT64;
                    if (t.contains("REAL") || t.contains("FLOA") || t.contains("DOUB")) return Column.DType.FLOAT64;
                    if (t.contains("BOOL")) return Column.DType.BOOLEAN;
                    if (t.contains("DATE") && !t.contains("TIME")) return Column.DType.DATE;
                    if (t.contains("TIME")) return Column.DType.DATETIME;
                    if (t.contains("BLOB")) return Column.DType.BINARY;
                }
                return Column.DType.STRING;
        }
    }

    public static String toSqlType(Column.DType dtype, boolean sqlite) {
        switch (dtype) {
            case INT32: return "INTEGER";
            case INT64: return sqlite ? "INTEGER" : "BIGINT";
            case FLOAT32: return sqlite ? "REAL" : "REAL";
            case FLOAT64: return sqlite ? "REAL" : "DOUBLE";
            case BOOLEAN: return sqlite ? "INTEGER" : "BOOLEAN";
            case DATE: return sqlite ? "TEXT" : "DATE";
            case DATETIME: return sqlite ? "TEXT" : "TIMESTAMP";
            case TIME: return sqlite ? "TEXT" : "TIME";
            case BINARY: return sqlite ? "BLOB" : "BYTEA";
            default: return sqlite ? "TEXT" : "VARCHAR(4096)";
        }
    }

    public static Object getValue(ResultSet rs, int oneBasedCol, Column.DType dtype) throws SQLException {
        switch (dtype) {
            case INT32: {
                int x = rs.getInt(oneBasedCol);
                return rs.wasNull() ? null : Integer.valueOf(x);
            }
            case INT64: {
                long x = rs.getLong(oneBasedCol);
                return rs.wasNull() ? null : Long.valueOf(x);
            }
            case FLOAT32: {
                float x = rs.getFloat(oneBasedCol);
                return rs.wasNull() ? null : Float.valueOf(x);
            }
            case FLOAT64: {
                double x = rs.getDouble(oneBasedCol);
                return rs.wasNull() ? null : Double.valueOf(x);
            }
            case BOOLEAN: {
                boolean x = rs.getBoolean(oneBasedCol);
                return rs.wasNull() ? null : Boolean.valueOf(x);
            }
            case DATE: {
                java.sql.Date d = rs.getDate(oneBasedCol);
                return d == null ? null : d.toLocalDate();
            }
            case DATETIME: {
                java.sql.Timestamp ts = rs.getTimestamp(oneBasedCol);
                return ts == null ? null : ts.toLocalDateTime();
            }
            case TIME: {
                java.sql.Time t = rs.getTime(oneBasedCol);
                return t == null ? null : t.toLocalTime();
            }
            case BINARY:
                return rs.getBytes(oneBasedCol);
            default: {
                Object v = rs.getObject(oneBasedCol);
                if (v == null) return null;
                if (v instanceof BigDecimal) return ((BigDecimal) v).doubleValue();
                if (v instanceof LocalDate || v instanceof LocalDateTime || v instanceof LocalTime) return v;
                if (v instanceof java.sql.Date) return ((java.sql.Date) v).toLocalDate();
                if (v instanceof java.sql.Timestamp) return ((java.sql.Timestamp) v).toLocalDateTime();
                if (v instanceof java.sql.Time) return ((java.sql.Time) v).toLocalTime();
                if (v instanceof byte[]) return v;
                if (v instanceof Number || v instanceof Boolean) return v;
                return String.valueOf(v);
            }
        }
    }

    public static Column.DType[] dtypesFromMeta(ResultSetMetaData meta) throws SQLException {
        int n = meta.getColumnCount();
        Column.DType[] out = new Column.DType[n];
        for (int i = 1; i <= n; i++) {
            out[i - 1] = fromJdbc(meta.getColumnType(i), meta.getColumnTypeName(i), meta.getScale(i));
        }
        return out;
    }

    public static String[] namesFromMeta(ResultSetMetaData meta) throws SQLException {
        int n = meta.getColumnCount();
        String[] names = new String[n];
        Set<String> seen = new HashSet<>();
        for (int i = 1; i <= n; i++) {
            String label = meta.getColumnLabel(i);
            if (label == null || label.isEmpty()) label = meta.getColumnName(i);
            if (label == null || label.isEmpty()) label = "col_" + (i - 1);
            String base = label;
            int k = 1;
            while (!seen.add(label)) label = base + "_" + (k++);
            names[i - 1] = label;
        }
        return names;
    }
}
