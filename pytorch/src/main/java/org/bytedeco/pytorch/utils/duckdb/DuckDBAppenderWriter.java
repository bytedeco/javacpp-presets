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

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.duckdb.DuckDBAppender;
import org.duckdb.DuckDBConnection;

import java.math.BigDecimal;
import java.sql.SQLException;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.util.Collection;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * High-throughput bulk writer built on official {@link DuckDBAppender}.
 *
 * <p>Unlike JDBC {@code PreparedStatement} batch inserts, Appender writes
 * columnar chunks directly into DuckDB storage — the path used by DuckDB's
 * own Python / C++ bindings for multi-million-row loads (Meta feature-log
 * ingest, ByteDance offline sample dump, Google TFX bulk materialize).
 *
 * <pre>{@code
 * try (DuckDB db = DuckDB.open(path, DuckDBConfig.etlBulkLoad())) {
 *     db.execute("CREATE TABLE events(user_id BIGINT, item_id BIGINT, ts TIMESTAMP, label INTEGER)");
 *     long n = DuckDBAppenderWriter.appendDataFrame(db, "events", df);
 * }
 * }</pre>
 */
public final class DuckDBAppenderWriter implements AutoCloseable {

    private final DuckDBAppender appender;
    private final int columnCount;
    private long rowsAppended;
    private final int flushEvery;
    private long sinceFlush;

    private DuckDBAppenderWriter(DuckDBAppender appender, int columnCount, int flushEvery) {
        this.appender = Objects.requireNonNull(appender, "appender");
        this.columnCount = columnCount;
        this.flushEvery = Math.max(0, flushEvery);
    }

    /**
     * Open an appender for {@code schema.table} (schema may be {@code null} → main).
     *
     * @param columnCount expected physical columns (for validation on {@link #row})
     * @param flushEvery  flush after this many rows (0 = only on {@link #flush}/{@link #close})
     */
    public static DuckDBAppenderWriter open(DuckDBConnection conn, String schema, String table,
                                            int columnCount, int flushEvery) throws SQLException {
        Objects.requireNonNull(conn, "conn");
        Objects.requireNonNull(table, "table");
        DuckDBAppender ap = schema == null || schema.isBlank()
                ? conn.createAppender(table)
                : conn.createAppender(schema, table);
        return new DuckDBAppenderWriter(ap, columnCount, flushEvery);
    }

    public static DuckDBAppenderWriter open(DuckDBConnection conn, String table, int columnCount)
            throws SQLException {
        return open(conn, null, table, columnCount, 50_000);
    }

    /** Append an entire {@link DataFrame} via Appender (fast path). */
    public static long appendDataFrame(DuckDB db, String table, DataFrame df) throws Exception {
        return appendDataFrame(db, null, table, df, 50_000);
    }

    public static long appendDataFrame(DuckDB db, String schema, String table, DataFrame df,
                                       int flushEvery) throws Exception {
        Objects.requireNonNull(db, "db");
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        DuckDBConnection conn = db.nativeConnection();
        List<Column> cols = df.columns();
        int nCols = cols.size();
        int nRows = df.rowCount();
        if (nRows == 0 || nCols == 0) return 0L;

        try (DuckDBAppenderWriter w = open(conn, schema, table, nCols, flushEvery)) {
            // cache column arrays for locality
            Column[] colArr = cols.toArray(new Column[0]);
            Column.DType[] dtypes = new Column.DType[nCols];
            for (int c = 0; c < nCols; c++) dtypes[c] = colArr[c].dtype();

            for (int r = 0; r < nRows; r++) {
                w.appender.beginRow();
                for (int c = 0; c < nCols; c++) {
                    appendCell(w.appender, colArr[c].get(r), dtypes[c]);
                }
                w.appender.endRow();
                w.rowsAppended++;
                w.sinceFlush++;
                if (w.flushEvery > 0 && w.sinceFlush >= w.flushEvery) {
                    w.appender.flush();
                    w.sinceFlush = 0;
                }
            }
            w.appender.flush();
            w.sinceFlush = 0;
            return w.rowsAppended;
        }
    }

    /** Append one logical row; values length must equal columnCount. */
    public DuckDBAppenderWriter row(Object... values) throws SQLException {
        if (values == null || values.length != columnCount) {
            throw new IllegalArgumentException(
                    "expected " + columnCount + " values, got "
                            + (values == null ? 0 : values.length));
        }
        appender.beginRow();
        for (Object v : values) {
            appendObject(appender, v);
        }
        appender.endRow();
        rowsAppended++;
        sinceFlush++;
        if (flushEvery > 0 && sinceFlush >= flushEvery) {
            appender.flush();
            sinceFlush = 0;
        }
        return this;
    }

    public long flush() throws SQLException {
        long f = appender.flush();
        sinceFlush = 0;
        return f;
    }

    public long rowsAppended() {
        return rowsAppended;
    }

    public DuckDBAppender raw() {
        return appender;
    }

    @Override
    public void close() throws SQLException {
        try {
            if (!appender.isClosed()) {
                appender.flush();
            }
        } finally {
            appender.close();
        }
    }

    // ---- cell encoding -----------------------------------------------------

    static void appendCell(DuckDBAppender ap, Object v, Column.DType dtype) throws SQLException {
        if (v == null) {
            ap.appendNull();
            return;
        }
        switch (dtype) {
            case INT32:
                ap.append(asInt(v));
                break;
            case INT64:
                ap.append(asLong(v));
                break;
            case FLOAT32:
                ap.append(asFloat(v));
                break;
            case FLOAT64:
                ap.append(asDouble(v));
                break;
            case BOOLEAN:
                ap.append(asBoolean(v));
                break;
            case DATE:
                if (v instanceof LocalDate) ap.append((LocalDate) v);
                else if (v instanceof java.sql.Date) ap.append(((java.sql.Date) v).toLocalDate());
                else ap.append(LocalDate.parse(v.toString()));
                break;
            case DATETIME:
                if (v instanceof LocalDateTime) ap.append((LocalDateTime) v);
                else if (v instanceof OffsetDateTime) ap.append((OffsetDateTime) v);
                else if (v instanceof Date) ap.append((Date) v);
                else if (v instanceof java.sql.Timestamp)
                    ap.append(((java.sql.Timestamp) v).toLocalDateTime());
                else ap.append(LocalDateTime.parse(v.toString().replace(' ', 'T')));
                break;
            case TIME:
                if (v instanceof LocalTime) ap.append((LocalTime) v);
                else if (v instanceof java.sql.Time) ap.append(((java.sql.Time) v).toLocalTime());
                else ap.append(LocalTime.parse(v.toString()));
                break;
            case BINARY:
            case IMAGE:
            case AUDIO:
            case VIDEO:
                ap.append(asBytes(v));
                break;
            case VECTOR:
            case EMBEDDING:
                // store as FLOAT[] list — DuckDB LIST/ARRAY
                if (v instanceof float[]) {
                    ap.append((float[]) v);
                } else if (v instanceof double[]) {
                    ap.append((double[]) v);
                } else if (v instanceof Collection) {
                    ap.append((Collection<?>) v);
                } else {
                    ap.append(v.toString());
                }
                break;
            case LIST:
                if (v instanceof Collection) ap.append((Collection<?>) v);
                else if (v instanceof int[]) ap.append((int[]) v);
                else if (v instanceof long[]) ap.append((long[]) v);
                else if (v instanceof float[]) ap.append((float[]) v);
                else if (v instanceof double[]) ap.append((double[]) v);
                else ap.append(v.toString());
                break;
            case MAP:
                if (v instanceof Map) ap.append((Map<?, ?>) v);
                else ap.append(v.toString());
                break;
            case JSON:
                ap.append(v.toString());
                break;
            case STRING:
            default:
                ap.append(v.toString());
                break;
        }
    }

    static void appendObject(DuckDBAppender ap, Object v) throws SQLException {
        if (v == null) {
            ap.appendNull();
            return;
        }
        if (v instanceof Boolean) ap.append((Boolean) v);
        else if (v instanceof Byte) ap.append((Byte) v);
        else if (v instanceof Short) ap.append((Short) v);
        else if (v instanceof Integer) ap.append((Integer) v);
        else if (v instanceof Long) ap.append((Long) v);
        else if (v instanceof Float) ap.append((Float) v);
        else if (v instanceof Double) ap.append((Double) v);
        else if (v instanceof BigDecimal) ap.append((BigDecimal) v);
        else if (v instanceof String) ap.append((String) v);
        else if (v instanceof byte[]) ap.append((byte[]) v);
        else if (v instanceof UUID) ap.append((UUID) v);
        else if (v instanceof LocalDate) ap.append((LocalDate) v);
        else if (v instanceof LocalTime) ap.append((LocalTime) v);
        else if (v instanceof LocalDateTime) ap.append((LocalDateTime) v);
        else if (v instanceof OffsetDateTime) ap.append((OffsetDateTime) v);
        else if (v instanceof Date) ap.append((Date) v);
        else if (v instanceof float[]) ap.append((float[]) v);
        else if (v instanceof double[]) ap.append((double[]) v);
        else if (v instanceof int[]) ap.append((int[]) v);
        else if (v instanceof long[]) ap.append((long[]) v);
        else if (v instanceof Collection) ap.append((Collection<?>) v);
        else if (v instanceof Map) ap.append((Map<?, ?>) v);
        else ap.append(v.toString());
    }

    private static int asInt(Object v) {
        if (v instanceof Number) return ((Number) v).intValue();
        if (v instanceof Boolean) return (Boolean) v ? 1 : 0;
        return Integer.parseInt(v.toString());
    }

    private static long asLong(Object v) {
        if (v instanceof Number) return ((Number) v).longValue();
        if (v instanceof Boolean) return (Boolean) v ? 1L : 0L;
        return Long.parseLong(v.toString());
    }

    private static float asFloat(Object v) {
        if (v instanceof Number) return ((Number) v).floatValue();
        return Float.parseFloat(v.toString());
    }

    private static double asDouble(Object v) {
        if (v instanceof Number) return ((Number) v).doubleValue();
        return Double.parseDouble(v.toString());
    }

    private static boolean asBoolean(Object v) {
        if (v instanceof Boolean) return (Boolean) v;
        if (v instanceof Number) return ((Number) v).intValue() != 0;
        String s = v.toString().trim().toLowerCase();
        return "1".equals(s) || "true".equals(s) || "t".equals(s) || "yes".equals(s);
    }

    private static byte[] asBytes(Object v) {
        if (v instanceof byte[]) return (byte[]) v;
        if (v instanceof String) return ((String) v).getBytes(java.nio.charset.StandardCharsets.UTF_8);
        // ImageData / AudioData / VideoData often expose bytes() or similar — fall back to string
        return v.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
    }
}
