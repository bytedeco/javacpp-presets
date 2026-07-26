package org.bytedeco.pytorch.data.dataframe.sql;

import org.bytedeco.pytorch.data.dataframe.Column;
import org.bytedeco.pytorch.data.dataframe.DataFrame;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.Timestamp;
import java.sql.Types;
import java.time.Instant;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Write a {@link DataFrame} into a JDBC table (CREATE + batch INSERT).
 */
public final class SqlWriter {
    private SqlWriter() {}

    public static void write(DataFrame df, Connection c, String table) throws Exception {
        write(df, c, table, SqlOptions.defaults());
    }

    public static void write(DataFrame df, Connection c, String table, SqlOptions options) throws Exception {
        if (df == null) throw new IllegalArgumentException("dataframe required");
        if (c == null) throw new IllegalArgumentException("connection required");
        if (table == null || table.isBlank()) throw new IllegalArgumentException("table required");
        SqlOptions opt = options == null ? SqlOptions.defaults() : options;
        boolean sqlite = Sqlite.isSqlite(c);
        String qualified = SqlReader.qualify(table, opt);

        boolean prevAuto = c.getAutoCommit();
        try {
            if (opt.autoCommitAroundWrite() && prevAuto) {
                c.setAutoCommit(false);
            }

            ensureTable(c, df, qualified, opt, sqlite);
            insertRows(c, df, qualified, opt, sqlite);

            if (opt.autoCommitAroundWrite() && !prevAuto) {
                // leave transaction to caller
            } else if (opt.autoCommitAroundWrite()) {
                c.commit();
            }
        } catch (Exception e) {
            if (opt.autoCommitAroundWrite()) {
                try { c.rollback(); } catch (SQLException ignored) {}
            }
            throw e;
        } finally {
            if (opt.autoCommitAroundWrite() && prevAuto) {
                try { c.setAutoCommit(true); } catch (SQLException ignored) {}
            }
        }
    }

    private static void ensureTable(Connection c, DataFrame df, String qualified,
                                    SqlOptions opt, boolean sqlite) throws SQLException {
        boolean exists = tableExists(c, stripQuotes(qualified), sqlite);
        switch (opt.ifExists()) {
            case FAIL:
                if (exists) {
                    throw new SQLException("Table already exists: " + qualified
                        + " (ifExists=FAIL). Use REPLACE or APPEND.");
                }
                createTable(c, df, qualified, opt, sqlite);
                break;
            case REPLACE:
                if (exists) {
                    try (Statement st = c.createStatement()) {
                        st.execute("DROP TABLE " + qualified);
                    }
                }
                createTable(c, df, qualified, opt, sqlite);
                break;
            case APPEND:
                if (!exists) createTable(c, df, qualified, opt, sqlite);
                break;
        }
    }

    private static void createTable(Connection c, DataFrame df, String qualified,
                                    SqlOptions opt, boolean sqlite) throws SQLException {
        StringBuilder ddl = new StringBuilder();
        ddl.append("CREATE TABLE ").append(qualified).append(" (");
        List<String> cols = new ArrayList<>();
        if (opt.index()) {
            cols.add(SqlReader.ident(opt.indexLabel(), opt.quoteIdentifiers())
                + " " + JdbcTypeMap.toSqlType(Column.DType.INT64, sqlite));
        }
        for (int i = 0; i < df.columnCount(); i++) {
            Column col = df.column(i);
            Column.DType dt = col.dtype();
            if (opt.dtype() != null && opt.dtype().containsKey(col.name())) {
                dt = opt.dtype().get(col.name());
            }
            cols.add(SqlReader.ident(col.name(), opt.quoteIdentifiers())
                + " " + JdbcTypeMap.toSqlType(dt, sqlite));
        }
        ddl.append(String.join(", ", cols)).append(')');
        try (Statement st = c.createStatement()) {
            st.execute(ddl.toString());
        }
    }

    private static void insertRows(Connection c, DataFrame df, String qualified,
                                   SqlOptions opt, boolean sqlite) throws SQLException {
        List<String> colIdents = new ArrayList<>();
        if (opt.index()) colIdents.add(SqlReader.ident(opt.indexLabel(), opt.quoteIdentifiers()));
        for (int i = 0; i < df.columnCount(); i++) {
            colIdents.add(SqlReader.ident(df.column(i).name(), opt.quoteIdentifiers()));
        }
        StringBuilder sql = new StringBuilder();
        sql.append("INSERT INTO ").append(qualified).append(" (")
            .append(String.join(", ", colIdents)).append(") VALUES (");
        for (int i = 0; i < colIdents.size(); i++) {
            if (i > 0) sql.append(", ");
            sql.append('?');
        }
        sql.append(')');

        int chunk = opt.chunksize();
        try (PreparedStatement ps = c.prepareStatement(sql.toString())) {
            int batch = 0;
            for (int r = 0; r < df.rowCount(); r++) {
                int param = 1;
                if (opt.index()) {
                    ps.setLong(param++, r);
                }
                for (int cIdx = 0; cIdx < df.columnCount(); cIdx++) {
                    Column col = df.column(cIdx);
                    setParam(ps, param++, col.get(r), col.dtype(), sqlite);
                }
                ps.addBatch();
                batch++;
                if (batch >= chunk) {
                    ps.executeBatch();
                    batch = 0;
                }
            }
            if (batch > 0) ps.executeBatch();
        }
    }

    private static void setParam(PreparedStatement ps, int idx, Object val,
                                 Column.DType dtype, boolean sqlite) throws SQLException {
        if (val == null) {
            ps.setObject(idx, null);
            return;
        }
        switch (dtype) {
            case INT32:
                ps.setInt(idx, val instanceof Number ? ((Number) val).intValue()
                    : Integer.parseInt(String.valueOf(val)));
                break;
            case INT64:
                ps.setLong(idx, val instanceof Number ? ((Number) val).longValue()
                    : Long.parseLong(String.valueOf(val)));
                break;
            case FLOAT32:
                ps.setFloat(idx, val instanceof Number ? ((Number) val).floatValue()
                    : Float.parseFloat(String.valueOf(val)));
                break;
            case FLOAT64:
                ps.setDouble(idx, val instanceof Number ? ((Number) val).doubleValue()
                    : Double.parseDouble(String.valueOf(val)));
                break;
            case BOOLEAN:
                if (sqlite) {
                    boolean b = val instanceof Boolean ? (Boolean) val
                        : Boolean.parseBoolean(String.valueOf(val));
                    ps.setInt(idx, b ? 1 : 0);
                } else {
                    ps.setBoolean(idx, val instanceof Boolean ? (Boolean) val
                        : Boolean.parseBoolean(String.valueOf(val)));
                }
                break;
            case DATE:
                if (val instanceof LocalDate) {
                    ps.setDate(idx, java.sql.Date.valueOf((LocalDate) val));
                } else if (val instanceof java.sql.Date) {
                    ps.setDate(idx, (java.sql.Date) val);
                } else {
                    ps.setString(idx, String.valueOf(val));
                }
                break;
            case DATETIME:
                if (val instanceof LocalDateTime) {
                    ps.setTimestamp(idx, Timestamp.valueOf((LocalDateTime) val));
                } else if (val instanceof Instant) {
                    ps.setTimestamp(idx, Timestamp.from((Instant) val));
                } else if (val instanceof Timestamp) {
                    ps.setTimestamp(idx, (Timestamp) val);
                } else {
                    ps.setString(idx, String.valueOf(val));
                }
                break;
            case TIME:
                if (val instanceof LocalTime) {
                    ps.setTime(idx, java.sql.Time.valueOf((LocalTime) val));
                } else {
                    ps.setString(idx, String.valueOf(val));
                }
                break;
            case BINARY:
                if (val instanceof byte[]) ps.setBytes(idx, (byte[]) val);
                else ps.setBytes(idx, String.valueOf(val).getBytes(java.nio.charset.StandardCharsets.UTF_8));
                break;
            default:
                ps.setString(idx, String.valueOf(val));
        }
    }

    private static boolean tableExists(Connection c, String table, boolean sqlite) {
        // table may be schema.table
        String name = table;
        String schema = null;
        int dot = table.lastIndexOf('.');
        if (dot > 0) {
            schema = table.substring(0, dot);
            name = table.substring(dot + 1);
        }
        try {
            var meta = c.getMetaData();
            // SQLite: schema often "main"
            try (var rs = meta.getTables(null, schema, name, new String[]{"TABLE", "VIEW"})) {
                if (rs.next()) return true;
            }
            // try case-insensitive for SQLite
            try (var rs = meta.getTables(null, null, null, new String[]{"TABLE", "VIEW"})) {
                while (rs.next()) {
                    String t = rs.getString("TABLE_NAME");
                    if (t != null && t.equalsIgnoreCase(name)) return true;
                }
            }
        } catch (SQLException ignored) {}
        return false;
    }

    private static String stripQuotes(String ident) {
        String s = ident;
        if (s.startsWith("\"") && s.endsWith("\"") && s.length() >= 2) {
            s = s.substring(1, s.length() - 1).replace("\"\"", "\"");
        }
        return s;
    }
}
