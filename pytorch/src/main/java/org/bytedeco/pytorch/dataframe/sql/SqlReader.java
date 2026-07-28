package org.bytedeco.pytorch.dataframe.sql;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.Statement;

/**
 * Read JDBC {@link ResultSet} / SQL queries into a {@link DataFrame}.
 */
public final class SqlReader {
    private SqlReader() {}

    public static DataFrame read(Connection c, String sql) throws Exception {
        return read(c, sql, SqlOptions.defaults());
    }

    public static DataFrame read(Connection c, String sql, SqlOptions options) throws Exception {
        if (c == null) throw new IllegalArgumentException("connection required");
        if (sql == null || sql.isBlank()) throw new IllegalArgumentException("sql required");
        SqlOptions opt = options == null ? SqlOptions.defaults() : options;
        try (Statement st = c.createStatement()) {
            if (opt.fetchSize() > 0) {
                try { st.setFetchSize(opt.fetchSize()); } catch (Exception ignored) {}
            }
            try (ResultSet rs = st.executeQuery(sql)) {
                return fromResultSet(rs, opt);
            }
        }
    }

    public static DataFrame readTable(Connection c, String table) throws Exception {
        return readTable(c, table, SqlOptions.defaults());
    }

    public static DataFrame readTable(Connection c, String table, SqlOptions options) throws Exception {
        SqlOptions opt = options == null ? SqlOptions.defaults() : options;
        String qualified = qualify(table, opt);
        return read(c, "SELECT * FROM " + qualified, opt);
    }

    public static DataFrame fromResultSet(ResultSet rs) throws Exception {
        return fromResultSet(rs, SqlOptions.defaults());
    }

    public static DataFrame fromResultSet(ResultSet rs, SqlOptions options) throws Exception {
        SqlOptions opt = options == null ? SqlOptions.defaults() : options;
        ResultSetMetaData meta = rs.getMetaData();
        String[] names = JdbcTypeMap.namesFromMeta(meta);
        Column.DType[] dtypes = JdbcTypeMap.dtypesFromMeta(meta);

        // optional dtype overrides
        if (opt.dtype() != null) {
            for (int i = 0; i < names.length; i++) {
                Column.DType t = opt.dtype().get(names[i]);
                if (t != null) dtypes[i] = t;
            }
        }

        DataFrame df = DataFrame.create();
        for (int i = 0; i < names.length; i++) df.addColumn(names[i], dtypes[i]);

        int max = -1; // unlimited; SqlOptions has no maxRows — use full result
        while (rs.next()) {
            int ri = df.addEmptyRow();
            for (int i = 0; i < names.length; i++) {
                Object v = JdbcTypeMap.getValue(rs, i + 1, dtypes[i]);
                df.set(ri, names[i], v);
            }
            if (max >= 0 && df.rowCount() >= max) break;
        }
        return df;
    }

    static String qualify(String table, SqlOptions opt) {
        String t = table.trim();
        boolean quote = opt.quoteIdentifiers();
        if (opt.schema() != null && !opt.schema().isEmpty() && !t.contains(".")) {
            return ident(opt.schema(), quote) + "." + ident(t, quote);
        }
        // already qualified or plain
        if (t.contains(".") && !t.contains("\"")) {
            String[] parts = t.split("\\.", 2);
            return ident(parts[0], quote) + "." + ident(parts[1], quote);
        }
        return ident(t, quote);
    }

    static String ident(String name, boolean quote) {
        if (!quote) return name;
        // strip existing quotes
        String n = name;
        if ((n.startsWith("\"") && n.endsWith("\"")) || (n.startsWith("`") && n.endsWith("`"))) {
            return n;
        }
        return "\"" + n.replace("\"", "\"\"") + "\"";
    }
}
