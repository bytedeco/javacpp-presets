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
package org.bytedeco.pytorch.utils.duckdb;

import org.bytedeco.pytorch.data.dataframe.DataFrame;
import org.bytedeco.pytorch.data.dataframe.sql.SqlReader;
import org.bytedeco.pytorch.data.dataframe.sql.SqlWriter;

import java.io.Closeable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**
 * Official <a href="https://duckdb.org/docs/stable/clients/java">DuckDB JDBC</a> wrapper with
 * first-class {@link DataFrame} interop.
 *
 * <p>Uses {@code org.duckdb:duckdb_jdbc} (currently {@code 1.5.5.0}). DuckDB natively scans
 * Parquet / CSV / JSON / ORC / Arrow via table functions — no materialization required for
 * {@code read_parquet}/{@code read_csv_auto}/{@code read_json_auto}/{@code read_orc}.
 *
 * <pre>{@code
 * try (DuckDB db = DuckDB.inMemory()) {
 *     DataFrame df = db.query("SELECT * FROM read_parquet('data.parquet') WHERE x > 0");
 *     db.register("t", df);                 // DataFrame → DuckDB table
 *     DataFrame agg = db.query("SELECT label, count(*) n FROM t GROUP BY 1");
 *     db.exportParquet(agg, "out.parquet"); // COPY ... TO
 * }
 * }</pre>
 *
 * <p>Also exposes static helpers that open a short-lived connection for one-shot scans.
 */
public final class DuckDB implements Closeable {

    public static final String VERSION = "1.5.5.0";
    public static final String DRIVER = "org.duckdb.DuckDBDriver";
    public static final String URL_MEMORY = "jdbc:duckdb:";
    public static final String URL_PREFIX = "jdbc:duckdb:";

    static {
        try {
            Class.forName(DRIVER);
        } catch (ClassNotFoundException e) {
            // DriverManager can still resolve via ServiceLoader / JDBC 4+
        }
    }

    private final Connection connection;
    private final String url;
    private final boolean owned;
    private final Map<String, String> registered = new LinkedHashMap<>();

    private DuckDB(Connection connection, String url, boolean owned) {
        this.connection = Objects.requireNonNull(connection, "connection");
        this.url = url == null ? URL_MEMORY : url;
        this.owned = owned;
    }

    // ---- factories -------------------------------------------------------

    /** In-memory DuckDB database. */
    public static DuckDB inMemory() throws SQLException {
        return open(URL_MEMORY, null);
    }

    /** Persistent DuckDB database file (created if missing). */
    public static DuckDB open(Path dbFile) throws Exception {
        Objects.requireNonNull(dbFile, "dbFile");
        if (dbFile.getParent() != null) {
            Files.createDirectories(dbFile.getParent());
        }
        return open(URL_PREFIX + dbFile.toAbsolutePath(), null);
    }

    public static DuckDB open(String jdbcUrl) throws SQLException {
        return open(jdbcUrl, null);
    }

    public static DuckDB open(String jdbcUrl, Properties props) throws SQLException {
        Connection c = props == null
                ? DriverManager.getConnection(jdbcUrl)
                : DriverManager.getConnection(jdbcUrl, props);
        return new DuckDB(c, jdbcUrl, true);
    }

    /** Wrap an existing DuckDB (or compatible) connection without taking ownership. */
    public static DuckDB wrap(Connection connection) {
        return new DuckDB(connection, null, false);
    }

    public Connection connection() {
        return connection;
    }

    public String url() {
        return url;
    }

    public Map<String, String> registeredTables() {
        return Collections.unmodifiableMap(registered);
    }

    // ---- SQL / DataFrame -------------------------------------------------

    /** Run a SQL query and materialize the result as a {@link DataFrame}. */
    public DataFrame query(String sql) throws Exception {
        Objects.requireNonNull(sql, "sql");
        return SqlReader.read(connection, sql);
    }

    /** Execute a non-query statement (DDL / DML / COPY / CREATE VIEW …). */
    public boolean execute(String sql) throws SQLException {
        try (Statement st = connection.createStatement()) {
            return st.execute(sql);
        }
    }

    public int executeUpdate(String sql) throws SQLException {
        try (Statement st = connection.createStatement()) {
            return st.executeUpdate(sql);
        }
    }

    /**
     * Register a {@link DataFrame} as a physical DuckDB table (via JDBC insert).
     * Replaces any existing table of the same name.
     */
    public void register(String table, DataFrame df) throws Exception {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        // REPLACE drops+recreates; works for DuckDB and SQLite dialect paths in SqlWriter
        var opts = org.bytedeco.pytorch.data.dataframe.sql.SqlOptions.builder()
                .ifExists(org.bytedeco.pytorch.data.dataframe.sql.SqlOptions.IfExists.REPLACE)
                .quoteIdentifiers(true)
                .index(false)
                .build();
        df.toSql(connection, table, opts);
        registered.put(table, "dataframe rows=" + df.rowCount());
    }

    /** Drop a previously registered / created table. */
    public void unregister(String table) throws SQLException {
        execute("DROP TABLE IF EXISTS " + sanitizeIdent(table));
        registered.remove(table);
    }

    /**
     * Create a DuckDB view that scans an external file via a table function
     * ({@code read_parquet}, {@code read_csv_auto}, {@code read_json_auto}, {@code read_orc}, …).
     */
    public void registerFile(String table, String path, FileFormat format) throws SQLException {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(path, "path");
        FileFormat fmt = format == null ? detectFormat(path) : format;
        String fn = fmt.tableFunction(path);
        String t = sanitizeIdent(table);
        execute("CREATE OR REPLACE VIEW " + t + " AS SELECT * FROM " + fn);
        registered.put(table, fmt.name().toLowerCase(Locale.ROOT) + ":" + path);
    }

    public void registerParquet(String table, String path) throws SQLException {
        registerFile(table, path, FileFormat.PARQUET);
    }

    public void registerCsv(String table, String path) throws SQLException {
        registerFile(table, path, FileFormat.CSV);
    }

    public void registerJson(String table, String path) throws SQLException {
        registerFile(table, path, FileFormat.JSON);
    }

    public void registerOrc(String table, String path) throws SQLException {
        registerFile(table, path, FileFormat.ORC);
    }

    public void registerArrow(String table, String path) throws SQLException {
        registerFile(table, path, FileFormat.ARROW);
    }

    // ---- one-shot file scans (static + instance) -------------------------

    public DataFrame readParquet(String path) throws Exception {
        return query("SELECT * FROM " + FileFormat.PARQUET.tableFunction(path));
    }

    public DataFrame readCsv(String path) throws Exception {
        return query("SELECT * FROM " + FileFormat.CSV.tableFunction(path));
    }

    public DataFrame readJson(String path) throws Exception {
        return query("SELECT * FROM " + FileFormat.JSON.tableFunction(path));
    }

    public DataFrame readOrc(String path) throws Exception {
        return query("SELECT * FROM " + FileFormat.ORC.tableFunction(path));
    }

    public DataFrame readArrow(String path) throws Exception {
        return query("SELECT * FROM " + FileFormat.ARROW.tableFunction(path));
    }

    /** One-shot in-memory scan without keeping a DuckDB instance open. */
    public static DataFrame scanParquet(String path) throws Exception {
        try (DuckDB db = inMemory()) {
            return db.readParquet(path);
        }
    }

    public static DataFrame scanCsv(String path) throws Exception {
        try (DuckDB db = inMemory()) {
            return db.readCsv(path);
        }
    }

    public static DataFrame scanJson(String path) throws Exception {
        try (DuckDB db = inMemory()) {
            return db.readJson(path);
        }
    }

    public static DataFrame scanOrc(String path) throws Exception {
        try (DuckDB db = inMemory()) {
            return db.readOrc(path);
        }
    }

    public static DataFrame scanSql(String sql) throws Exception {
        try (DuckDB db = inMemory()) {
            return db.query(sql);
        }
    }

    // ---- export / COPY ---------------------------------------------------

    /** {@code COPY (query) TO 'path' (FORMAT PARQUET)}. */
    public void exportParquet(String sqlOrTable, String path) throws SQLException {
        export(sqlOrTable, path, FileFormat.PARQUET);
    }

    public void exportCsv(String sqlOrTable, String path) throws SQLException {
        export(sqlOrTable, path, FileFormat.CSV);
    }

    public void exportJson(String sqlOrTable, String path) throws SQLException {
        export(sqlOrTable, path, FileFormat.JSON);
    }

    public void export(String sqlOrTable, String path, FileFormat format) throws SQLException {
        Objects.requireNonNull(sqlOrTable, "sqlOrTable");
        Objects.requireNonNull(path, "path");
        FileFormat fmt = format == null ? FileFormat.PARQUET : format;
        String source = looksLikeSql(sqlOrTable)
                ? "(" + sqlOrTable + ")"
                : sanitizeIdent(sqlOrTable);
        String opts = fmt.copyOptions();
        String sql = "COPY " + source + " TO '" + escapePath(path) + "'"
                + (opts.isEmpty() ? "" : " (" + opts + ")");
        execute(sql);
    }

    /** Write a DataFrame to Parquet via DuckDB (register temp → COPY). */
    public void exportParquet(DataFrame df, String path) throws Exception {
        String tmp = "_df_export_" + System.nanoTime();
        try {
            register(tmp, df);
            exportParquet(tmp, path);
        } finally {
            try { unregister(tmp); } catch (SQLException ignored) {}
        }
    }

    public void exportCsv(DataFrame df, String path) throws Exception {
        String tmp = "_df_export_" + System.nanoTime();
        try {
            register(tmp, df);
            exportCsv(tmp, path);
        } finally {
            try { unregister(tmp); } catch (SQLException ignored) {}
        }
    }

    // ---- catalog helpers -------------------------------------------------

    public List<String> tables() throws SQLException {
        List<String> out = new ArrayList<>();
        try (ResultSet rs = connection.getMetaData().getTables(null, null, "%", new String[]{"TABLE", "VIEW"})) {
            while (rs.next()) {
                out.add(rs.getString("TABLE_NAME"));
            }
        }
        return out;
    }

    public DataFrame showTables() throws Exception {
        return query("SHOW TABLES");
    }

    public DataFrame describe(String table) throws Exception {
        return query("DESCRIBE " + sanitizeIdent(table));
    }

    public DataFrame summarize(String table) throws Exception {
        return query("SUMMARIZE " + sanitizeIdent(table));
    }

    /** DuckDB version string from {@code SELECT version()}. */
    public String duckdbVersion() throws Exception {
        DataFrame df = query("SELECT version() AS v");
        if (df.rowCount() == 0) return VERSION;
        Object v = df.get(0, "v");
        return v == null ? VERSION : v.toString();
    }

    // ---- interop aliases -------------------------------------------------

    /** Alias of {@link #register(String, DataFrame)}. */
    public void fromDataFrame(String table, DataFrame df) throws Exception {
        register(table, df);
    }

    /** Alias of {@link #query(String)}. */
    public DataFrame toDataFrame(String sql) throws Exception {
        return query(sql);
    }

    public DataFrame tableToDataFrame(String table) throws Exception {
        return query("SELECT * FROM " + sanitizeIdent(table));
    }

    // ---- close -----------------------------------------------------------

    @Override
    public void close() {
        if (owned) {
            try {
                connection.close();
            } catch (SQLException ignored) {
            }
        }
    }

    // ---- format enum -----------------------------------------------------

    public enum FileFormat {
        PARQUET {
            @Override public String tableFunction(String path) {
                return "read_parquet('" + escapePath(path) + "')";
            }
            @Override String copyOptions() { return "FORMAT PARQUET"; }
        },
        CSV {
            @Override public String tableFunction(String path) {
                return "read_csv_auto('" + escapePath(path) + "')";
            }
            @Override String copyOptions() { return "FORMAT CSV, HEADER true"; }
        },
        JSON {
            @Override public String tableFunction(String path) {
                return "read_json_auto('" + escapePath(path) + "')";
            }
            @Override String copyOptions() { return "FORMAT JSON"; }
        },
        ORC {
            @Override public String tableFunction(String path) {
                return "read_orc('" + escapePath(path) + "')";
            }
            @Override String copyOptions() { return "FORMAT PARQUET"; } // ORC write via parquet fallback
        },
        ARROW {
            @Override public String tableFunction(String path) {
                // DuckDB can read Arrow IPC via read_parquet on some builds; prefer arrow_scan when available
                return "read_parquet('" + escapePath(path) + "')";
            }
            @Override String copyOptions() { return "FORMAT PARQUET"; }
        };

        public abstract String tableFunction(String path);
        abstract String copyOptions();
    }

    // ---- helpers ---------------------------------------------------------

    public static FileFormat detectFormat(String path) {
        String p = path.toLowerCase(Locale.ROOT);
        if (p.endsWith(".parquet") || p.endsWith(".pq")) return FileFormat.PARQUET;
        if (p.endsWith(".csv") || p.endsWith(".tsv")) return FileFormat.CSV;
        if (p.endsWith(".json") || p.endsWith(".jsonl") || p.endsWith(".ndjson")) return FileFormat.JSON;
        if (p.endsWith(".orc")) return FileFormat.ORC;
        if (p.endsWith(".arrow") || p.endsWith(".feather") || p.endsWith(".ipc")) return FileFormat.ARROW;
        return FileFormat.PARQUET;
    }

    static String escapePath(String path) {
        return path.replace("'", "''");
    }

    static String sanitizeIdent(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("empty identifier");
        }
        if (name.matches("[A-Za-z_][A-Za-z0-9_]*")) {
            return name;
        }
        return "\"" + name.replace("\"", "\"\"") + "\"";
    }

    private static boolean looksLikeSql(String s) {
        String t = s.trim().toLowerCase(Locale.ROOT);
        return t.startsWith("select") || t.startsWith("with") || t.startsWith("from")
                || t.contains(" ") || t.contains("(");
    }
}
