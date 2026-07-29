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
import org.bytedeco.pytorch.dataframe.sql.SqlOptions;
import org.bytedeco.pytorch.dataframe.sql.SqlReader;
import org.duckdb.DuckDBAppender;
import org.duckdb.DuckDBConnection;
import org.duckdb.DuckDBDriver;
import org.duckdb.DuckDBFunctions;
import org.duckdb.DuckDBPreparedStatement;
import org.duckdb.DuckDBResultSet;
import org.duckdb.DuckDBScalarFunctionBuilder;
import org.duckdb.DuckDBSingleValueAppender;
import org.duckdb.DuckDBTableFunctionBuilder;
import org.duckdb.ProfilerPrintFormat;
import org.duckdb.QueryProgress;

import java.io.Closeable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
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
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * Enterprise wrapper over official <a href="https://duckdb.org/docs/stable/clients/java">DuckDB JDBC</a>
 * ({@code org.duckdb:duckdb_jdbc}) with first-class {@link DataFrame} interop.
 *
 * <p>Uses the real SDK surface — not just generic JDBC:
 * <ul>
 *   <li>{@link DuckDBConnection} — {@code duplicate()}, {@code createAppender()},
 *       {@code registerArrowStream()}, profiling</li>
 *   <li>{@link DuckDBAppender} — bulk columnar ingest via {@link DuckDBAppenderWriter}</li>
 *   <li>{@link DuckDBFunctions} — Java scalar / table UDFs</li>
 *   <li>{@link DuckDBPreparedStatement#getQueryProgress()} — long-query progress</li>
 *   <li>{@link DuckDBDriver} connection properties + session {@code SET} via {@link DuckDBConfig}</li>
 * </ul>
 *
 * <h2>Typical recsys / multimodal usage</h2>
 * <pre>{@code
 * // Offline feature join (Meta/ByteDance style)
 * try (DuckDB db = DuckDB.open(Path.of("features.duckdb"),
 *         DuckDBConfig.offlineFeatureEngineering().memoryLimit("16GB"))) {
 *     db.registerParquet("events", "s3://bucket/events/**.parquet"); // or local glob
 *     db.registerParquet("user_feat", "user_feat.parquet");
 *     DataFrame train = db.query("""
 *         SELECT e.*, u.age_bucket, u.city_hash
 *         FROM events e LEFT JOIN user_feat u USING (user_id)
 *         WHERE e.dt BETWEEN '2024-01-01' AND '2024-01-07'
 *         """);
 *     db.appendDataFrame("train_samples", train);   // Appender path
 *     db.exportParquet("train_samples", "train.parquet");
 * }
 *
 * // Multimodal catalog
 * try (DuckDB db = DuckDB.inMemory(DuckDBConfig.multimodalCatalog())) {
 *     db.ensureMediaCatalog();
 *     db.upsertMediaMeta("img_001", "image", "/data/a.jpg", 1024, 768, null, emb);
 * }
 * }</pre>
 *
 * @see DuckDBConfig
 * @see DuckDBAppenderWriter
 * @see DuckDBPool
 * @see DuckDBFeatureStore
 * @see DuckDBMultimodal
 * @see DuckDBAnalytics
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
    private final DuckDBConnection nativeConn;
    private final String url;
    private final boolean owned;
    private final DuckDBConfig config;
    private final Map<String, String> registered = new LinkedHashMap<>();

    private DuckDB(Connection connection, String url, boolean owned, DuckDBConfig config) {
        this.connection = Objects.requireNonNull(connection, "connection");
        this.url = url == null ? URL_MEMORY : url;
        this.owned = owned;
        this.config = config;
        DuckDBConnection nc = null;
        try {
            if (connection instanceof DuckDBConnection) {
                nc = (DuckDBConnection) connection;
            } else if (connection.isWrapperFor(DuckDBConnection.class)) {
                nc = connection.unwrap(DuckDBConnection.class);
            }
        } catch (SQLException ignored) {
            nc = null;
        }
        this.nativeConn = nc;
    }

    // ---- factories -------------------------------------------------------

    /** In-memory DuckDB with default analytics config. */
    public static DuckDB inMemory() throws SQLException {
        return inMemory(DuckDBConfig.analytics());
    }

    public static DuckDB inMemory(DuckDBConfig config) throws SQLException {
        return open(URL_MEMORY, config);
    }

    /** Persistent DuckDB database file (created if missing). */
    public static DuckDB open(Path dbFile) throws Exception {
        return open(dbFile, DuckDBConfig.create());
    }

    public static DuckDB open(Path dbFile, DuckDBConfig config) throws Exception {
        Objects.requireNonNull(dbFile, "dbFile");
        if (dbFile.getParent() != null) {
            Files.createDirectories(dbFile.getParent());
        }
        return open(URL_PREFIX + dbFile.toAbsolutePath(), config);
    }

    public static DuckDB open(String jdbcUrl) throws SQLException {
        return open(jdbcUrl, (DuckDBConfig) null);
    }

    public static DuckDB open(String jdbcUrl, Properties props) throws SQLException {
        Connection c = props == null
                ? DriverManager.getConnection(jdbcUrl)
                : DriverManager.getConnection(jdbcUrl, props);
        return new DuckDB(c, jdbcUrl, true, null);
    }

    public static DuckDB open(String jdbcUrl, DuckDBConfig config) throws SQLException {
        DuckDBConfig cfg = config == null ? DuckDBConfig.create() : config;
        Properties props = cfg.toJdbcProperties();
        Connection c;
        try {
            // Prefer official factory when possible
            c = DuckDBConnection.newConnection(jdbcUrl, false, props);
        } catch (Exception e) {
            try {
                c = DriverManager.getConnection(jdbcUrl, props);
            } catch (SQLException se) {
                se.addSuppressed(e instanceof SQLException ? (SQLException) e
                        : new SQLException(e));
                throw se;
            }
        }
        DuckDB db = new DuckDB(c, jdbcUrl, true, cfg);
        cfg.apply(c);
        return db;
    }

    /** Read-only open of an existing DB file. */
    public static DuckDB openReadOnly(Path dbFile) throws Exception {
        return open(dbFile, DuckDBConfig.readOnlyServing());
    }

    /** Wrap an existing connection without taking ownership. */
    public static DuckDB wrap(Connection connection) {
        return new DuckDB(connection, null, false, null);
    }

    /**
     * Wrap a native {@link DuckDBConnection} (e.g. from {@link DuckDBConnection#duplicate()})
     * optionally taking ownership.
     */
    public static DuckDB wrapNative(DuckDBConnection connection, String url, boolean owned) {
        return new DuckDB(connection, url, owned, null);
    }

    // ---- accessors -------------------------------------------------------

    public Connection connection() {
        return connection;
    }

    /**
     * Official DuckDB connection, or {@code null} if the underlying JDBC connection
     * is not a {@link DuckDBConnection} (should not happen with duckdb_jdbc).
     */
    public DuckDBConnection nativeConnection() throws SQLException {
        if (nativeConn != null) return nativeConn;
        throw new SQLException("Underlying connection is not org.duckdb.DuckDBConnection — "
                + "ensure org.duckdb:duckdb_jdbc is on the classpath and URL is jdbc:duckdb:");
    }

    public boolean hasNative() {
        return nativeConn != null;
    }

    public String url() {
        return url;
    }

    public DuckDBConfig config() {
        return config;
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

    public DataFrame query(String sql, SqlOptions options) throws Exception {
        Objects.requireNonNull(sql, "sql");
        return SqlReader.read(connection, sql, options);
    }

    /** Parameterized query ({@code ?} placeholders). */
    public DataFrame query(String sql, Object... params) throws Exception {
        Objects.requireNonNull(sql, "sql");
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            bindParams(ps, params);
            try (ResultSet rs = ps.executeQuery()) {
                return SqlReader.fromResultSet(rs);
            }
        }
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

    public int executeUpdate(String sql, Object... params) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            bindParams(ps, params);
            return ps.executeUpdate();
        }
    }

    /** Run multiple statements (scripts with {@code ;} separators via Statement). */
    public void executeScript(String script) throws SQLException {
        Objects.requireNonNull(script, "script");
        try (Statement st = connection.createStatement()) {
            for (String part : splitStatements(script)) {
                if (!part.isBlank()) st.execute(part);
            }
        }
    }

    /**
     * Official prepared statement with {@link QueryProgress} support.
     * Caller must close the returned statement.
     */
    public DuckDBPreparedStatement prepareNative(String sql) throws SQLException {
        return nativeConnection().prepare(sql);
    }

    public QueryProgress queryProgress(DuckDBPreparedStatement ps) throws SQLException {
        return ps.getQueryProgress();
    }

    // ---- Appender bulk path (official SDK) --------------------------------

    /**
     * Create a table from DataFrame schema (if needed) and bulk-load via
     * {@link DuckDBAppender}. Prefer this over {@link #register} for &gt;100k rows.
     */
    public long appendDataFrame(String table, DataFrame df) throws Exception {
        return appendDataFrame(table, df, true);
    }

    public long appendDataFrame(String table, DataFrame df, boolean createIfMissing) throws Exception {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        if (createIfMissing) {
            ensureTableFromDataFrame(table, df, false);
        }
        long n = DuckDBAppenderWriter.appendDataFrame(this, table, df);
        registered.put(table, "appender rows+=" + n + " total_df=" + df.rowCount());
        return n;
    }

    /** REPLACE semantics: DROP + CREATE + Appender load. */
    public long replaceWithDataFrame(String table, DataFrame df) throws Exception {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        ensureTableFromDataFrame(table, df, true);
        long n = DuckDBAppenderWriter.appendDataFrame(this, table, df);
        registered.put(table, "appender replace rows=" + n);
        return n;
    }

    public DuckDBAppender createAppender(String table) throws SQLException {
        return nativeConnection().createAppender(table);
    }

    public DuckDBAppender createAppender(String schema, String table) throws SQLException {
        return nativeConnection().createAppender(schema, table);
    }

    /** @deprecated prefer {@link #createAppender(String, String)} (official bulk path). */
    @Deprecated
    public DuckDBSingleValueAppender createSingleValueAppender(String schema, String table)
            throws SQLException {
        return nativeConnection().createSingleValueAppender(schema, table);
    }

    public DuckDBAppenderWriter appender(String table, int columnCount) throws SQLException {
        return DuckDBAppenderWriter.open(nativeConnection(), table, columnCount);
    }

    // ---- register DataFrame (JDBC path, small data) -----------------------

    /**
     * Register a {@link DataFrame} as a physical DuckDB table (via JDBC insert).
     * For large frames prefer {@link #appendDataFrame} / {@link #replaceWithDataFrame}.
     */
    public void register(String table, DataFrame df) throws Exception {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        var opts = SqlOptions.builder()
                .ifExists(SqlOptions.IfExists.REPLACE)
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

    /** Glob-friendly parquet registration (DuckDB list / hive partitioning). */
    public void registerParquet(String table, String path, boolean hivePartitioning)
            throws SQLException {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(path, "path");
        String t = sanitizeIdent(table);
        String opts = hivePartitioning ? ", hive_partitioning=true" : "";
        execute("CREATE OR REPLACE VIEW " + t + " AS SELECT * FROM read_parquet('"
                + escapePath(path) + "'" + opts + ")");
        registered.put(table, "parquet:" + path);
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

    /**
     * Register an Arrow stream object via official
     * {@link DuckDBConnection#registerArrowStream(String, Object)}.
     * {@code arrowStream} is typically an Arrow Stream / Reader from the Arrow Java API.
     */
    public void registerArrowStream(String viewName, Object arrowStream) throws SQLException {
        Objects.requireNonNull(viewName, "viewName");
        Objects.requireNonNull(arrowStream, "arrowStream");
        nativeConnection().registerArrowStream(viewName, arrowStream);
        registered.put(viewName, "arrow_stream");
    }

    // ---- one-shot file scans (static + instance) -------------------------

    public DataFrame readParquet(String path) throws Exception {
        return query("SELECT * FROM " + FileFormat.PARQUET.tableFunction(path));
    }

    public DataFrame readParquet(String path, String... columns) throws Exception {
        if (columns == null || columns.length == 0) return readParquet(path);
        String cols = String.join(", ", columns);
        return query("SELECT " + cols + " FROM " + FileFormat.PARQUET.tableFunction(path));
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

    /** Partitioned parquet export (Hive-style) for training dumps. */
    public void exportParquetPartitioned(String sqlOrTable, String dir, String... partitionCols)
            throws SQLException {
        Objects.requireNonNull(sqlOrTable, "sqlOrTable");
        Objects.requireNonNull(dir, "dir");
        String source = looksLikeSql(sqlOrTable)
                ? "(" + sqlOrTable + ")"
                : sanitizeIdent(sqlOrTable);
        StringBuilder opts = new StringBuilder("FORMAT PARQUET");
        if (partitionCols != null && partitionCols.length > 0) {
            opts.append(", PARTITION_BY (");
            for (int i = 0; i < partitionCols.length; i++) {
                if (i > 0) opts.append(", ");
                opts.append(sanitizeIdent(partitionCols[i]));
            }
            opts.append(')');
        }
        execute("COPY " + source + " TO '" + escapePath(dir) + "' (" + opts + ")");
    }

    /** Write a DataFrame to Parquet via DuckDB (Appender/register temp → COPY). */
    public void exportParquet(DataFrame df, String path) throws Exception {
        String tmp = "_df_export_" + System.nanoTime();
        try {
            replaceWithDataFrame(tmp, df);
            exportParquet(tmp, path);
        } finally {
            try { unregister(tmp); } catch (SQLException ignored) {}
        }
    }

    public void exportCsv(DataFrame df, String path) throws Exception {
        String tmp = "_df_export_" + System.nanoTime();
        try {
            replaceWithDataFrame(tmp, df);
            exportCsv(tmp, path);
        } finally {
            try { unregister(tmp); } catch (SQLException ignored) {}
        }
    }

    public void exportJson(DataFrame df, String path) throws Exception {
        String tmp = "_df_export_" + System.nanoTime();
        try {
            replaceWithDataFrame(tmp, df);
            exportJson(tmp, path);
        } finally {
            try { unregister(tmp); } catch (SQLException ignored) {}
        }
    }

    /** Write DataFrame into a named table of this DuckDB instance. */
    public void writeTable(String table, DataFrame df) throws Exception {
        replaceWithDataFrame(table, df);
    }

    /** Write DataFrame into a named table with SqlOptions (REPLACE/APPEND/FAIL). */
    public void writeTable(String table, DataFrame df, SqlOptions options) throws Exception {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        SqlOptions opt = options == null
                ? SqlOptions.builder().ifExists(SqlOptions.IfExists.REPLACE).build()
                : options;
        if (opt.ifExists() == SqlOptions.IfExists.APPEND) {
            ensureTableFromDataFrame(table, df, false);
            appendDataFrame(table, df, false);
        } else if (opt.ifExists() == SqlOptions.IfExists.REPLACE) {
            replaceWithDataFrame(table, df);
        } else {
            df.toSql(connection, table, opt);
            registered.put(table, "dataframe rows=" + df.rowCount());
        }
    }

    // ---- extensions / secrets / httpfs (common enterprise path) ----------

    public void installExtension(String name) throws SQLException {
        execute("INSTALL " + sanitizeIdent(name));
    }

    public void loadExtension(String name) throws SQLException {
        execute("LOAD " + sanitizeIdent(name));
    }

    public void installAndLoad(String name) throws SQLException {
        installExtension(name);
        loadExtension(name);
    }

    /** Convenience: httpfs for remote parquet/csv (S3/GCS/HTTP). */
    public void enableHttpfs() throws SQLException {
        installAndLoad("httpfs");
    }

    public void enableJson() throws SQLException {
        installAndLoad("json");
    }

    public void enableIcu() throws SQLException {
        installAndLoad("icu");
    }

    /** Set S3 credentials for httpfs (explicit keys; prefer env/IAM in prod). */
    public void configureS3(String accessKey, String secretKey, String region) throws SQLException {
        enableHttpfs();
        if (region != null) execute("SET s3_region='" + escapePath(region) + "'");
        if (accessKey != null) execute("SET s3_access_key_id='" + escapePath(accessKey) + "'");
        if (secretKey != null) execute("SET s3_secret_access_key='" + escapePath(secretKey) + "'");
    }

    // ---- Java UDFs (official DuckDBFunctions) ----------------------------

    /**
     * Register a simple Java scalar UDF (e.g. feature hash, bucketize).
     * Uses {@link DuckDBFunctions#scalarFunction()}.
     */
    public void registerScalar(String name, Function<Object, Object> fn,
                               Class<?> inType, Class<?> outType) throws SQLException {
        try (DuckDBScalarFunctionBuilder b = DuckDBFunctions.scalarFunction()) {
            b.withName(name)
                    .withParameter(inType)
                    .withReturnType(outType)
                    .withFunction(fn)
                    .register(connection);
        }
    }

    public void registerScalar(String name, BiFunction<Object, Object, Object> fn,
                               Class<?> left, Class<?> right, Class<?> outType) throws SQLException {
        try (DuckDBScalarFunctionBuilder b = DuckDBFunctions.scalarFunction()) {
            b.withName(name)
                    .withParameter(left)
                    .withParameter(right)
                    .withReturnType(outType)
                    .withFunction(fn)
                    .register(connection);
        }
    }

    public DuckDBScalarFunctionBuilder scalarFunction() throws SQLException {
        return DuckDBFunctions.scalarFunction();
    }

    public DuckDBTableFunctionBuilder tableFunction() throws SQLException {
        return DuckDBFunctions.tableFunction();
    }

    // ---- profiling / settings / catalog ----------------------------------

    public void enableProfiling() throws SQLException {
        execute("PRAGMA enable_profiling");
    }

    public void disableProfiling() throws SQLException {
        execute("PRAGMA disable_profiling");
    }

    public String profilingInformation() throws SQLException {
        return profilingInformation(ProfilerPrintFormat.QUERY_TREE);
    }

    public String profilingInformation(ProfilerPrintFormat format) throws SQLException {
        return nativeConnection().getProfilingInformation(
                format == null ? ProfilerPrintFormat.QUERY_TREE : format);
    }

    public void set(String name, String value) throws SQLException {
        Objects.requireNonNull(name, "name");
        execute("SET " + name + " = " + quoteSetValue(value));
    }

    public void setThreads(int n) throws SQLException {
        execute("SET threads = " + n);
    }

    public void setMemoryLimit(String limit) throws SQLException {
        execute("SET memory_limit = '" + escapePath(limit) + "'");
    }

    public DataFrame settings() throws Exception {
        return query("SELECT * FROM duckdb_settings()");
    }

    public DataFrame extensions() throws Exception {
        return query("SELECT * FROM duckdb_extensions()");
    }

    public List<String> tables() throws SQLException {
        List<String> out = new ArrayList<>();
        try (ResultSet rs = connection.getMetaData()
                .getTables(null, null, "%", new String[]{"TABLE", "VIEW"})) {
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

    public DataFrame show(String table, int limit) throws Exception {
        return query("SELECT * FROM " + sanitizeIdent(table) + " LIMIT " + Math.max(1, limit));
    }

    /** DuckDB version string from {@code SELECT version()}. */
    public String duckdbVersion() throws Exception {
        DataFrame df = query("SELECT version() AS v");
        if (df.rowCount() == 0) return VERSION;
        Object v = df.get(0, "v");
        return v == null ? VERSION : v.toString();
    }

    // ---- transaction helpers ---------------------------------------------

    public void begin() throws SQLException {
        connection.setAutoCommit(false);
    }

    public void commit() throws SQLException {
        connection.commit();
        connection.setAutoCommit(true);
    }

    public void rollback() throws SQLException {
        try {
            connection.rollback();
        } finally {
            connection.setAutoCommit(true);
        }
    }

    public <T> T inTransaction(Function<DuckDB, T> work) throws Exception {
        boolean prev = connection.getAutoCommit();
        connection.setAutoCommit(false);
        try {
            T result = work.apply(this);
            connection.commit();
            return result;
        } catch (Exception e) {
            try { connection.rollback(); } catch (SQLException ignored) {}
            throw e;
        } finally {
            try { connection.setAutoCommit(prev); } catch (SQLException ignored) {}
        }
    }

    /** Duplicate connection (official API) for parallel readers. */
    public DuckDB duplicate() throws SQLException {
        DuckDBConnection dup = nativeConnection().duplicate();
        return wrapNative(dup, url, true);
    }

    // ---- schema helpers for Appender -------------------------------------

    public void ensureTableFromDataFrame(String table, DataFrame df, boolean replace)
            throws SQLException {
        String t = sanitizeIdent(table);
        if (replace) {
            execute("DROP TABLE IF EXISTS " + t);
        }
        if (!replace && tableExists(table)) return;
        StringBuilder ddl = new StringBuilder("CREATE TABLE IF NOT EXISTS ").append(t).append(" (");
        List<Column> cols = df.columns();
        for (int i = 0; i < cols.size(); i++) {
            if (i > 0) ddl.append(", ");
            Column c = cols.get(i);
            ddl.append(sanitizeIdent(c.name())).append(' ').append(duckType(c.dtype()));
        }
        ddl.append(')');
        execute(ddl.toString());
    }

    public boolean tableExists(String table) throws SQLException {
        try (ResultSet rs = connection.getMetaData().getTables(null, null, table, null)) {
            if (rs.next()) return true;
        }
        // case-insensitive fallback
        String lower = table.toLowerCase(Locale.ROOT);
        for (String t : tables()) {
            if (t != null && t.toLowerCase(Locale.ROOT).equals(lower)) return true;
        }
        return false;
    }

    static String duckType(Column.DType dtype) {
        if (dtype == null) return "VARCHAR";
        switch (dtype) {
            case INT32: return "INTEGER";
            case INT64: return "BIGINT";
            case FLOAT32: return "FLOAT";
            case FLOAT64: return "DOUBLE";
            case BOOLEAN: return "BOOLEAN";
            case DATE: return "DATE";
            case DATETIME: return "TIMESTAMP";
            case TIME: return "TIME";
            case BINARY:
            case IMAGE:
            case AUDIO:
            case VIDEO: return "BLOB";
            case VECTOR:
            case EMBEDDING: return "FLOAT[]";
            case LIST: return "VARCHAR[]";
            case JSON: return "JSON";
            case MAP: return "MAP(VARCHAR, VARCHAR)";
            case STRING:
            default: return "VARCHAR";
        }
    }

    // ---- interop aliases -------------------------------------------------

    public void fromDataFrame(String table, DataFrame df) throws Exception {
        replaceWithDataFrame(table, df);
    }

    public DataFrame toDataFrame(String sql) throws Exception {
        return query(sql);
    }

    public DataFrame tableToDataFrame(String table) throws Exception {
        return query("SELECT * FROM " + sanitizeIdent(table));
    }

    // ---- media catalog convenience (delegates structure; see DuckDBMultimodal)

    public void ensureMediaCatalog() throws SQLException {
        execute("""
            CREATE TABLE IF NOT EXISTS media_catalog (
              media_id   VARCHAR PRIMARY KEY,
              modality   VARCHAR NOT NULL,
              uri        VARCHAR,
              width      INTEGER,
              height     INTEGER,
              duration_ms BIGINT,
              sample_rate INTEGER,
              channels   INTEGER,
              codec      VARCHAR,
              bytes      BIGINT,
              embedding  FLOAT[],
              labels     VARCHAR[],
              meta_json  JSON,
              updated_at TIMESTAMP DEFAULT current_timestamp
            )
            """);
        execute("""
            CREATE TABLE IF NOT EXISTS media_frames (
              media_id   VARCHAR,
              frame_idx  INTEGER,
              pts_ms     BIGINT,
              uri        VARCHAR,
              embedding  FLOAT[],
              PRIMARY KEY (media_id, frame_idx)
            )
            """);
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
            @Override String copyOptions() { return "FORMAT ORC"; }
        },
        ARROW {
            @Override public String tableFunction(String path) {
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

    private static void bindParams(PreparedStatement ps, Object... params) throws SQLException {
        if (params == null) return;
        for (int i = 0; i < params.length; i++) {
            ps.setObject(i + 1, params[i]);
        }
    }

    private static String quoteSetValue(String v) {
        if (v == null) return "NULL";
        String t = v.trim();
        // numbers / booleans / percentages unquoted; memory sizes & paths quoted
        if (t.matches("(?i)true|false|null")
                || t.matches("-?\\d+(\\.\\d+)?")
                || t.matches("\\d+(\\.\\d+)?%")) {
            return t;
        }
        return "'" + t.replace("'", "''") + "'";
    }

    private static List<String> splitStatements(String script) {
        // simple split on ';' outside quotes
        List<String> parts = new ArrayList<>();
        StringBuilder cur = new StringBuilder();
        boolean inSingle = false;
        for (int i = 0; i < script.length(); i++) {
            char ch = script.charAt(i);
            if (ch == '\'') {
                inSingle = !inSingle;
                cur.append(ch);
            } else if (ch == ';' && !inSingle) {
                parts.add(cur.toString());
                cur.setLength(0);
            } else {
                cur.append(ch);
            }
        }
        if (cur.length() > 0) parts.add(cur.toString());
        return parts;
    }

    /** Export Arrow stream from a result (official {@link DuckDBResultSet#arrowExportStream}). */
    public Object arrowExportStream(String sql, Object arrowAllocator, long batchSize)
            throws SQLException {
        try (Statement st = connection.createStatement();
             ResultSet rs = st.executeQuery(sql)) {
            if (rs instanceof DuckDBResultSet) {
                return ((DuckDBResultSet) rs).arrowExportStream(arrowAllocator, batchSize);
            }
            throw new SQLException("ResultSet is not DuckDBResultSet; cannot arrowExportStream");
        }
    }
}
