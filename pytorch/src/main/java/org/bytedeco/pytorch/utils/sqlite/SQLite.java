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
package org.bytedeco.pytorch.utils.sqlite;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.sql.SqlOptions;
import org.bytedeco.pytorch.dataframe.sql.SqlReader;
import org.sqlite.Function;
import org.sqlite.SQLiteCommitListener;
import org.sqlite.SQLiteConnection;
import org.sqlite.SQLiteUpdateListener;
import org.sqlite.SQLiteConfig.JournalMode;
import org.sqlite.SQLiteConfig.SynchronousMode;

import java.io.Closeable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.Connection;
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

/**
 * Enterprise wrapper over official <a href="https://github.com/xerial/sqlite-jdbc">sqlite-jdbc</a>
 * ({@code org.xerial:sqlite-jdbc}) with first-class {@link DataFrame} interop.
 *
 * <p>Uses the real SDK surface:
 * <ul>
 *   <li>{@link org.sqlite.SQLiteConfig} — WAL / sync / cache / mmap via {@link SQLiteConfig}</li>
 *   <li>{@link SQLiteConnection#serialize}/{@code deserialize} — in-proc snapshot ship</li>
 *   <li>{@link Function} — Java UDFs (feature hash, bucketize)</li>
 *   <li>Update / commit listeners for cache invalidation hooks</li>
 *   <li>Online backup via {@code .backup} SQL extension commands</li>
 * </ul>
 *
 * <h2>Where SQLite fits vs DuckDB in recsys</h2>
 * <ul>
 *   <li><b>DuckDB</b> — offline OLAP, parquet scans, PIT joins, training dumps</li>
 *   <li><b>SQLite</b> — online/nearline feature KV, embedding sidecars, edge/on-device
 *       (Apple Core ML style), single-writer multi-reader WAL caches (Meta/ByteDance
 *       process-local feature mirrors)</li>
 * </ul>
 *
 * <pre>{@code
 * try (SQLite db = SQLite.open(Path.of("features.db"), SQLiteConfig.onlineFeatureCache())) {
 *     db.execute("CREATE TABLE user_feat (user_id INTEGER PRIMARY KEY, ctr REAL, emb BLOB)");
 *     db.upsert("user_feat", Map.of("user_id", 1L, "ctr", 0.12, "emb", floatsToBlob(emb)));
 *     DataFrame df = db.query("SELECT * FROM user_feat WHERE user_id = ?", 1L);
 *     db.backupTo(Path.of("features.bak.db"));
 * }
 * }</pre>
 *
 * @see SQLiteConfig
 * @see SQLitePool
 * @see SQLiteFeatureCache
 * @see SQLiteEmbeddingStore
 */
public final class SQLite implements Closeable {

    public static final String VERSION = "3.49.1.0";
    public static final String DRIVER = "org.sqlite.JDBC";
    public static final String URL_MEMORY = "jdbc:sqlite::memory:";
    public static final String URL_PREFIX = "jdbc:sqlite:";

    static {
        try {
            Class.forName(DRIVER);
        } catch (ClassNotFoundException ignored) {
            // ServiceLoader may still resolve
        }
    }

    private final Connection connection;
    private final SQLiteConnection nativeConn;
    private final String url;
    private final boolean owned;
    private final SQLiteConfig config;
    private final Map<String, String> registered = new LinkedHashMap<>();

    private SQLite(Connection connection, String url, boolean owned, SQLiteConfig config) {
        this.connection = Objects.requireNonNull(connection, "connection");
        this.url = url == null ? URL_MEMORY : url;
        this.owned = owned;
        this.config = config;
        SQLiteConnection nc = null;
        try {
            if (connection instanceof SQLiteConnection) {
                nc = (SQLiteConnection) connection;
            } else if (connection.isWrapperFor(SQLiteConnection.class)) {
                nc = connection.unwrap(SQLiteConnection.class);
            }
        } catch (SQLException ignored) {
            nc = null;
        }
        this.nativeConn = nc;
    }

    // ---- factories ---------------------------------------------------------

    public static SQLite inMemory() throws SQLException {
        return inMemory(SQLiteConfig.inMemory());
    }

    public static SQLite inMemory(SQLiteConfig config) throws SQLException {
        return open(URL_MEMORY, config);
    }

    /** Shared in-memory DB (visible across connections with same name). */
    public static SQLite sharedMemory(String name) throws SQLException {
        return sharedMemory(name, SQLiteConfig.inMemory());
    }

    public static SQLite sharedMemory(String name, SQLiteConfig config) throws SQLException {
        String n = (name == null || name.isEmpty()) ? "main" : name;
        String url = "jdbc:sqlite:file:" + n + "?mode=memory&cache=shared";
        return open(url, config);
    }

    public static SQLite open(Path dbFile) throws Exception {
        return open(dbFile, SQLiteConfig.analytics());
    }

    public static SQLite open(Path dbFile, SQLiteConfig config) throws Exception {
        Objects.requireNonNull(dbFile, "dbFile");
        if (dbFile.getParent() != null) {
            Files.createDirectories(dbFile.getParent());
        }
        return open(URL_PREFIX + dbFile.toAbsolutePath(), config);
    }

    public static SQLite openReadOnly(Path dbFile) throws Exception {
        return open(dbFile, SQLiteConfig.readOnlyServing());
    }

    public static SQLite open(String jdbcUrl) throws SQLException {
        return open(jdbcUrl, (SQLiteConfig) null);
    }

    public static SQLite open(String jdbcUrl, Properties props) throws SQLException {
        String url = normalizeUrl(jdbcUrl);
        Connection c = props == null
                ? java.sql.DriverManager.getConnection(url)
                : java.sql.DriverManager.getConnection(url, props);
        return new SQLite(c, url, true, null);
    }

    public static SQLite open(String jdbcUrl, SQLiteConfig config) throws SQLException {
        String url = normalizeUrl(jdbcUrl);
        SQLiteConfig cfg = config == null ? SQLiteConfig.create() : config;
        Connection c = cfg.nativeConfig().createConnection(url);
        // apply() is invoked by createConnection for most pragmas; add extras
        applyExtras(c, cfg);
        return new SQLite(c, url, true, cfg);
    }

    public static SQLite wrap(Connection connection) {
        return new SQLite(connection, null, false, null);
    }

    // ---- accessors ---------------------------------------------------------

    public Connection connection() {
        return connection;
    }

    public SQLiteConnection nativeConnection() throws SQLException {
        if (nativeConn != null) return nativeConn;
        throw new SQLException("Underlying connection is not org.sqlite.SQLiteConnection — "
                + "ensure org.xerial:sqlite-jdbc is on the classpath");
    }

    public boolean hasNative() {
        return nativeConn != null;
    }

    public String url() {
        return url;
    }

    public SQLiteConfig config() {
        return config;
    }

    public Map<String, String> registeredTables() {
        return Collections.unmodifiableMap(registered);
    }

    public String libVersion() throws SQLException {
        if (nativeConn != null) return nativeConn.libversion();
        DataFrame df;
        try {
            df = query("SELECT sqlite_version() AS v");
        } catch (Exception e) {
            throw new SQLException(e);
        }
        if (df.rowCount() == 0) return VERSION;
        Object v = df.get(0, "v");
        return v == null ? VERSION : v.toString();
    }

    // ---- SQL / DataFrame ---------------------------------------------------

    public DataFrame query(String sql) throws Exception {
        Objects.requireNonNull(sql, "sql");
        return SqlReader.read(connection, sql);
    }

    public DataFrame query(String sql, SqlOptions options) throws Exception {
        return SqlReader.read(connection, sql, options);
    }

    public DataFrame query(String sql, Object... params) throws Exception {
        Objects.requireNonNull(sql, "sql");
        try (PreparedStatement ps = connection.prepareStatement(sql)) {
            bindParams(ps, params);
            try (ResultSet rs = ps.executeQuery()) {
                return SqlReader.fromResultSet(rs);
            }
        }
    }

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

    public void executeScript(String script) throws SQLException {
        Objects.requireNonNull(script, "script");
        try (Statement st = connection.createStatement()) {
            for (String part : splitStatements(script)) {
                if (!part.isBlank()) st.execute(part);
            }
        }
    }

    // ---- DataFrame register / write ----------------------------------------

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

    public void writeTable(String table, DataFrame df, SqlOptions options) throws Exception {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        df.toSql(connection, table, options == null
                ? SqlOptions.builder().ifExists(SqlOptions.IfExists.REPLACE).build()
                : options);
        registered.put(table, "dataframe rows=" + df.rowCount());
    }

    public void unregister(String table) throws SQLException {
        execute("DROP TABLE IF EXISTS " + sanitizeIdent(table));
        registered.remove(table);
    }

    public DataFrame tableToDataFrame(String table) throws Exception {
        return query("SELECT * FROM " + sanitizeIdent(table));
    }

    /**
     * High-throughput batch insert using a single transaction + prepared statement.
     * Prefer over row-at-a-time for feature cache warm-up.
     */
    public long batchInsert(String table, DataFrame df) throws Exception {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(df, "df");
        List<Column> cols = df.columns();
        if (cols.isEmpty() || df.rowCount() == 0) return 0L;
        ensureTableFromDataFrame(table, df, false);
        String t = sanitizeIdent(table);
        StringBuilder sb = new StringBuilder("INSERT INTO ").append(t).append(" (");
        StringBuilder ph = new StringBuilder();
        for (int i = 0; i < cols.size(); i++) {
            if (i > 0) {
                sb.append(", ");
                ph.append(", ");
            }
            sb.append(sanitizeIdent(cols.get(i).name()));
            ph.append('?');
        }
        sb.append(") VALUES (").append(ph).append(')');
        boolean prev = connection.getAutoCommit();
        connection.setAutoCommit(false);
        long n = 0;
        try (PreparedStatement ps = connection.prepareStatement(sb.toString())) {
            final int batchSize = 500;
            for (int r = 0; r < df.rowCount(); r++) {
                for (int c = 0; c < cols.size(); c++) {
                    Object v = cols.get(c).get(r);
                    setParam(ps, c + 1, v, cols.get(c).dtype());
                }
                ps.addBatch();
                n++;
                if (n % batchSize == 0) ps.executeBatch();
            }
            ps.executeBatch();
            connection.commit();
        } catch (Exception e) {
            try { connection.rollback(); } catch (SQLException ignored) {}
            throw e;
        } finally {
            try { connection.setAutoCommit(prev); } catch (SQLException ignored) {}
        }
        registered.put(table, "batch rows+=" + n);
        return n;
    }

    /** Upsert a single row from a map (column → value). Requires PRIMARY KEY. */
    public int upsert(String table, Map<String, Object> row) throws SQLException {
        Objects.requireNonNull(table, "table");
        Objects.requireNonNull(row, "row");
        if (row.isEmpty()) return 0;
        List<String> cols = new ArrayList<>(row.keySet());
        String t = sanitizeIdent(table);
        StringBuilder sb = new StringBuilder("INSERT INTO ").append(t).append(" (");
        StringBuilder ph = new StringBuilder();
        StringBuilder upd = new StringBuilder();
        for (int i = 0; i < cols.size(); i++) {
            if (i > 0) {
                sb.append(", ");
                ph.append(", ");
                upd.append(", ");
            }
            String c = sanitizeIdent(cols.get(i));
            sb.append(c);
            ph.append('?');
            upd.append(c).append("=excluded.").append(c);
        }
        sb.append(") VALUES (").append(ph).append(") ON CONFLICT DO UPDATE SET ").append(upd);
        try (PreparedStatement ps = connection.prepareStatement(sb.toString())) {
            for (int i = 0; i < cols.size(); i++) {
                ps.setObject(i + 1, row.get(cols.get(i)));
            }
            return ps.executeUpdate();
        }
    }

    // ---- pragmas / durability ----------------------------------------------

    public void enableWalSafe() throws SQLException {
        execute("PRAGMA journal_mode=WAL");
        execute("PRAGMA synchronous=NORMAL");
    }

    public void pragma(String name, String value) throws SQLException {
        execute("PRAGMA " + name + "=" + value);
    }

    public String pragma(String name) throws Exception {
        DataFrame df = query("PRAGMA " + name);
        if (df.rowCount() == 0 || df.columnCount() == 0) return null;
        Object v = df.columns().get(0).get(0);
        return v == null ? null : v.toString();
    }

    public void setBusyTimeout(int ms) throws SQLException {
        if (nativeConn != null) nativeConn.setBusyTimeout(ms);
        else execute("PRAGMA busy_timeout=" + Math.max(0, ms));
    }

    public void setMmapSize(long bytes) throws SQLException {
        execute("PRAGMA mmap_size=" + Math.max(0L, bytes));
    }

    public void optimize() throws SQLException {
        execute("PRAGMA optimize");
    }

    public void analyze() throws SQLException {
        execute("ANALYZE");
    }

    public void vacuum() throws SQLException {
        execute("VACUUM");
    }

    public void walCheckpoint(String mode) throws SQLException {
        // mode: PASSIVE, FULL, RESTART, TRUNCATE
        String m = mode == null ? "PASSIVE" : mode.toUpperCase(Locale.ROOT);
        execute("PRAGMA wal_checkpoint(" + m + ")");
    }

    // ---- backup / serialize (official SDK) ---------------------------------

    /**
     * Online backup to another DB file using sqlite-jdbc backup command.
     * {@code backup to 'path'} is supported via ExtendedCommand.
     */
    public void backupTo(Path dest) throws SQLException {
        Objects.requireNonNull(dest, "dest");
        try {
            if (dest.getParent() != null) Files.createDirectories(dest.getParent());
        } catch (Exception e) {
            throw new SQLException("Cannot create backup parent dir: " + dest, e);
        }
        // xerial ExtendedCommand: backup to <file>
        execute("backup to '" + escapePath(dest.toAbsolutePath().toString()) + "'");
    }

    public void restoreFrom(Path src) throws SQLException {
        Objects.requireNonNull(src, "src");
        execute("restore from '" + escapePath(src.toAbsolutePath().toString()) + "'");
    }

    /** Serialize schema {@code main} (or named) to bytes — official {@link SQLiteConnection#serialize}. */
    public byte[] serialize(String schema) throws SQLException {
        String s = schema == null || schema.isBlank() ? "main" : schema;
        return nativeConnection().serialize(s);
    }

    public void deserialize(String schema, byte[] data) throws SQLException {
        Objects.requireNonNull(data, "data");
        String s = schema == null || schema.isBlank() ? "main" : schema;
        nativeConnection().deserialize(s, data);
    }

    // ---- Java UDFs (org.sqlite.Function) -----------------------------------

    /**
     * Register a deterministic scalar UDF.
     * Example: feature bucketize, murmur-like hash for categorical ids.
     */
    public void registerFunction(String name, int nArgs, Function fn) throws SQLException {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(fn, "fn");
        Function.create(connection, name, fn, nArgs, Function.FLAG_DETERMINISTIC);
    }

    public void registerFunction(String name, Function fn) throws SQLException {
        Function.create(connection, name, fn);
    }

    public void destroyFunction(String name) throws SQLException {
        Function.destroy(connection, name);
    }

    /**
     * Built-in: {@code feat_hash(text) -> int} stable 32-bit hash for categorical features.
     */
    public void registerFeatureHashUDF() throws SQLException {
        registerFunction("feat_hash", 1, new Function() {
            @Override
            protected void xFunc() throws SQLException {
                String s = value_text(0);
                if (s == null) {
                    result();
                    return;
                }
                result(stableHash(s));
            }
        });
    }

    /**
     * Built-in: {@code bucketize(value, boundaries_csv) -> int} — Google-style bucket feature.
     * {@code boundaries_csv} e.g. {@code "0,1,5,10,50"}.
     */
    public void registerBucketizeUDF() throws SQLException {
        registerFunction("bucketize", 2, new Function() {
            @Override
            protected void xFunc() throws SQLException {
                double v = value_double(0);
                String bounds = value_text(1);
                if (bounds == null || bounds.isBlank()) {
                    result(0);
                    return;
                }
                String[] parts = bounds.split(",");
                int b = 0;
                for (int i = 0; i < parts.length; i++) {
                    double edge = Double.parseDouble(parts[i].trim());
                    if (v >= edge) b = i + 1;
                    else break;
                }
                result(b);
            }
        });
    }

    // ---- listeners ---------------------------------------------------------

    public void addUpdateListener(SQLiteUpdateListener listener) throws SQLException {
        nativeConnection().addUpdateListener(listener);
    }

    public void removeUpdateListener(SQLiteUpdateListener listener) throws SQLException {
        nativeConnection().removeUpdateListener(listener);
    }

    public void addCommitListener(SQLiteCommitListener listener) throws SQLException {
        nativeConnection().addCommitListener(listener);
    }

    public void removeCommitListener(SQLiteCommitListener listener) throws SQLException {
        nativeConnection().removeCommitListener(listener);
    }

    // ---- catalog -----------------------------------------------------------

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
        return query("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
                + "AND name NOT LIKE 'sqlite_%' ORDER BY name");
    }

    public DataFrame tableInfo(String table) throws Exception {
        return query("PRAGMA table_info(" + sanitizeIdent(table) + ")");
    }

    public boolean tableExists(String table) throws SQLException {
        try (PreparedStatement ps = connection.prepareStatement(
                "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ? LIMIT 1")) {
            ps.setString(1, table);
            try (ResultSet rs = ps.executeQuery()) {
                return rs.next();
            }
        }
    }

    public void ensureTableFromDataFrame(String table, DataFrame df, boolean replace)
            throws SQLException {
        String t = sanitizeIdent(table);
        if (replace) execute("DROP TABLE IF EXISTS " + t);
        if (!replace && tableExists(table)) return;
        StringBuilder ddl = new StringBuilder("CREATE TABLE IF NOT EXISTS ").append(t).append(" (");
        List<Column> cols = df.columns();
        for (int i = 0; i < cols.size(); i++) {
            if (i > 0) ddl.append(", ");
            Column c = cols.get(i);
            ddl.append(sanitizeIdent(c.name())).append(' ').append(sqliteType(c.dtype()));
        }
        ddl.append(')');
        execute(ddl.toString());
    }

    // ---- transaction -------------------------------------------------------

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

    public <T> T inTransaction(SqlFunction<SQLite, T> work) throws Exception {
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

    // ---- float[] ↔ BLOB helpers (embedding sidecars) -----------------------

    public static byte[] floatsToBlob(float[] v) {
        if (v == null) return null;
        ByteBuffer buf = ByteBuffer.allocate(v.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float x : v) buf.putFloat(x);
        return buf.array();
    }

    public static float[] blobToFloats(byte[] blob) {
        if (blob == null || blob.length < 4) return null;
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        float[] out = new float[blob.length / 4];
        for (int i = 0; i < out.length; i++) out[i] = buf.getFloat();
        return out;
    }

    public static byte[] doublesToBlob(double[] v) {
        if (v == null) return null;
        ByteBuffer buf = ByteBuffer.allocate(v.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (double x : v) buf.putDouble(x);
        return buf.array();
    }

    public static double[] blobToDoubles(byte[] blob) {
        if (blob == null || blob.length < 8) return null;
        ByteBuffer buf = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN);
        double[] out = new double[blob.length / 8];
        for (int i = 0; i < out.length; i++) out[i] = buf.getDouble();
        return out;
    }

    // ---- close -------------------------------------------------------------

    @Override
    public void close() {
        if (owned) {
            try {
                connection.close();
            } catch (SQLException ignored) {
            }
        }
    }

    // ---- helpers -----------------------------------------------------------

    static String sqliteType(Column.DType dtype) {
        if (dtype == null) return "TEXT";
        switch (dtype) {
            case INT32:
            case INT64:
            case BOOLEAN: return "INTEGER";
            case FLOAT32:
            case FLOAT64: return "REAL";
            case BINARY:
            case VECTOR:
            case EMBEDDING:
            case IMAGE:
            case AUDIO:
            case VIDEO: return "BLOB";
            case DATE:
            case DATETIME:
            case TIME:
            case JSON:
            case STRING:
            default: return "TEXT";
        }
    }

    static String sanitizeIdent(String name) {
        if (name == null || name.isBlank()) {
            throw new IllegalArgumentException("empty identifier");
        }
        if (name.matches("[A-Za-z_][A-Za-z0-9_]*")) return name;
        return "\"" + name.replace("\"", "\"\"") + "\"";
    }

    static String escapePath(String path) {
        return path.replace("'", "''");
    }

    private static String normalizeUrl(String jdbcUrl) {
        Objects.requireNonNull(jdbcUrl, "jdbcUrl");
        if (jdbcUrl.startsWith("jdbc:")) return jdbcUrl;
        if (":memory:".equals(jdbcUrl)) return URL_MEMORY;
        return URL_PREFIX + jdbcUrl;
    }

    private static void applyExtras(Connection c, SQLiteConfig cfg) throws SQLException {
        try (Statement st = c.createStatement()) {
            if (cfg.mmapSize() >= 0) {
                st.execute("PRAGMA mmap_size=" + cfg.mmapSize());
            }
            for (Map.Entry<String, String> e : cfg.extraPragmasView().entrySet()) {
                st.execute("PRAGMA " + e.getKey() + "=" + e.getValue());
            }
        }
    }

    private static void bindParams(PreparedStatement ps, Object... params) throws SQLException {
        if (params == null) return;
        for (int i = 0; i < params.length; i++) {
            Object v = params[i];
            if (v instanceof float[]) {
                ps.setBytes(i + 1, floatsToBlob((float[]) v));
            } else if (v instanceof double[]) {
                ps.setBytes(i + 1, doublesToBlob((double[]) v));
            } else {
                ps.setObject(i + 1, v);
            }
        }
    }

    private static void setParam(PreparedStatement ps, int idx, Object v, Column.DType dtype)
            throws SQLException {
        if (v == null) {
            ps.setObject(idx, null);
            return;
        }
        if (dtype == Column.DType.VECTOR || dtype == Column.DType.EMBEDDING) {
            if (v instanceof float[]) {
                ps.setBytes(idx, floatsToBlob((float[]) v));
                return;
            }
            if (v instanceof double[]) {
                ps.setBytes(idx, doublesToBlob((double[]) v));
                return;
            }
        }
        if (v instanceof float[]) {
            ps.setBytes(idx, floatsToBlob((float[]) v));
        } else if (v instanceof double[]) {
            ps.setBytes(idx, doublesToBlob((double[]) v));
        } else if (v instanceof byte[]) {
            ps.setBytes(idx, (byte[]) v);
        } else {
            ps.setObject(idx, v);
        }
    }

    static int stableHash(String s) {
        // FNV-1a 32-bit — stable across JVMs (unlike String.hashCode undocumented changes)
        int h = 0x811c9dc5;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x01000193;
        }
        return h;
    }

    private static List<String> splitStatements(String script) {
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

    @FunctionalInterface
    public interface SqlFunction<T, R> {
        R apply(T t) throws Exception;
    }
}
