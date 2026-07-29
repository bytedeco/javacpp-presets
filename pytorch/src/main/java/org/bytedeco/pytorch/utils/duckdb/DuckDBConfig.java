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

import org.duckdb.DuckDBDriver;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**
 * Enterprise configuration for official {@code org.duckdb:duckdb_jdbc}.
 *
 * <p>Combines JDBC connection properties ({@link DuckDBDriver} constants) with
 * session {@code SET} statements applied after open. Presets mirror how large
 * recsys / multimodal platforms size embedded OLAP engines:
 * <ul>
 *   <li>{@link #offlineFeatureEngineering()} — Meta/ByteDance offline feature
 *       join &amp; sequence materialization (many threads, large memory, temp spill)</li>
 *   <li>{@link #analytics()} — Google/Tencent ranking eval, funnel, cohort
 *       (stream results, moderate memory)</li>
 *   <li>{@link #readOnlyServing()} — Apple/edge-style read-only scans of
 *       pre-built feature / embedding tables</li>
 *   <li>{@link #etlBulkLoad()} — high-throughput Appender ingest</li>
 * </ul>
 *
 * <pre>{@code
 * DuckDBConfig cfg = DuckDBConfig.offlineFeatureEngineering()
 *     .memoryLimit("16GB")
 *     .threads(8)
 *     .tempDirectory("/tmp/duckdb")
 *     .userAgent("jnitorch-recsys/1.0");
 * try (DuckDB db = DuckDB.open(path, cfg)) { ... }
 * }</pre>
 */
public final class DuckDBConfig {

    public enum AccessMode {
        AUTOMATIC(DuckDBDriver.DUCKDB_ACCESS_MODE_AUTOMATIC),
        READ_ONLY(DuckDBDriver.DUCKDB_ACCESS_MODE_READ_ONLY),
        READ_WRITE(DuckDBDriver.DUCKDB_ACCESS_MODE_READ_WRITE);

        final String wire;
        AccessMode(String wire) { this.wire = wire; }
    }

    private final Map<String, String> jdbcProps = new LinkedHashMap<>();
    private final Map<String, String> settings = new LinkedHashMap<>();
    private boolean streamResults;
    private boolean pinDb;
    private boolean instanceCache = true;
    private Boolean autoCommit;
    private boolean jfrMemoryMonitor;
    private String userAgent = "jnitorch-duckdb";

    private DuckDBConfig() {}

    public static DuckDBConfig create() {
        return new DuckDBConfig();
    }

    /**
     * Offline feature engineering (Meta Feature Store / ByteDance lagrange-style):
     * multi-threaded, large memory, allow temp spill, writeable.
     */
    public static DuckDBConfig offlineFeatureEngineering() {
        return create()
                .accessMode(AccessMode.READ_WRITE)
                .threads(Math.max(4, Runtime.getRuntime().availableProcessors()))
                .memoryLimit("8GB")
                .preserveInsertionOrder(false)
                .enableObjectCache(true)
                .enableProgressBar(false)
                .userAgent("jnitorch-recsys-offline");
    }

    /**
     * Analytics / ranking evaluation (Google TFX eval, Tencent WeSee analytics):
     * stream results for large queries, balanced resources.
     */
    public static DuckDBConfig analytics() {
        return create()
                .accessMode(AccessMode.READ_WRITE)
                .threads(Math.max(2, Runtime.getRuntime().availableProcessors() / 2))
                .memoryLimit("4GB")
                .streamResults(true)
                .userAgent("jnitorch-analytics");
    }

    /**
     * Read-only serving / edge scan of pre-materialized tables
     * (Apple on-device style, Meta serving snapshot readers).
     */
    public static DuckDBConfig readOnlyServing() {
        return create()
                .accessMode(AccessMode.READ_ONLY)
                .readOnly(true)
                .threads(Math.max(2, Runtime.getRuntime().availableProcessors() / 2))
                .memoryLimit("2GB")
                .streamResults(true)
                .pinDb(true)
                .userAgent("jnitorch-serving-ro");
    }

    /**
     * Bulk ETL / Appender load path — insertion order off, object cache on.
     */
    public static DuckDBConfig etlBulkLoad() {
        return create()
                .accessMode(AccessMode.READ_WRITE)
                .threads(Math.max(4, Runtime.getRuntime().availableProcessors()))
                .memoryLimit("8GB")
                .preserveInsertionOrder(false)
                .enableObjectCache(true)
                .userAgent("jnitorch-etl");
    }

    /** Multimodal catalog scans (video/audio/image metadata + embeddings). */
    public static DuckDBConfig multimodalCatalog() {
        return create()
                .accessMode(AccessMode.READ_WRITE)
                .threads(Math.max(2, Runtime.getRuntime().availableProcessors() / 2))
                .memoryLimit("4GB")
                .enableObjectCache(true)
                .userAgent("jnitorch-multimodal");
    }

    // ---- JDBC properties (DuckDBDriver) ------------------------------------

    public DuckDBConfig accessMode(AccessMode mode) {
        if (mode != null) {
            jdbcProps.put(DuckDBDriver.DUCKDB_ACCESS_MODE_PROPERTY, mode.wire);
        }
        return this;
    }

    public DuckDBConfig readOnly(boolean v) {
        jdbcProps.put(DuckDBDriver.DUCKDB_READONLY_PROPERTY, Boolean.toString(v));
        return this;
    }

    public DuckDBConfig streamResults(boolean v) {
        this.streamResults = v;
        jdbcProps.put(DuckDBDriver.JDBC_STREAM_RESULTS, Boolean.toString(v));
        return this;
    }

    public DuckDBConfig pinDb(boolean v) {
        this.pinDb = v;
        jdbcProps.put(DuckDBDriver.JDBC_PIN_DB, Boolean.toString(v));
        return this;
    }

    public DuckDBConfig instanceCache(boolean v) {
        this.instanceCache = v;
        jdbcProps.put(DuckDBDriver.JDBC_INSTANCE_CACHE, Boolean.toString(v));
        return this;
    }

    public DuckDBConfig autoCommit(boolean v) {
        this.autoCommit = v;
        jdbcProps.put(DuckDBDriver.JDBC_AUTO_COMMIT, Boolean.toString(v));
        return this;
    }

    public DuckDBConfig jfrMemoryMonitor(boolean v) {
        this.jfrMemoryMonitor = v;
        jdbcProps.put(DuckDBDriver.JDBC_JFR_MEMORY_MONITOR, Boolean.toString(v));
        return this;
    }

    public DuckDBConfig ignoreUnsupportedOptions(boolean v) {
        jdbcProps.put(DuckDBDriver.JDBC_IGNORE_UNSUPPORTED_OPTIONS, Boolean.toString(v));
        return this;
    }

    public DuckDBConfig userAgent(String agent) {
        this.userAgent = agent == null || agent.isBlank() ? "jnitorch-duckdb" : agent;
        jdbcProps.put(DuckDBDriver.DUCKDB_USER_AGENT_PROPERTY, this.userAgent);
        return this;
    }

    public DuckDBConfig jdbcProperty(String key, String value) {
        Objects.requireNonNull(key, "key");
        if (value == null) jdbcProps.remove(key);
        else jdbcProps.put(key, value);
        return this;
    }

    // ---- SET settings (applied after connect) ------------------------------

    public DuckDBConfig threads(int n) {
        if (n > 0) settings.put("threads", Integer.toString(n));
        return this;
    }

    /** e.g. {@code "8GB"}, {@code "512MB"}, {@code "80%"}. */
    public DuckDBConfig memoryLimit(String limit) {
        if (limit != null && !limit.isBlank()) settings.put("memory_limit", limit.trim());
        return this;
    }

    public DuckDBConfig maxMemory(String limit) {
        return memoryLimit(limit);
    }

    public DuckDBConfig tempDirectory(String path) {
        if (path != null && !path.isBlank()) settings.put("temp_directory", path.trim());
        return this;
    }

    public DuckDBConfig maxTempDirectorySize(String size) {
        if (size != null && !size.isBlank()) settings.put("max_temp_directory_size", size.trim());
        return this;
    }

    public DuckDBConfig preserveInsertionOrder(boolean v) {
        settings.put("preserve_insertion_order", Boolean.toString(v));
        return this;
    }

    public DuckDBConfig enableObjectCache(boolean v) {
        settings.put("enable_object_cache", Boolean.toString(v));
        return this;
    }

    public DuckDBConfig enableProgressBar(boolean v) {
        settings.put("enable_progress_bar", Boolean.toString(v));
        return this;
    }

    public DuckDBConfig enableExternalAccess(boolean v) {
        settings.put("enable_external_access", Boolean.toString(v));
        return this;
    }

    public DuckDBConfig defaultNullOrder(String order) {
        if (order != null && !order.isBlank()) {
            settings.put("default_null_order", order.trim());
        }
        return this;
    }

    public DuckDBConfig defaultOrder(String order) {
        if (order != null && !order.isBlank()) {
            settings.put("default_order", order.trim().toLowerCase(Locale.ROOT));
        }
        return this;
    }

    /** Arbitrary DuckDB setting name/value (validated by engine at apply time). */
    public DuckDBConfig set(String name, String value) {
        Objects.requireNonNull(name, "name");
        if (value == null) settings.remove(name);
        else settings.put(name, value);
        return this;
    }

    public DuckDBConfig set(String name, long value) {
        return set(name, Long.toString(value));
    }

    public DuckDBConfig set(String name, boolean value) {
        return set(name, Boolean.toString(value));
    }

    // ---- materialize -------------------------------------------------------

    /** JDBC {@link Properties} for {@code DriverManager.getConnection}. */
    public Properties toJdbcProperties() {
        Properties p = new Properties();
        for (Map.Entry<String, String> e : jdbcProps.entrySet()) {
            p.setProperty(e.getKey(), e.getValue());
        }
        if (!jdbcProps.containsKey(DuckDBDriver.DUCKDB_USER_AGENT_PROPERTY)) {
            p.setProperty(DuckDBDriver.DUCKDB_USER_AGENT_PROPERTY, userAgent);
        }
        return p;
    }

    /** Apply all {@code SET} statements on an open connection. */
    public void apply(Connection connection) throws SQLException {
        Objects.requireNonNull(connection, "connection");
        if (settings.isEmpty()) return;
        try (Statement st = connection.createStatement()) {
            for (Map.Entry<String, String> e : settings.entrySet()) {
                st.execute("SET " + e.getKey() + " = " + quoteSetting(e.getValue()));
            }
        }
    }

    public Map<String, String> settingsView() {
        return Map.copyOf(settings);
    }

    public Map<String, String> jdbcPropsView() {
        return Map.copyOf(jdbcProps);
    }

    public boolean streamResults() { return streamResults; }
    public boolean pinDb() { return pinDb; }
    public boolean instanceCache() { return instanceCache; }
    public Boolean autoCommit() { return autoCommit; }
    public boolean jfrMemoryMonitor() { return jfrMemoryMonitor; }
    public String userAgent() { return userAgent; }

    private static String quoteSetting(String v) {
        if (v == null) return "NULL";
        String t = v.trim();
        // bare numbers / booleans / percentages stay unquoted;
        // memory sizes (512MB) and paths must be string-quoted for SET.
        if (t.matches("(?i)true|false|null")
                || t.matches("-?\\d+(\\.\\d+)?")
                || t.matches("\\d+(\\.\\d+)?%")) {
            return t;
        }
        return "'" + t.replace("'", "''") + "'";
    }

    @Override
    public String toString() {
        return "DuckDBConfig{jdbc=" + jdbcProps + ", settings=" + settings
                + ", userAgent=" + userAgent + "}";
    }
}
