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
package org.bytedeco.pytorch.utils.doris;

import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;
import org.bytedeco.pytorch.utils.lake.ReplicaPolicy;

import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable options for Apache Doris (MySQL protocol query + HTTP Stream Load).
 *
 * <p>Public protocol references:
 * <ul>
 *   <li>JDBC: {@code jdbc:mysql://fe_host:9030/db}</li>
 *   <li>Stream Load: {@code PUT /api/{db}/{table}/_stream_load}</li>
 * </ul>
 *
 * @see <a href="https://doris.apache.org/">Apache Doris</a>
 */
public final class DorisOptions {

    public enum LoadFormat {
        JSON,
        CSV,
        PARQUET
    }

    public enum TableModel {
        DUPLICATE,
        UNIQUE,
        AGGREGATE
    }

    private final String feHost;
    private final int queryPort;
    private final int httpPort;
    private final String database;
    private final String table;
    private final String username;
    private final String password;
    private final String jdbcUrl;
    private final int fetchSize;
    private final int batchRows;
    private final int connectTimeoutMs;
    private final int socketTimeoutMs;
    private final int poolSize;
    private final long poolBorrowTimeoutMs;
    private final LoadFormat loadFormat;
    private final String columnSeparator;
    private final String lineDelimiter;
    private final boolean twoPhaseCommit;
    private final boolean partialColumns;
    private final int maxFilterRatioPercent;
    private final String labelPrefix;
    private final int replicationNum;
    private final TableModel tableModel;
    private final String[] keys;
    private final String[] distributeBy;
    private final int buckets;
    private final PartitionFilter partitionFilter;
    private final ReplicaPolicy replicaPolicy;
    private final String[] columns;
    private final String where;
    private final Duration idleStop;
    private final Map<String, String> properties;
    private final Map<String, String> streamLoadHeaders;

    private DorisOptions(Builder b) {
        this.feHost = b.feHost == null || b.feHost.isBlank() ? "127.0.0.1" : b.feHost.trim();
        this.queryPort = b.queryPort > 0 ? b.queryPort : 9030;
        this.httpPort = b.httpPort > 0 ? b.httpPort : 8030;
        this.database = b.database;
        this.table = b.table;
        this.username = b.username == null ? "root" : b.username;
        this.password = b.password == null ? "" : b.password;
        this.jdbcUrl = b.jdbcUrl;
        this.fetchSize = Math.max(0, b.fetchSize);
        this.batchRows = Math.max(1, b.batchRows);
        this.connectTimeoutMs = Math.max(0, b.connectTimeoutMs);
        this.socketTimeoutMs = Math.max(0, b.socketTimeoutMs);
        this.poolSize = Math.max(1, b.poolSize);
        this.poolBorrowTimeoutMs = Math.max(0L, b.poolBorrowTimeoutMs);
        this.loadFormat = b.loadFormat == null ? LoadFormat.JSON : b.loadFormat;
        this.columnSeparator = b.columnSeparator == null ? "\t" : b.columnSeparator;
        this.lineDelimiter = b.lineDelimiter == null ? "\n" : b.lineDelimiter;
        this.twoPhaseCommit = b.twoPhaseCommit;
        this.partialColumns = b.partialColumns;
        this.maxFilterRatioPercent = Math.max(0, Math.min(100, b.maxFilterRatioPercent));
        this.labelPrefix = b.labelPrefix == null ? "jnitorch" : b.labelPrefix;
        this.replicationNum = Math.max(1, b.replicationNum);
        this.tableModel = b.tableModel == null ? TableModel.DUPLICATE : b.tableModel;
        this.keys = b.keys;
        this.distributeBy = b.distributeBy;
        this.buckets = Math.max(1, b.buckets);
        this.partitionFilter = b.partitionFilter;
        this.replicaPolicy = b.replicaPolicy == null ? ReplicaPolicy.defaults() : b.replicaPolicy;
        this.columns = b.columns;
        this.where = b.where;
        this.idleStop = b.idleStop == null ? Duration.ofSeconds(30) : b.idleStop;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
        this.streamLoadHeaders = Collections.unmodifiableMap(new LinkedHashMap<>(b.streamLoadHeaders));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static DorisOptions of(String feHost, String database, String table) {
        return builder().feHost(feHost).database(database).table(table).build();
    }

    public static DorisOptions fromLakeOptions(LakeOptions o) {
        Objects.requireNonNull(o, "options");
        Builder b = builder();
        if (o.uri() != null) {
            applyUri(b, o.uri());
        }
        if (o.namespaceName() != null) b.database(o.namespaceName());
        if (o.table() != null) b.table(o.table());
        if (o.username() != null) b.username(o.username());
        if (o.password() != null) b.password(o.password());
        b.fetchSize(o.fetchSize())
                .batchRows(o.batchRows())
                .connectTimeoutMs(o.connectTimeoutMs())
                .socketTimeoutMs(o.socketTimeoutMs())
                .partitionFilter(o.partitionFilter())
                .replicaPolicy(o.replicaPolicy())
                .columns(o.columns())
                .idleStop(o.idleStop())
                .properties(o.properties());
        String http = o.property("http_port", null);
        if (http != null) {
            try { b.httpPort(Integer.parseInt(http)); } catch (NumberFormatException ignored) {}
        }
        String pool = o.property("pool_size", null);
        if (pool != null) {
            try { b.poolSize(Integer.parseInt(pool)); } catch (NumberFormatException ignored) {}
        }
        return b.build();
    }

    public LakeOptions toLakeOptions() {
        return LakeOptions.builder(LakeFormat.DORIS)
                .uri(jdbcUrl())
                .namespaceName(database)
                .table(table)
                .username(username)
                .password(password)
                .fetchSize(fetchSize)
                .batchRows(batchRows)
                .connectTimeoutMs(connectTimeoutMs)
                .socketTimeoutMs(socketTimeoutMs)
                .partitionFilter(partitionFilter)
                .replicaPolicy(replicaPolicy)
                .columns(columns)
                .idleStop(idleStop)
                .property("http_port", Integer.toString(httpPort))
                .property("pool_size", Integer.toString(poolSize))
                .properties(properties)
                .build();
    }

    /** Effective JDBC URL (explicit or built from FE host / query port / database). */
    public String jdbcUrl() {
        if (jdbcUrl != null && !jdbcUrl.isBlank()) return jdbcUrl;
        StringBuilder sb = new StringBuilder("jdbc:mysql://")
                .append(feHost).append(':').append(queryPort).append('/');
        if (database != null && !database.isBlank()) sb.append(database);
        sb.append("?useSSL=false&allowPublicKeyRetrieval=true&useUnicode=true&characterEncoding=utf8");
        if (connectTimeoutMs > 0) sb.append("&connectTimeout=").append(connectTimeoutMs);
        if (socketTimeoutMs > 0) sb.append("&socketTimeout=").append(socketTimeoutMs);
        return sb.toString();
    }

    /** Stream Load base URL: {@code http://fe:httpPort}. */
    public String httpBaseUrl() {
        return "http://" + feHost + ":" + httpPort;
    }

    public String streamLoadPath() {
        if (database == null || table == null) {
            throw new IllegalStateException("database and table required for stream load");
        }
        return "/api/" + database + "/" + table + "/_stream_load";
    }

    public String feHost() { return feHost; }
    public int queryPort() { return queryPort; }
    public int httpPort() { return httpPort; }
    public String database() { return database; }
    public String table() { return table; }
    public String username() { return username; }
    public String password() { return password; }
    public int fetchSize() { return fetchSize; }
    public int batchRows() { return batchRows; }
    public int connectTimeoutMs() { return connectTimeoutMs; }
    public int socketTimeoutMs() { return socketTimeoutMs; }
    public int poolSize() { return poolSize; }
    public long poolBorrowTimeoutMs() { return poolBorrowTimeoutMs; }
    public LoadFormat loadFormat() { return loadFormat; }
    public String columnSeparator() { return columnSeparator; }
    public String lineDelimiter() { return lineDelimiter; }
    public boolean twoPhaseCommit() { return twoPhaseCommit; }
    public boolean partialColumns() { return partialColumns; }
    public int maxFilterRatioPercent() { return maxFilterRatioPercent; }
    public String labelPrefix() { return labelPrefix; }
    public int replicationNum() { return replicationNum; }
    public TableModel tableModel() { return tableModel; }
    public String[] keys() { return keys == null ? null : keys.clone(); }
    public String[] distributeBy() { return distributeBy == null ? null : distributeBy.clone(); }
    public int buckets() { return buckets; }
    public PartitionFilter partitionFilter() { return partitionFilter; }
    public ReplicaPolicy replicaPolicy() { return replicaPolicy; }
    public String[] columns() { return columns == null ? null : columns.clone(); }
    public String where() { return where; }
    public Duration idleStop() { return idleStop; }
    public Map<String, String> properties() { return properties; }
    public Map<String, String> streamLoadHeaders() { return streamLoadHeaders; }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.feHost = feHost;
        b.queryPort = queryPort;
        b.httpPort = httpPort;
        b.database = database;
        b.table = table;
        b.username = username;
        b.password = password;
        b.jdbcUrl = jdbcUrl;
        b.fetchSize = fetchSize;
        b.batchRows = batchRows;
        b.connectTimeoutMs = connectTimeoutMs;
        b.socketTimeoutMs = socketTimeoutMs;
        b.poolSize = poolSize;
        b.poolBorrowTimeoutMs = poolBorrowTimeoutMs;
        b.loadFormat = loadFormat;
        b.columnSeparator = columnSeparator;
        b.lineDelimiter = lineDelimiter;
        b.twoPhaseCommit = twoPhaseCommit;
        b.partialColumns = partialColumns;
        b.maxFilterRatioPercent = maxFilterRatioPercent;
        b.labelPrefix = labelPrefix;
        b.replicationNum = replicationNum;
        b.tableModel = tableModel;
        b.keys = keys;
        b.distributeBy = distributeBy;
        b.buckets = buckets;
        b.partitionFilter = partitionFilter;
        b.replicaPolicy = replicaPolicy;
        b.columns = columns;
        b.where = where;
        b.idleStop = idleStop;
        b.properties.putAll(properties);
        b.streamLoadHeaders.putAll(streamLoadHeaders);
        return b;
    }

    /**
     * Parse {@code doris://user:pass@host:9030/db/table?http_port=8030}.
     */
    public static DorisOptions fromUri(String uri) {
        Builder b = builder();
        applyUri(b, uri);
        return b.build();
    }

    private static void applyUri(Builder b, String uri) {
        if (uri == null || uri.isBlank()) return;
        String s = uri.trim();
        if (s.startsWith("jdbc:mysql://")) {
            b.jdbcUrl(s);
            // try extract db
            int slash = s.indexOf('/', "jdbc:mysql://".length());
            if (slash > 0) {
                int q = s.indexOf('?', slash);
                String path = q > 0 ? s.substring(slash + 1, q) : s.substring(slash + 1);
                if (!path.isBlank() && !path.contains("/")) b.database(path);
            }
            int hostStart = "jdbc:mysql://".length();
            int hostEnd = s.indexOf('/', hostStart);
            if (hostEnd < 0) hostEnd = s.indexOf('?', hostStart);
            if (hostEnd < 0) hostEnd = s.length();
            String hostPort = s.substring(hostStart, hostEnd);
            int colon = hostPort.lastIndexOf(':');
            if (colon > 0) {
                b.feHost(hostPort.substring(0, colon));
                try { b.queryPort(Integer.parseInt(hostPort.substring(colon + 1))); } catch (NumberFormatException ignored) {}
            } else {
                b.feHost(hostPort);
            }
            return;
        }
        if (s.startsWith("doris://")) s = s.substring("doris://".length());
        else if (s.startsWith("mysql://")) s = s.substring("mysql://".length());

        String userInfo = null;
        String rest = s;
        int at = s.lastIndexOf('@');
        if (at >= 0) {
            userInfo = s.substring(0, at);
            rest = s.substring(at + 1);
        }
        if (userInfo != null) {
            int colon = userInfo.indexOf(':');
            if (colon >= 0) {
                b.username(userInfo.substring(0, colon));
                b.password(userInfo.substring(colon + 1));
            } else {
                b.username(userInfo);
            }
        }
        String hostPart;
        String path = null;
        String query = null;
        int q = rest.indexOf('?');
        if (q >= 0) {
            query = rest.substring(q + 1);
            rest = rest.substring(0, q);
        }
        int slash = rest.indexOf('/');
        if (slash >= 0) {
            hostPart = rest.substring(0, slash);
            path = rest.substring(slash + 1);
        } else {
            hostPart = rest;
        }
        int colon = hostPart.lastIndexOf(':');
        if (colon > 0) {
            b.feHost(hostPart.substring(0, colon));
            try { b.queryPort(Integer.parseInt(hostPart.substring(colon + 1))); } catch (NumberFormatException ignored) {}
        } else if (!hostPart.isBlank()) {
            b.feHost(hostPart);
        }
        if (path != null && !path.isBlank()) {
            String[] parts = path.split("/");
            if (parts.length >= 1 && !parts[0].isBlank()) b.database(parts[0]);
            if (parts.length >= 2 && !parts[1].isBlank()) b.table(parts[1]);
        }
        if (query != null) {
            for (String part : query.split("&")) {
                int eq = part.indexOf('=');
                if (eq <= 0) continue;
                String k = part.substring(0, eq);
                String v = part.substring(eq + 1);
                switch (k) {
                    case "http_port" -> {
                        try { b.httpPort(Integer.parseInt(v)); } catch (NumberFormatException ignored) {}
                    }
                    case "user", "username" -> b.username(v);
                    case "password", "pwd" -> b.password(v);
                    case "database", "db" -> b.database(v);
                    case "table" -> b.table(v);
                    case "pool_size" -> {
                        try { b.poolSize(Integer.parseInt(v)); } catch (NumberFormatException ignored) {}
                    }
                    default -> b.property(k, v);
                }
            }
        }
    }

    public static final class Builder {
        private String feHost = "127.0.0.1";
        private int queryPort = 9030;
        private int httpPort = 8030;
        private String database;
        private String table;
        private String username = "root";
        private String password = "";
        private String jdbcUrl;
        private int fetchSize = 2048;
        private int batchRows = 4096;
        private int connectTimeoutMs = 10_000;
        private int socketTimeoutMs = 120_000;
        private int poolSize = 8;
        private long poolBorrowTimeoutMs = 30_000L;
        private LoadFormat loadFormat = LoadFormat.JSON;
        private String columnSeparator = "\t";
        private String lineDelimiter = "\n";
        private boolean twoPhaseCommit = false;
        private boolean partialColumns = false;
        private int maxFilterRatioPercent = 0;
        private String labelPrefix = "jnitorch";
        private int replicationNum = 3;
        private TableModel tableModel = TableModel.DUPLICATE;
        private String[] keys;
        private String[] distributeBy;
        private int buckets = 10;
        private PartitionFilter partitionFilter;
        private ReplicaPolicy replicaPolicy;
        private String[] columns;
        private String where;
        private Duration idleStop = Duration.ofSeconds(30);
        private final Map<String, String> properties = new LinkedHashMap<>();
        private final Map<String, String> streamLoadHeaders = new LinkedHashMap<>();

        public Builder feHost(String h) { this.feHost = h; return this; }
        public Builder queryPort(int p) { this.queryPort = p; return this; }
        public Builder httpPort(int p) { this.httpPort = p; return this; }
        public Builder database(String d) { this.database = d; return this; }
        public Builder table(String t) { this.table = t; return this; }
        public Builder username(String u) { this.username = u; return this; }
        public Builder password(String p) { this.password = p; return this; }
        public Builder jdbcUrl(String u) { this.jdbcUrl = u; return this; }
        public Builder fetchSize(int n) { this.fetchSize = n; return this; }
        public Builder batchRows(int n) { this.batchRows = n; return this; }
        public Builder connectTimeoutMs(int ms) { this.connectTimeoutMs = ms; return this; }
        public Builder socketTimeoutMs(int ms) { this.socketTimeoutMs = ms; return this; }
        public Builder poolSize(int n) { this.poolSize = n; return this; }
        public Builder poolBorrowTimeoutMs(long ms) { this.poolBorrowTimeoutMs = ms; return this; }
        public Builder loadFormat(LoadFormat f) { this.loadFormat = f; return this; }
        public Builder columnSeparator(String s) { this.columnSeparator = s; return this; }
        public Builder lineDelimiter(String s) { this.lineDelimiter = s; return this; }
        public Builder twoPhaseCommit(boolean v) { this.twoPhaseCommit = v; return this; }
        public Builder partialColumns(boolean v) { this.partialColumns = v; return this; }
        public Builder maxFilterRatioPercent(int p) { this.maxFilterRatioPercent = p; return this; }
        public Builder labelPrefix(String p) { this.labelPrefix = p; return this; }
        public Builder replicationNum(int n) { this.replicationNum = n; return this; }
        public Builder tableModel(TableModel m) { this.tableModel = m; return this; }
        public Builder keys(String... k) { this.keys = k; return this; }
        public Builder distributeBy(String... c) { this.distributeBy = c; return this; }
        public Builder buckets(int n) { this.buckets = n; return this; }
        public Builder partitionFilter(PartitionFilter f) { this.partitionFilter = f; return this; }
        public Builder replicaPolicy(ReplicaPolicy p) { this.replicaPolicy = p; return this; }
        public Builder columns(String... c) { this.columns = c; return this; }
        public Builder where(String w) { this.where = w; return this; }
        public Builder idleStop(Duration d) { this.idleStop = d; return this; }
        public Builder property(String k, String v) {
            if (k != null && v != null) properties.put(k, v);
            return this;
        }
        public Builder properties(Map<String, String> m) {
            if (m != null) properties.putAll(m);
            return this;
        }
        public Builder streamLoadHeader(String k, String v) {
            if (k != null && v != null) streamLoadHeaders.put(k, v);
            return this;
        }

        public DorisOptions build() {
            return new DorisOptions(this);
        }
    }
}
