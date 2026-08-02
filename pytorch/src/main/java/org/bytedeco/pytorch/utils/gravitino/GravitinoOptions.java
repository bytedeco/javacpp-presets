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
package org.bytedeco.pytorch.utils.gravitino;

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
 * Options for Apache Gravitino REST federation client.
 *
 * <p>URI points at the Gravitino server (e.g. {@code http://localhost:8090}).
 * Full table name: {@code metalake.catalog.schema.table} or set fields separately.</p>
 *
 * @see <a href="https://gravitino.apache.org/">Apache Gravitino</a>
 */
public final class GravitinoOptions {

    private final String uri;
    private final String metalake;
    private final String catalogName;
    private final String schemaName;
    private final String table;
    private final String username;
    private final String password;
    private final String authToken;
    private final String apiPrefix;
    private final int connectTimeoutMs;
    private final int socketTimeoutMs;
    private final int batchRows;
    private final int parallelism;
    private final PartitionFilter partitionFilter;
    private final ReplicaPolicy replicaPolicy;
    private final String[] columns;
    private final Duration idleStop;
    private final Map<String, String> properties;
    /** Local mock registry path for offline tests (JSON file or dir). */
    private final String mockRegistryPath;

    private GravitinoOptions(Builder b) {
        this.uri = b.uri;
        this.metalake = b.metalake;
        this.catalogName = b.catalogName;
        this.schemaName = b.schemaName;
        this.table = b.table;
        this.username = b.username;
        this.password = b.password;
        this.authToken = b.authToken;
        this.apiPrefix = b.apiPrefix == null || b.apiPrefix.isBlank() ? "/api/metalakes" : b.apiPrefix;
        this.connectTimeoutMs = Math.max(0, b.connectTimeoutMs);
        this.socketTimeoutMs = Math.max(0, b.socketTimeoutMs);
        this.batchRows = Math.max(1, b.batchRows);
        this.parallelism = Math.max(1, b.parallelism);
        this.partitionFilter = b.partitionFilter;
        this.replicaPolicy = b.replicaPolicy == null ? ReplicaPolicy.defaults() : b.replicaPolicy;
        this.columns = b.columns;
        this.idleStop = b.idleStop == null ? Duration.ofSeconds(30) : b.idleStop;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
        this.mockRegistryPath = b.mockRegistryPath;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static GravitinoOptions of(String uri, String metalake) {
        return builder().uri(uri).metalake(metalake).build();
    }

    /**
     * Parse full name {@code metalake.catalog.schema.table} (2–4 segments).
     */
    public static GravitinoOptions fromFullName(String uri, String fullName) {
        Builder b = builder().uri(uri);
        if (fullName != null && !fullName.isBlank()) {
            String[] parts = fullName.split("\\.");
            if (parts.length >= 1) b.metalake(parts[0]);
            if (parts.length >= 2) b.catalogName(parts[1]);
            if (parts.length >= 3) b.schemaName(parts[2]);
            if (parts.length >= 4) b.table(parts[3]);
            else if (parts.length == 3) b.table(parts[2]);
        }
        return b.build();
    }

    public static GravitinoOptions fromLakeOptions(LakeOptions o) {
        Objects.requireNonNull(o, "options");
        Builder b = builder();
        if (o.uri() != null) b.uri(o.uri());
        if (o.namespaceName() != null) {
            // allow metalake.catalog.schema in namespace
            String ns = o.namespaceName();
            String[] parts = ns.split("\\.");
            if (parts.length >= 1) b.metalake(parts[0]);
            if (parts.length >= 2) b.catalogName(parts[1]);
            if (parts.length >= 3) b.schemaName(parts[2]);
        }
        if (o.table() != null) b.table(o.table());
        if (o.username() != null) b.username(o.username());
        if (o.password() != null) b.password(o.password());
        b.batchRows(o.batchRows())
                .parallelism(o.parallelism())
                .partitionFilter(o.partitionFilter())
                .replicaPolicy(o.replicaPolicy())
                .columns(o.columns())
                .idleStop(o.idleStop())
                .properties(o.properties())
                .connectTimeoutMs(o.connectTimeoutMs())
                .socketTimeoutMs(o.socketTimeoutMs());
        String token = o.property("auth_token", null);
        if (token != null) b.authToken(token);
        String prefix = o.property("api_prefix", null);
        if (prefix != null) b.apiPrefix(prefix);
        String mock = o.property("mock_registry", null);
        if (mock != null) b.mockRegistryPath(mock);
        String full = o.property("full_name", null);
        if (full != null) {
            GravitinoOptions parsed = fromFullName(o.uri(), full);
            return parsed.toBuilder()
                    .batchRows(o.batchRows())
                    .parallelism(o.parallelism())
                    .partitionFilter(o.partitionFilter())
                    .replicaPolicy(o.replicaPolicy())
                    .columns(o.columns())
                    .idleStop(o.idleStop())
                    .properties(o.properties())
                    .username(o.username())
                    .password(o.password())
                    .authToken(token)
                    .apiPrefix(prefix != null ? prefix : parsed.apiPrefix())
                    .mockRegistryPath(mock)
                    .build();
        }
        return b.build();
    }

    public LakeOptions toLakeOptions() {
        return LakeOptions.builder(LakeFormat.GRAVITINO)
                .uri(uri)
                .namespaceName(qualifiedNamespace())
                .table(table)
                .username(username)
                .password(password)
                .batchRows(batchRows)
                .parallelism(parallelism)
                .partitionFilter(partitionFilter)
                .replicaPolicy(replicaPolicy)
                .columns(columns)
                .idleStop(idleStop)
                .connectTimeoutMs(connectTimeoutMs)
                .socketTimeoutMs(socketTimeoutMs)
                .properties(properties)
                .build();
    }

    public String qualifiedNamespace() {
        StringBuilder sb = new StringBuilder();
        if (metalake != null) sb.append(metalake);
        if (catalogName != null) {
            if (sb.length() > 0) sb.append('.');
            sb.append(catalogName);
        }
        if (schemaName != null) {
            if (sb.length() > 0) sb.append('.');
            sb.append(schemaName);
        }
        return sb.toString();
    }

    public String fullName() {
        String ns = qualifiedNamespace();
        if (table == null) return ns;
        return ns.isEmpty() ? table : ns + "." + table;
    }

    public String uri() { return uri; }
    public String metalake() { return metalake; }
    public String catalogName() { return catalogName; }
    public String schemaName() { return schemaName; }
    public String table() { return table; }
    public String username() { return username; }
    public String password() { return password; }
    public String authToken() { return authToken; }
    public String apiPrefix() { return apiPrefix; }
    public int connectTimeoutMs() { return connectTimeoutMs; }
    public int socketTimeoutMs() { return socketTimeoutMs; }
    public int batchRows() { return batchRows; }
    public int parallelism() { return parallelism; }
    public PartitionFilter partitionFilter() { return partitionFilter; }
    public ReplicaPolicy replicaPolicy() { return replicaPolicy; }
    public String[] columns() { return columns == null ? null : columns.clone(); }
    public Duration idleStop() { return idleStop; }
    public Map<String, String> properties() { return properties; }
    public String mockRegistryPath() { return mockRegistryPath; }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.uri = uri;
        b.metalake = metalake;
        b.catalogName = catalogName;
        b.schemaName = schemaName;
        b.table = table;
        b.username = username;
        b.password = password;
        b.authToken = authToken;
        b.apiPrefix = apiPrefix;
        b.connectTimeoutMs = connectTimeoutMs;
        b.socketTimeoutMs = socketTimeoutMs;
        b.batchRows = batchRows;
        b.parallelism = parallelism;
        b.partitionFilter = partitionFilter;
        b.replicaPolicy = replicaPolicy;
        b.columns = columns;
        b.idleStop = idleStop;
        b.properties.putAll(properties);
        b.mockRegistryPath = mockRegistryPath;
        return b;
    }

    public static final class Builder {
        private String uri;
        private String metalake;
        private String catalogName;
        private String schemaName;
        private String table;
        private String username;
        private String password;
        private String authToken;
        private String apiPrefix = "/api/metalakes";
        private int connectTimeoutMs = 10_000;
        private int socketTimeoutMs = 60_000;
        private int batchRows = 4096;
        private int parallelism = 1;
        private PartitionFilter partitionFilter;
        private ReplicaPolicy replicaPolicy;
        private String[] columns;
        private Duration idleStop = Duration.ofSeconds(30);
        private final Map<String, String> properties = new LinkedHashMap<>();
        private String mockRegistryPath;

        public Builder uri(String u) { this.uri = u; return this; }
        public Builder metalake(String m) { this.metalake = m; return this; }
        public Builder catalogName(String c) { this.catalogName = c; return this; }
        public Builder schemaName(String s) { this.schemaName = s; return this; }
        public Builder table(String t) { this.table = t; return this; }
        public Builder username(String u) { this.username = u; return this; }
        public Builder password(String p) { this.password = p; return this; }
        public Builder authToken(String t) { this.authToken = t; return this; }
        public Builder apiPrefix(String p) { this.apiPrefix = p; return this; }
        public Builder connectTimeoutMs(int ms) { this.connectTimeoutMs = ms; return this; }
        public Builder socketTimeoutMs(int ms) { this.socketTimeoutMs = ms; return this; }
        public Builder batchRows(int n) { this.batchRows = n; return this; }
        public Builder parallelism(int n) { this.parallelism = n; return this; }
        public Builder partitionFilter(PartitionFilter f) { this.partitionFilter = f; return this; }
        public Builder replicaPolicy(ReplicaPolicy p) { this.replicaPolicy = p; return this; }
        public Builder columns(String... c) { this.columns = c; return this; }
        public Builder idleStop(Duration d) { this.idleStop = d; return this; }
        public Builder mockRegistryPath(String p) { this.mockRegistryPath = p; return this; }
        public Builder property(String k, String v) {
            if (k != null && v != null) properties.put(k, v);
            return this;
        }
        public Builder properties(Map<String, String> m) {
            if (m != null) properties.putAll(m);
            return this;
        }

        public GravitinoOptions build() {
            return new GravitinoOptions(this);
        }
    }
}
