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
package org.bytedeco.pytorch.utils.lake;

import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Cross-engine lake connection / I/O options.
 *
 * <p>Engine-specific options (Doris FE HTTP port, Iceberg warehouse, …) live in
 * dedicated *Options classes; this type carries the common surface used by
 * {@link LakeFactory} and {@link LakeStream}.</p>
 */
public final class LakeOptions {

    private final LakeFormat format;
    private final String uri;
    private final String namespaceName;
    private final String table;
    private final String username;
    private final String password;
    private final String warehouse;
    private final int batchRows;
    private final int fetchSize;
    private final int parallelism;
    private final int connectTimeoutMs;
    private final int socketTimeoutMs;
    private final Duration idleStop;
    private final PartitionFilter partitionFilter;
    private final ReplicaPolicy replicaPolicy;
    private final String[] columns;
    private final Long snapshotId;
    private final Long asOfTimeMs;
    private final Map<String, String> properties;

    private LakeOptions(Builder b) {
        this.format = Objects.requireNonNull(b.format, "format");
        this.uri = b.uri;
        this.namespaceName = b.namespaceName;
        this.table = b.table;
        this.username = b.username;
        this.password = b.password;
        this.warehouse = b.warehouse;
        this.batchRows = Math.max(1, b.batchRows);
        this.fetchSize = Math.max(0, b.fetchSize);
        this.parallelism = Math.max(1, b.parallelism);
        this.connectTimeoutMs = Math.max(0, b.connectTimeoutMs);
        this.socketTimeoutMs = Math.max(0, b.socketTimeoutMs);
        this.idleStop = b.idleStop;
        this.partitionFilter = b.partitionFilter;
        this.replicaPolicy = b.replicaPolicy == null ? ReplicaPolicy.defaults() : b.replicaPolicy;
        this.columns = b.columns;
        this.snapshotId = b.snapshotId;
        this.asOfTimeMs = b.asOfTimeMs;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
    }

    public static Builder builder(LakeFormat format) {
        return new Builder(format);
    }

    public static LakeOptions of(LakeFormat format, String uri) {
        return builder(format).uri(uri).build();
    }

    public LakeFormat format() { return format; }
    public String uri() { return uri; }
    public String namespaceName() { return namespaceName; }
    public String table() { return table; }
    public String username() { return username; }
    public String password() { return password; }
    public String warehouse() { return warehouse; }
    public int batchRows() { return batchRows; }
    public int fetchSize() { return fetchSize; }
    public int parallelism() { return parallelism; }
    public int connectTimeoutMs() { return connectTimeoutMs; }
    public int socketTimeoutMs() { return socketTimeoutMs; }
    public Duration idleStop() { return idleStop; }
    public PartitionFilter partitionFilter() { return partitionFilter; }
    public ReplicaPolicy replicaPolicy() { return replicaPolicy; }
    public String[] columns() { return columns == null ? null : columns.clone(); }
    public Long snapshotId() { return snapshotId; }
    public Long asOfTimeMs() { return asOfTimeMs; }
    public Map<String, String> properties() { return properties; }

    public String property(String key, String defaultValue) {
        return properties.getOrDefault(key, defaultValue);
    }

    public Builder toBuilder() {
        Builder b = new Builder(format);
        b.uri = uri;
        b.namespaceName = namespaceName;
        b.table = table;
        b.username = username;
        b.password = password;
        b.warehouse = warehouse;
        b.batchRows = batchRows;
        b.fetchSize = fetchSize;
        b.parallelism = parallelism;
        b.connectTimeoutMs = connectTimeoutMs;
        b.socketTimeoutMs = socketTimeoutMs;
        b.idleStop = idleStop;
        b.partitionFilter = partitionFilter;
        b.replicaPolicy = replicaPolicy;
        b.columns = columns;
        b.snapshotId = snapshotId;
        b.asOfTimeMs = asOfTimeMs;
        b.properties.putAll(properties);
        return b;
    }

    public static final class Builder {
        private final LakeFormat format;
        private String uri;
        private String namespaceName;
        private String table;
        private String username;
        private String password;
        private String warehouse;
        private int batchRows = 4096;
        private int fetchSize = 2048;
        private int parallelism = 1;
        private int connectTimeoutMs = 10_000;
        private int socketTimeoutMs = 120_000;
        private Duration idleStop = Duration.ofSeconds(30);
        private PartitionFilter partitionFilter;
        private ReplicaPolicy replicaPolicy;
        private String[] columns;
        private Long snapshotId;
        private Long asOfTimeMs;
        private final Map<String, String> properties = new LinkedHashMap<>();

        private Builder(LakeFormat format) {
            this.format = format;
        }

        public Builder uri(String uri) { this.uri = uri; return this; }
        public Builder namespaceName(String ns) { this.namespaceName = ns; return this; }
        public Builder table(String table) { this.table = table; return this; }
        public Builder username(String u) { this.username = u; return this; }
        public Builder password(String p) { this.password = p; return this; }
        public Builder warehouse(String w) { this.warehouse = w; return this; }
        public Builder batchRows(int n) { this.batchRows = n; return this; }
        public Builder fetchSize(int n) { this.fetchSize = n; return this; }
        public Builder parallelism(int n) { this.parallelism = n; return this; }
        public Builder connectTimeoutMs(int ms) { this.connectTimeoutMs = ms; return this; }
        public Builder socketTimeoutMs(int ms) { this.socketTimeoutMs = ms; return this; }
        public Builder idleStop(Duration d) { this.idleStop = d; return this; }
        public Builder partitionFilter(PartitionFilter f) { this.partitionFilter = f; return this; }
        public Builder replicaPolicy(ReplicaPolicy p) { this.replicaPolicy = p; return this; }
        public Builder columns(String... cols) { this.columns = cols; return this; }
        public Builder snapshotId(Long id) { this.snapshotId = id; return this; }
        public Builder asOfTimeMs(Long ms) { this.asOfTimeMs = ms; return this; }
        public Builder property(String k, String v) {
            if (k != null && v != null) properties.put(k, v);
            return this;
        }
        public Builder properties(Map<String, String> m) {
            if (m != null) properties.putAll(m);
            return this;
        }

        public LakeOptions build() {
            return new LakeOptions(this);
        }
    }
}
