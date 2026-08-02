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
package org.bytedeco.pytorch.utils.iceberg;

import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;
import org.bytedeco.pytorch.utils.lake.ReplicaPolicy;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Options for Apache Iceberg (local warehouse / REST catalog, no Hadoop runtime).
 *
 * <p>Primary path: {@code file://} or plain filesystem warehouse with
 * {@code metadata/version-hint.text} + manifests + Parquet data files.</p>
 *
 * @see <a href="https://iceberg.apache.org/">Apache Iceberg</a>
 */
public final class IcebergOptions {

    public enum CatalogType {
        /** Local directory warehouse (HadoopTables-style layout without Hadoop). */
        HADOOP_WAREHOUSE,
        /** Iceberg REST catalog HTTP endpoint. */
        REST
    }

    private final CatalogType catalogType;
    private final String warehouse;
    private final String restUri;
    private final String namespaceName;
    private final String table;
    private final String username;
    private final String password;
    private final int batchRows;
    private final int parallelism;
    private final Long snapshotId;
    private final Long asOfTimeMs;
    private final Long fromSnapshotId;
    private final PartitionFilter partitionFilter;
    private final ReplicaPolicy replicaPolicy;
    private final String[] columns;
    private final Duration idleStop;
    private final Map<String, String> properties;

    private IcebergOptions(Builder b) {
        this.catalogType = b.catalogType == null ? CatalogType.HADOOP_WAREHOUSE : b.catalogType;
        this.warehouse = b.warehouse;
        this.restUri = b.restUri;
        this.namespaceName = b.namespaceName;
        this.table = b.table;
        this.username = b.username;
        this.password = b.password;
        this.batchRows = Math.max(1, b.batchRows);
        this.parallelism = Math.max(1, b.parallelism);
        this.snapshotId = b.snapshotId;
        this.asOfTimeMs = b.asOfTimeMs;
        this.fromSnapshotId = b.fromSnapshotId;
        this.partitionFilter = b.partitionFilter;
        this.replicaPolicy = b.replicaPolicy == null ? ReplicaPolicy.defaults() : b.replicaPolicy;
        this.columns = b.columns;
        this.idleStop = b.idleStop == null ? Duration.ofSeconds(30) : b.idleStop;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static IcebergOptions warehouse(String warehousePath) {
        return builder().warehouse(warehousePath).build();
    }

    public static IcebergOptions warehouse(Path warehousePath) {
        return warehouse(warehousePath.toAbsolutePath().toString());
    }

    public static IcebergOptions fromLakeOptions(LakeOptions o) {
        Objects.requireNonNull(o, "options");
        Builder b = builder();
        if (o.warehouse() != null) b.warehouse(o.warehouse());
        else if (o.uri() != null) {
            String u = o.uri();
            if (u.startsWith("rest://") || u.startsWith("http://") || u.startsWith("https://")) {
                b.catalogType(CatalogType.REST).restUri(u.replace("rest://", "http://"));
            } else {
                b.warehouse(stripFileScheme(u));
            }
        }
        if (o.namespaceName() != null) b.namespaceName(o.namespaceName());
        if (o.table() != null) b.table(o.table());
        if (o.username() != null) b.username(o.username());
        if (o.password() != null) b.password(o.password());
        b.batchRows(o.batchRows())
                .parallelism(o.parallelism())
                .snapshotId(o.snapshotId())
                .asOfTimeMs(o.asOfTimeMs())
                .partitionFilter(o.partitionFilter())
                .replicaPolicy(o.replicaPolicy())
                .columns(o.columns())
                .idleStop(o.idleStop())
                .properties(o.properties());
        String from = o.property("from_snapshot_id", null);
        if (from != null) {
            try { b.fromSnapshotId(Long.parseLong(from)); } catch (NumberFormatException ignored) {}
        }
        return b.build();
    }

    public LakeOptions toLakeOptions() {
        return LakeOptions.builder(LakeFormat.ICEBERG)
                .uri(restUri != null ? restUri : warehouse)
                .warehouse(warehouse)
                .namespaceName(namespaceName)
                .table(table)
                .username(username)
                .password(password)
                .batchRows(batchRows)
                .parallelism(parallelism)
                .snapshotId(snapshotId)
                .asOfTimeMs(asOfTimeMs)
                .partitionFilter(partitionFilter)
                .replicaPolicy(replicaPolicy)
                .columns(columns)
                .idleStop(idleStop)
                .properties(properties)
                .build();
    }

    static String stripFileScheme(String uri) {
        if (uri == null) return null;
        if (uri.startsWith("file://")) return uri.substring("file://".length());
        if (uri.startsWith("file:")) return uri.substring("file:".length());
        return uri;
    }

    public CatalogType catalogType() { return catalogType; }
    public String warehouse() { return warehouse; }
    public String restUri() { return restUri; }
    public String namespaceName() { return namespaceName; }
    public String table() { return table; }
    public String username() { return username; }
    public String password() { return password; }
    public int batchRows() { return batchRows; }
    public int parallelism() { return parallelism; }
    public Long snapshotId() { return snapshotId; }
    public Long asOfTimeMs() { return asOfTimeMs; }
    public Long fromSnapshotId() { return fromSnapshotId; }
    public PartitionFilter partitionFilter() { return partitionFilter; }
    public ReplicaPolicy replicaPolicy() { return replicaPolicy; }
    public String[] columns() { return columns == null ? null : columns.clone(); }
    public Duration idleStop() { return idleStop; }
    public Map<String, String> properties() { return properties; }

    public Path tablePath() {
        if (warehouse == null || table == null) {
            throw new IllegalStateException("warehouse and table required");
        }
        Path wh = Path.of(stripFileScheme(warehouse));
        if (namespaceName != null && !namespaceName.isBlank()) {
            return wh.resolve(namespaceName).resolve(table);
        }
        return wh.resolve(table);
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.catalogType = catalogType;
        b.warehouse = warehouse;
        b.restUri = restUri;
        b.namespaceName = namespaceName;
        b.table = table;
        b.username = username;
        b.password = password;
        b.batchRows = batchRows;
        b.parallelism = parallelism;
        b.snapshotId = snapshotId;
        b.asOfTimeMs = asOfTimeMs;
        b.fromSnapshotId = fromSnapshotId;
        b.partitionFilter = partitionFilter;
        b.replicaPolicy = replicaPolicy;
        b.columns = columns;
        b.idleStop = idleStop;
        b.properties.putAll(properties);
        return b;
    }

    public static final class Builder {
        private CatalogType catalogType = CatalogType.HADOOP_WAREHOUSE;
        private String warehouse;
        private String restUri;
        private String namespaceName;
        private String table;
        private String username;
        private String password;
        private int batchRows = 4096;
        private int parallelism = 1;
        private Long snapshotId;
        private Long asOfTimeMs;
        private Long fromSnapshotId;
        private PartitionFilter partitionFilter;
        private ReplicaPolicy replicaPolicy;
        private String[] columns;
        private Duration idleStop = Duration.ofSeconds(30);
        private final Map<String, String> properties = new LinkedHashMap<>();

        public Builder catalogType(CatalogType t) { this.catalogType = t; return this; }
        public Builder warehouse(String w) { this.warehouse = w; return this; }
        public Builder restUri(String u) { this.restUri = u; this.catalogType = CatalogType.REST; return this; }
        public Builder namespaceName(String ns) { this.namespaceName = ns; return this; }
        public Builder table(String t) { this.table = t; return this; }
        public Builder username(String u) { this.username = u; return this; }
        public Builder password(String p) { this.password = p; return this; }
        public Builder batchRows(int n) { this.batchRows = n; return this; }
        public Builder parallelism(int n) { this.parallelism = n; return this; }
        public Builder snapshotId(Long id) { this.snapshotId = id; return this; }
        public Builder asOfTimeMs(Long ms) { this.asOfTimeMs = ms; return this; }
        public Builder fromSnapshotId(Long id) { this.fromSnapshotId = id; return this; }
        public Builder partitionFilter(PartitionFilter f) { this.partitionFilter = f; return this; }
        public Builder replicaPolicy(ReplicaPolicy p) { this.replicaPolicy = p; return this; }
        public Builder columns(String... c) { this.columns = c; return this; }
        public Builder idleStop(Duration d) { this.idleStop = d; return this; }
        public Builder property(String k, String v) {
            if (k != null && v != null) properties.put(k, v);
            return this;
        }
        public Builder properties(Map<String, String> m) {
            if (m != null) properties.putAll(m);
            return this;
        }

        public IcebergOptions build() {
            return new IcebergOptions(this);
        }
    }
}
