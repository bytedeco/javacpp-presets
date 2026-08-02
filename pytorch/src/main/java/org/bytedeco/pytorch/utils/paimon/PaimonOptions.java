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
package org.bytedeco.pytorch.utils.paimon;

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
 * Options for Apache Paimon lightweight adapter (schema/snapshot + Parquet).
 *
 * <p>Primary path: local warehouse with {@code schema/}, {@code snapshot/}, Parquet
 * data files. Full paimon-core optional (runtime scope).</p>
 *
 * @see <a href="https://paimon.apache.org/">Apache Paimon</a>
 */
public final class PaimonOptions {

    public enum SnapshotPolicy {
        LATEST,
        AS_OF_TIME,
        FROM_SNAPSHOT_ID,
        EARLIEST
    }

    private final String warehouse;
    private final String namespaceName;
    private final String table;
    private final Long snapshotId;
    private final Long asOfTimeMs;
    private final String fromSnapshotId;
    private final int batchRows;
    private final int parallelism;
    private final PartitionFilter partitionFilter;
    private final ReplicaPolicy replicaPolicy;
    private final String[] columns;
    private final Duration idleStop;
    private final Map<String, String> properties;

    private PaimonOptions(Builder b) {
        this.warehouse = b.warehouse;
        this.namespaceName = b.namespaceName;
        this.table = b.table;
        this.snapshotId = b.snapshotId;
        this.asOfTimeMs = b.asOfTimeMs;
        this.fromSnapshotId = b.fromSnapshotId;
        this.batchRows = Math.max(1, b.batchRows);
        this.parallelism = Math.max(1, b.parallelism);
        this.partitionFilter = b.partitionFilter;
        this.replicaPolicy = b.replicaPolicy == null ? ReplicaPolicy.defaults() : b.replicaPolicy;
        this.columns = b.columns;
        this.idleStop = b.idleStop == null ? Duration.ofSeconds(30) : b.idleStop;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static PaimonOptions warehouse(String path) {
        return builder().warehouse(path).build();
    }

    public static PaimonOptions warehouse(Path path) {
        return warehouse(path.toAbsolutePath().toString());
    }

    public static PaimonOptions fromLakeOptions(LakeOptions o) {
        Objects.requireNonNull(o, "options");
        Builder b = builder();
        if (o.warehouse() != null) b.warehouse(o.warehouse());
        else if (o.uri() != null) b.warehouse(stripFileScheme(o.uri()));
        if (o.namespaceName() != null) b.namespaceName(o.namespaceName());
        if (o.table() != null) b.table(o.table());
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
        if (from != null) b.fromSnapshotId(from);
        return b.build();
    }

    public LakeOptions toLakeOptions() {
        return LakeOptions.builder(LakeFormat.PAIMON)
                .uri(warehouse)
                .warehouse(warehouse)
                .namespaceName(namespaceName)
                .table(table)
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

    public String warehouse() { return warehouse; }
    public String namespaceName() { return namespaceName; }
    public String table() { return table; }
    public Long snapshotId() { return snapshotId; }
    public Long asOfTimeMs() { return asOfTimeMs; }
    public String fromSnapshotId() { return fromSnapshotId; }
    public int batchRows() { return batchRows; }
    public int parallelism() { return parallelism; }
    public PartitionFilter partitionFilter() { return partitionFilter; }
    public ReplicaPolicy replicaPolicy() { return replicaPolicy; }
    public String[] columns() { return columns == null ? null : columns.clone(); }
    public Duration idleStop() { return idleStop; }
    public Map<String, String> properties() { return properties; }

    public Path tablePath() {
        if (warehouse == null || warehouse.isBlank()) {
            throw new IllegalStateException("warehouse required");
        }
        Path root = Path.of(stripFileScheme(warehouse)).toAbsolutePath().normalize();
        if (table != null && !table.isBlank()) {
            if (namespaceName != null && !namespaceName.isBlank()) {
                return root.resolve(namespaceName).resolve(table);
            }
            return root.resolve(table);
        }
        return root;
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.warehouse = warehouse;
        b.namespaceName = namespaceName;
        b.table = table;
        b.snapshotId = snapshotId;
        b.asOfTimeMs = asOfTimeMs;
        b.fromSnapshotId = fromSnapshotId;
        b.batchRows = batchRows;
        b.parallelism = parallelism;
        b.partitionFilter = partitionFilter;
        b.replicaPolicy = replicaPolicy;
        b.columns = columns;
        b.idleStop = idleStop;
        b.properties.putAll(properties);
        return b;
    }

    public static final class Builder {
        private String warehouse;
        private String namespaceName;
        private String table;
        private Long snapshotId;
        private Long asOfTimeMs;
        private String fromSnapshotId;
        private int batchRows = 4096;
        private int parallelism = 1;
        private PartitionFilter partitionFilter;
        private ReplicaPolicy replicaPolicy;
        private String[] columns;
        private Duration idleStop = Duration.ofSeconds(30);
        private final Map<String, String> properties = new LinkedHashMap<>();

        public Builder warehouse(String w) { this.warehouse = w; return this; }
        public Builder namespaceName(String ns) { this.namespaceName = ns; return this; }
        public Builder table(String t) { this.table = t; return this; }
        public Builder snapshotId(Long id) { this.snapshotId = id; return this; }
        public Builder asOfTimeMs(Long ms) { this.asOfTimeMs = ms; return this; }
        public Builder fromSnapshotId(String id) { this.fromSnapshotId = id; return this; }
        public Builder batchRows(int n) { this.batchRows = n; return this; }
        public Builder parallelism(int n) { this.parallelism = n; return this; }
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

        public PaimonOptions build() {
            return new PaimonOptions(this);
        }
    }
}
