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
package org.bytedeco.pytorch.utils.hudi;

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
 * Options for Apache Hudi lightweight adapter (timeline + Parquet, no Hadoop).
 *
 * <p>Primary path: local base path with {@code .hoodie/} timeline metadata and
 * partitioned / unpartitioned Parquet base files (COW). MOR full merge requires
 * optional full client — this adapter documents base-only reads for MOR.</p>
 *
 * @see <a href="https://hudi.apache.org/">Apache Hudi</a>
 */
public final class HudiOptions {

    public enum TableType {
        /** Copy-on-write: base Parquet only (default light path). */
        COPY_ON_WRITE,
        /** Merge-on-read: light path reads base files only (no log merge). */
        MERGE_ON_READ
    }

    private final String basePath;
    private final String namespaceName;
    private final String table;
    private final TableType tableType;
    private final int batchRows;
    private final int parallelism;
    private final String instantTime;
    private final String fromInstantTime;
    private final Long asOfTimeMs;
    private final PartitionFilter partitionFilter;
    private final ReplicaPolicy replicaPolicy;
    private final String[] columns;
    private final Duration idleStop;
    private final Map<String, String> properties;

    private HudiOptions(Builder b) {
        this.basePath = b.basePath;
        this.namespaceName = b.namespaceName;
        this.table = b.table;
        this.tableType = b.tableType == null ? TableType.COPY_ON_WRITE : b.tableType;
        this.batchRows = Math.max(1, b.batchRows);
        this.parallelism = Math.max(1, b.parallelism);
        this.instantTime = b.instantTime;
        this.fromInstantTime = b.fromInstantTime;
        this.asOfTimeMs = b.asOfTimeMs;
        this.partitionFilter = b.partitionFilter;
        this.replicaPolicy = b.replicaPolicy == null ? ReplicaPolicy.defaults() : b.replicaPolicy;
        this.columns = b.columns;
        this.idleStop = b.idleStop == null ? Duration.ofSeconds(30) : b.idleStop;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static HudiOptions basePath(String path) {
        return builder().basePath(path).build();
    }

    public static HudiOptions basePath(Path path) {
        return basePath(path.toAbsolutePath().toString());
    }

    public static HudiOptions fromLakeOptions(LakeOptions o) {
        Objects.requireNonNull(o, "options");
        Builder b = builder();
        if (o.warehouse() != null) {
            b.basePath(o.warehouse());
        } else if (o.uri() != null) {
            b.basePath(stripFileScheme(o.uri()));
        }
        if (o.namespaceName() != null) b.namespaceName(o.namespaceName());
        if (o.table() != null) b.table(o.table());
        b.batchRows(o.batchRows())
                .parallelism(o.parallelism())
                .asOfTimeMs(o.asOfTimeMs())
                .partitionFilter(o.partitionFilter())
                .replicaPolicy(o.replicaPolicy())
                .columns(o.columns())
                .idleStop(o.idleStop())
                .properties(o.properties());
        String tt = o.property("hoodie.table.type", null);
        if (tt != null) {
            if (tt.toUpperCase().contains("MOR") || tt.toUpperCase().contains("MERGE")) {
                b.tableType(TableType.MERGE_ON_READ);
            } else {
                b.tableType(TableType.COPY_ON_WRITE);
            }
        }
        String instant = o.property("instant_time", null);
        if (instant != null) b.instantTime(instant);
        String from = o.property("from_instant_time", null);
        if (from != null) b.fromInstantTime(from);
        return b.build();
    }

    public LakeOptions toLakeOptions() {
        return LakeOptions.builder(LakeFormat.HUDI)
                .uri(basePath)
                .warehouse(basePath)
                .namespaceName(namespaceName)
                .table(table)
                .batchRows(batchRows)
                .parallelism(parallelism)
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
        if (uri.startsWith("hudi://")) return uri.substring("hudi://".length());
        return uri;
    }

    public String basePath() { return basePath; }
    public String namespaceName() { return namespaceName; }
    public String table() { return table; }
    public TableType tableType() { return tableType; }
    public int batchRows() { return batchRows; }
    public int parallelism() { return parallelism; }
    public String instantTime() { return instantTime; }
    public String fromInstantTime() { return fromInstantTime; }
    public Long asOfTimeMs() { return asOfTimeMs; }
    public PartitionFilter partitionFilter() { return partitionFilter; }
    public ReplicaPolicy replicaPolicy() { return replicaPolicy; }
    public String[] columns() { return columns == null ? null : columns.clone(); }
    public Duration idleStop() { return idleStop; }
    public Map<String, String> properties() { return properties; }

    /**
     * Resolve physical table root: {@code basePath[/namespace]/table} or plain basePath.
     */
    public Path tablePath() {
        if (basePath == null || basePath.isBlank()) {
            throw new IllegalStateException("basePath required");
        }
        Path root = Path.of(stripFileScheme(basePath)).toAbsolutePath().normalize();
        if (table != null && !table.isBlank()) {
            if (namespaceName != null && !namespaceName.isBlank()) {
                return root.resolve(namespaceName).resolve(table);
            }
            // If basePath already points at the table (has .hoodie), use as-is when no ns
            Path hoodie = root.resolve(".hoodie");
            if (java.nio.file.Files.isDirectory(hoodie) && !root.getFileName().toString().equals(table)) {
                // base is warehouse-like
                return root.resolve(table);
            }
            if (java.nio.file.Files.isDirectory(hoodie)) {
                return root;
            }
            return root.resolve(table);
        }
        return root;
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.basePath = basePath;
        b.namespaceName = namespaceName;
        b.table = table;
        b.tableType = tableType;
        b.batchRows = batchRows;
        b.parallelism = parallelism;
        b.instantTime = instantTime;
        b.fromInstantTime = fromInstantTime;
        b.asOfTimeMs = asOfTimeMs;
        b.partitionFilter = partitionFilter;
        b.replicaPolicy = replicaPolicy;
        b.columns = columns;
        b.idleStop = idleStop;
        b.properties.putAll(properties);
        return b;
    }

    public static final class Builder {
        private String basePath;
        private String namespaceName;
        private String table;
        private TableType tableType = TableType.COPY_ON_WRITE;
        private int batchRows = 4096;
        private int parallelism = 1;
        private String instantTime;
        private String fromInstantTime;
        private Long asOfTimeMs;
        private PartitionFilter partitionFilter;
        private ReplicaPolicy replicaPolicy;
        private String[] columns;
        private Duration idleStop = Duration.ofSeconds(30);
        private final Map<String, String> properties = new LinkedHashMap<>();

        public Builder basePath(String p) { this.basePath = p; return this; }
        public Builder namespaceName(String ns) { this.namespaceName = ns; return this; }
        public Builder table(String t) { this.table = t; return this; }
        public Builder tableType(TableType t) { this.tableType = t; return this; }
        public Builder batchRows(int n) { this.batchRows = n; return this; }
        public Builder parallelism(int n) { this.parallelism = n; return this; }
        public Builder instantTime(String t) { this.instantTime = t; return this; }
        public Builder fromInstantTime(String t) { this.fromInstantTime = t; return this; }
        public Builder asOfTimeMs(Long ms) { this.asOfTimeMs = ms; return this; }
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

        public HudiOptions build() {
            return new HudiOptions(this);
        }
    }
}
