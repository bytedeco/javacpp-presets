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
package org.bytedeco.pytorch.utils.daft;

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
 * Options for Daft-style Arrow/Parquet bridge (no Daft Rust kernel).
 *
 * <p>Path may be a single Parquet/Arrow file, a directory of Parquet files,
 * or a Hive-style partitioned dataset root.</p>
 *
 * @see <a href="https://www.getdaft.io/">Daft</a>
 */
public final class DaftOptions {

    public enum IoFormat {
        AUTO,
        PARQUET,
        ARROW_IPC
    }

    private final String path;
    private final String namespaceName;
    private final String table;
    private final IoFormat ioFormat;
    private final int batchRows;
    private final int parallelism;
    private final PartitionFilter partitionFilter;
    private final ReplicaPolicy replicaPolicy;
    private final String[] columns;
    private final Duration idleStop;
    private final String filterExpression;
    private final Long limitRows;
    private final Map<String, String> properties;

    private DaftOptions(Builder b) {
        this.path = b.path;
        this.namespaceName = b.namespaceName;
        this.table = b.table;
        this.ioFormat = b.ioFormat == null ? IoFormat.AUTO : b.ioFormat;
        this.batchRows = Math.max(1, b.batchRows);
        this.parallelism = Math.max(1, b.parallelism);
        this.partitionFilter = b.partitionFilter;
        this.replicaPolicy = b.replicaPolicy == null ? ReplicaPolicy.defaults() : b.replicaPolicy;
        this.columns = b.columns;
        this.idleStop = b.idleStop == null ? Duration.ofSeconds(30) : b.idleStop;
        this.filterExpression = b.filterExpression;
        this.limitRows = b.limitRows;
        this.properties = Collections.unmodifiableMap(new LinkedHashMap<>(b.properties));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static DaftOptions path(String path) {
        return builder().path(path).build();
    }

    public static DaftOptions path(Path path) {
        return path(path.toAbsolutePath().toString());
    }

    public static DaftOptions fromLakeOptions(LakeOptions o) {
        Objects.requireNonNull(o, "options");
        Builder b = builder();
        if (o.warehouse() != null) b.path(o.warehouse());
        else if (o.uri() != null) b.path(stripFileScheme(o.uri()));
        if (o.namespaceName() != null) b.namespaceName(o.namespaceName());
        if (o.table() != null) b.table(o.table());
        b.batchRows(o.batchRows())
                .parallelism(o.parallelism())
                .partitionFilter(o.partitionFilter())
                .replicaPolicy(o.replicaPolicy())
                .columns(o.columns())
                .idleStop(o.idleStop())
                .properties(o.properties());
        String fmt = o.property("io_format", null);
        if (fmt != null) {
            try {
                b.ioFormat(IoFormat.valueOf(fmt.toUpperCase().replace('-', '_')));
            } catch (Exception ignored) {
            }
        }
        String filter = o.property("filter", null);
        if (filter != null) b.filterExpression(filter);
        return b.build();
    }

    public LakeOptions toLakeOptions() {
        return LakeOptions.builder(LakeFormat.DAFT)
                .uri(path)
                .warehouse(path)
                .namespaceName(namespaceName)
                .table(table)
                .batchRows(batchRows)
                .parallelism(parallelism)
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
        if (uri.startsWith("daft://")) return uri.substring("daft://".length());
        return uri;
    }

    public String path() { return path; }
    public String namespaceName() { return namespaceName; }
    public String table() { return table; }
    public IoFormat ioFormat() { return ioFormat; }
    public int batchRows() { return batchRows; }
    public int parallelism() { return parallelism; }
    public PartitionFilter partitionFilter() { return partitionFilter; }
    public ReplicaPolicy replicaPolicy() { return replicaPolicy; }
    public String[] columns() { return columns == null ? null : columns.clone(); }
    public Duration idleStop() { return idleStop; }
    public String filterExpression() { return filterExpression; }
    public Long limitRows() { return limitRows; }
    public Map<String, String> properties() { return properties; }

    public Path resolvedPath() {
        if (path == null || path.isBlank()) {
            throw new IllegalStateException("path required");
        }
        Path root = Path.of(stripFileScheme(path)).toAbsolutePath().normalize();
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
        b.path = path;
        b.namespaceName = namespaceName;
        b.table = table;
        b.ioFormat = ioFormat;
        b.batchRows = batchRows;
        b.parallelism = parallelism;
        b.partitionFilter = partitionFilter;
        b.replicaPolicy = replicaPolicy;
        b.columns = columns;
        b.idleStop = idleStop;
        b.filterExpression = filterExpression;
        b.limitRows = limitRows;
        b.properties.putAll(properties);
        return b;
    }

    public static final class Builder {
        private String path;
        private String namespaceName;
        private String table;
        private IoFormat ioFormat = IoFormat.AUTO;
        private int batchRows = 4096;
        private int parallelism = 1;
        private PartitionFilter partitionFilter;
        private ReplicaPolicy replicaPolicy;
        private String[] columns;
        private Duration idleStop = Duration.ofSeconds(30);
        private String filterExpression;
        private Long limitRows;
        private final Map<String, String> properties = new LinkedHashMap<>();

        public Builder path(String p) { this.path = p; return this; }
        public Builder namespaceName(String ns) { this.namespaceName = ns; return this; }
        public Builder table(String t) { this.table = t; return this; }
        public Builder ioFormat(IoFormat f) { this.ioFormat = f; return this; }
        public Builder batchRows(int n) { this.batchRows = n; return this; }
        public Builder parallelism(int n) { this.parallelism = n; return this; }
        public Builder partitionFilter(PartitionFilter f) { this.partitionFilter = f; return this; }
        public Builder replicaPolicy(ReplicaPolicy p) { this.replicaPolicy = p; return this; }
        public Builder columns(String... c) { this.columns = c; return this; }
        public Builder idleStop(Duration d) { this.idleStop = d; return this; }
        public Builder filterExpression(String e) { this.filterExpression = e; return this; }
        public Builder limitRows(Long n) { this.limitRows = n; return this; }
        public Builder property(String k, String v) {
            if (k != null && v != null) properties.put(k, v);
            return this;
        }
        public Builder properties(Map<String, String> m) {
            if (m != null) properties.putAll(m);
            return this;
        }

        public DaftOptions build() {
            return new DaftOptions(this);
        }
    }
}
