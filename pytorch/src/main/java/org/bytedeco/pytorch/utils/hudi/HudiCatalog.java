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

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeCapabilities;
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeSchema;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;
import org.bytedeco.pytorch.utils.lake.PartitionSpec;

import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Hudi catalog for local base paths (timeline + Parquet, no Hadoop / hudi-client).
 *
 * <p>Layout: {@code warehouse[/namespace]/table/.hoodie + data parquet}.</p>
 *
 * @see <a href="https://hudi.apache.org/">Apache Hudi</a>
 */
public final class HudiCatalog implements LakeCatalog {

    private static final Set<LakeCapabilities> CAPS = Set.copyOf(EnumSet.of(
            LakeCapabilities.COLUMN_PROJECTION,
            LakeCapabilities.PARTITION_PRUNING,
            LakeCapabilities.INCREMENTAL_SCAN,
            LakeCapabilities.HIGH_THROUGHPUT_APPEND
    ));

    private final HudiOptions options;
    private final Path warehouse;
    private final String defaultNamespace;
    private final String defaultTable;
    private final LakeMetrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public HudiCatalog(HudiOptions options) {
        this(options, null);
    }

    public HudiCatalog(HudiOptions options, LakeMetrics metrics) {
        this.options = Objects.requireNonNull(options, "options");
        this.metrics = metrics == null ? LakeMetrics.of("hudi-catalog") : metrics;
        if (options.basePath() == null || options.basePath().isBlank()) {
            throw new LakeException(LakeFormat.HUDI, "catalog", "basePath / warehouse required");
        }
        this.warehouse = Path.of(HudiOptions.stripFileScheme(options.basePath()))
                .toAbsolutePath().normalize();
        this.defaultNamespace = options.namespaceName() == null ? "" : options.namespaceName();
        this.defaultTable = options.table();
    }

    public static HudiCatalog open(HudiOptions options) {
        return new HudiCatalog(options);
    }

    public static HudiCatalog open(LakeOptions lakeOptions) {
        return open(HudiOptions.fromLakeOptions(lakeOptions));
    }

    public HudiOptions options() { return options; }
    public LakeMetrics metrics() { return metrics; }
    public Path warehouse() { return warehouse; }

    Path resolveTablePath(String namespaceName, String table) {
        ensureOpen();
        if (table == null || table.isBlank()) {
            // bound table path from options
            if (defaultTable != null && !defaultTable.isBlank()) {
                return options.tablePath();
            }
            // basePath itself is the table
            if (Files.isDirectory(warehouse.resolve(".hoodie"))) {
                return warehouse;
            }
            throw new LakeException(LakeFormat.HUDI, "catalog", "table name required");
        }
        String ns = namespaceName == null ? "" : namespaceName;
        if (ns.isBlank()) {
            Path direct = warehouse.resolve(table);
            if (Files.isDirectory(direct.resolve(".hoodie")) || Files.isDirectory(direct)) {
                return direct;
            }
            if (Files.isDirectory(warehouse.resolve(".hoodie"))
                    && warehouse.getFileName() != null
                    && warehouse.getFileName().toString().equals(table)) {
                return warehouse;
            }
            return direct;
        }
        return warehouse.resolve(ns).resolve(table);
    }

    HudiTimeline timelineFor(String namespaceName, String table) {
        return HudiTimeline.load(resolveTablePath(namespaceName, table));
    }

    LakeTable buildLakeTable(String namespaceName, String table, Path tablePath, HudiTimeline timeline) {
        LakeSchema schema = inferSchema(tablePath, timeline);
        Map<String, String> props = new java.util.LinkedHashMap<>(timeline.tableProperties());
        props.put("hoodie.table.type", options.tableType().name());
        HudiTimeline.Instant latest = timeline.latestCompleted();
        PartitionSpec spec = PartitionSpec.builder().build();
        // best-effort: detect hive partitions from first data file parent
        List<Path> files = timeline.discoverParquetFiles();
        if (!files.isEmpty()) {
            String part = HudiTimeline.partitionPathOf(tablePath, files.get(0));
            if (part != null && !part.isBlank() && part.contains("=")) {
                java.util.Set<String> keys = HudiTimeline.parseHivePartition(part).keySet();
                spec = PartitionSpec.builder().identityColumns(keys.toArray(new String[0])).build();
            }
        }
        LakeTable.Builder b = LakeTable.builder(LakeFormat.HUDI, table, schema)
                .namespaceName(namespaceName == null ? "" : namespaceName)
                .location(tablePath.toString())
                .partitionSpec(spec)
                .properties(props)
                .capabilities(
                        LakeCapabilities.COLUMN_PROJECTION,
                        LakeCapabilities.PARTITION_PRUNING,
                        LakeCapabilities.INCREMENTAL_SCAN,
                        LakeCapabilities.HIGH_THROUGHPUT_APPEND);
        if (latest != null) {
            try {
                b.currentSnapshotId(Long.parseLong(latest.instantTime().substring(0,
                        Math.min(14, latest.instantTime().length()))));
            } catch (Exception ignored) {
            }
            b.property("hoodie.latest.instant", latest.instantTime());
        }
        return b.build();
    }

    private LakeSchema inferSchema(Path tablePath, HudiTimeline timeline) {
        List<Path> files = timeline.discoverParquetFiles();
        if (files.isEmpty()) {
            return LakeSchema.builder()
                    .add("_hoodie_commit_time", Column.DType.STRING)
                    .add("id", Column.DType.STRING)
                    .build();
        }
        try {
            DataFrame sample = DataFrame.readParquet(files.get(0).toString());
            LakeSchema.Builder b = LakeSchema.builder();
            for (Column c : sample.columns()) {
                b.add(c.name(), c.dtype());
            }
            return b.build();
        } catch (Exception e) {
            return LakeSchema.builder().add("value", Column.DType.STRING).build();
        }
    }

    @Override
    public LakeFormat format() {
        return LakeFormat.HUDI;
    }

    @Override
    public Set<LakeCapabilities> capabilities() {
        return CAPS;
    }

    @Override
    public List<String> listNamespaces() {
        ensureOpen();
        if (!Files.isDirectory(warehouse)) return List.of();
        List<String> out = new ArrayList<>();
        // root-level table
        if (Files.isDirectory(warehouse.resolve(".hoodie"))) {
            out.add("");
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(warehouse)) {
            for (Path p : stream) {
                if (!Files.isDirectory(p)) continue;
                String name = p.getFileName().toString();
                if (name.startsWith(".")) continue;
                if (looksLikeNamespace(p)) out.add(name);
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.HUDI, "listNamespaces",
                    "failed to list " + warehouse, e);
        }
        return List.copyOf(out);
    }

    private boolean looksLikeNamespace(Path dir) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path child : stream) {
                if (Files.isDirectory(child.resolve(".hoodie"))) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    @Override
    public List<String> listTables(String namespaceName) {
        ensureOpen();
        Path root = (namespaceName == null || namespaceName.isBlank()) ? warehouse : warehouse.resolve(namespaceName);
        if (!Files.isDirectory(root)) return List.of();
        List<String> out = new ArrayList<>();
        if (Files.isDirectory(root.resolve(".hoodie"))) {
            out.add(root.getFileName() == null ? "table" : root.getFileName().toString());
            return List.copyOf(out);
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(root)) {
            for (Path p : stream) {
                if (Files.isDirectory(p.resolve(".hoodie")) || hasParquet(p)) {
                    out.add(p.getFileName().toString());
                }
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.HUDI, "listTables", "failed under " + root, e);
        }
        return List.copyOf(out);
    }

    private static boolean hasParquet(Path dir) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path p : stream) {
                String n = p.getFileName().toString().toLowerCase();
                if (n.endsWith(".parquet")) return true;
                if (Files.isDirectory(p) && !n.startsWith(".")) {
                    if (Files.isDirectory(p.resolve(".hoodie"))) return true;
                }
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    @Override
    public boolean tableExists(String namespaceName, String table) {
        ensureOpen();
        try {
            Path path = resolveTablePath(namespaceName, table);
            return Files.isDirectory(path.resolve(".hoodie")) || hasParquet(path);
        } catch (LakeException e) {
            return false;
        }
    }

    @Override
    public LakeTable loadTable(String namespaceName, String table) {
        Path path = resolveTablePath(namespaceName, table);
        if (!Files.isDirectory(path)) {
            throw new LakeException(LakeFormat.HUDI, "loadTable", "table path not found: " + path);
        }
        HudiTimeline timeline = HudiTimeline.load(path);
        String ns = namespaceName == null ? defaultNamespace : namespaceName;
        String name = (table == null || table.isBlank())
                ? (defaultTable != null ? defaultTable : path.getFileName().toString())
                : table;
        return buildLakeTable(ns, name, path, timeline);
    }

    @Override
    public LakeTable createTable(String namespaceName, String table, LakeSchema schema,
                                 PartitionSpec partitionSpec, Map<String, String> props) {
        ensureOpen();
        if (table == null || table.isBlank()) {
            throw new LakeException(LakeFormat.HUDI, "createTable", "table name required");
        }
        Path path = resolveTablePath(namespaceName, table);
        try {
            Files.createDirectories(path);
            Map<String, String> extra = new java.util.LinkedHashMap<>();
            if (props != null) extra.putAll(props);
            if (schema != null) {
                extra.put("jnitorch.schema.fields", String.join(",", schema.names()));
            }
            HudiTimeline.initTable(path, extra);
            // schema marker for empty tables
            if (schema != null && schema.size() > 0) {
                Path schemaFile = path.resolve(".hoodie").resolve("schema.json");
                StringBuilder sb = new StringBuilder("{\"fields\":[");
                for (int i = 0; i < schema.fields().size(); i++) {
                    LakeSchema.Field f = schema.fields().get(i);
                    if (i > 0) sb.append(',');
                    sb.append("{\"name\":\"").append(f.name())
                            .append("\",\"type\":\"").append(f.dtype()).append("\"}");
                }
                sb.append("]}");
                Files.writeString(schemaFile, sb.toString());
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.HUDI, "createTable",
                    "failed to create Hudi table at " + path, e);
        }
        return loadTable(namespaceName, table);
    }

    @Override
    public void dropTable(String namespaceName, String table, boolean ifExists) {
        ensureOpen();
        Path path = resolveTablePath(namespaceName, table);
        if (!Files.isDirectory(path)) {
            if (ifExists) return;
            throw new LakeException(LakeFormat.HUDI, "dropTable", "not found: " + path);
        }
        try {
            deleteRecursive(path);
        } catch (Exception e) {
            throw new LakeException(LakeFormat.HUDI, "dropTable", "failed at " + path, e);
        }
    }

    private static void deleteRecursive(Path root) throws Exception {
        if (!Files.exists(root)) return;
        if (Files.isDirectory(root)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(root)) {
                for (Path child : stream) deleteRecursive(child);
            }
        }
        Files.deleteIfExists(root);
    }

    @Override
    public LakeScan scan(String namespaceName, String table) {
        Path path = resolveTablePath(namespaceName, table);
        HudiTimeline timeline = HudiTimeline.load(path);
        LakeTable lt = buildLakeTable(
                namespaceName == null ? defaultNamespace : namespaceName,
                table == null || table.isBlank() ? defaultTable : table,
                path, timeline);
        return new HudiScan(this, path, timeline, lt);
    }

    @Override
    public LakeWrite write(String namespaceName, String table) {
        Path path = resolveTablePath(namespaceName, table);
        try {
            Files.createDirectories(path);
            if (!Files.isDirectory(path.resolve(".hoodie"))) {
                HudiTimeline.initTable(path, Map.of());
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.HUDI, "write", "failed to prepare " + path, e);
        }
        HudiTimeline timeline = HudiTimeline.load(path);
        LakeTable lt = buildLakeTable(
                namespaceName == null ? defaultNamespace : namespaceName,
                table == null || table.isBlank() ? defaultTable : table,
                path, timeline);
        return new HudiWrite(this, path, lt);
    }

    @Override
    public LakeStream stream(String namespaceName, String table) {
        return new HudiStream(this, namespaceName, table);
    }

    void ensureOpen() {
        if (closed.get()) {
            throw new LakeException(LakeFormat.HUDI, "catalog", "catalog closed");
        }
    }

    @Override
    public void close() {
        closed.set(true);
    }
}
