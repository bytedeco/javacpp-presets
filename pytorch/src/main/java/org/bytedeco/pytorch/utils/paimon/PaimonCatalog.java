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
 * Paimon catalog for local warehouses (schema/snapshot + Parquet, no paimon-core).
 *
 * @see <a href="https://paimon.apache.org/">Apache Paimon</a>
 */
public final class PaimonCatalog implements LakeCatalog {

    private static final Set<LakeCapabilities> CAPS = Set.copyOf(EnumSet.of(
            LakeCapabilities.COLUMN_PROJECTION,
            LakeCapabilities.PARTITION_PRUNING,
            LakeCapabilities.INCREMENTAL_SCAN,
            LakeCapabilities.HIGH_THROUGHPUT_APPEND
    ));

    private final PaimonOptions options;
    private final Path warehouse;
    private final String defaultNamespace;
    private final String defaultTable;
    private final LakeMetrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public PaimonCatalog(PaimonOptions options) {
        this(options, null);
    }

    public PaimonCatalog(PaimonOptions options, LakeMetrics metrics) {
        this.options = Objects.requireNonNull(options, "options");
        this.metrics = metrics == null ? LakeMetrics.of("paimon-catalog") : metrics;
        if (options.warehouse() == null || options.warehouse().isBlank()) {
            throw new LakeException(LakeFormat.PAIMON, "catalog", "warehouse required");
        }
        this.warehouse = Path.of(PaimonOptions.stripFileScheme(options.warehouse()))
                .toAbsolutePath().normalize();
        this.defaultNamespace = options.namespaceName() == null ? "" : options.namespaceName();
        this.defaultTable = options.table();
    }

    public static PaimonCatalog open(PaimonOptions options) {
        return new PaimonCatalog(options);
    }

    public static PaimonCatalog open(LakeOptions lakeOptions) {
        return open(PaimonOptions.fromLakeOptions(lakeOptions));
    }

    public PaimonOptions options() { return options; }
    public LakeMetrics metrics() { return metrics; }
    public Path warehouse() { return warehouse; }

    Path resolveTablePath(String namespaceName, String table) {
        ensureOpen();
        if (table == null || table.isBlank()) {
            if (defaultTable != null && !defaultTable.isBlank()) return options.tablePath();
            if (Files.isDirectory(warehouse.resolve("schema"))
                    || Files.isDirectory(warehouse.resolve("snapshot"))) {
                return warehouse;
            }
            throw new LakeException(LakeFormat.PAIMON, "catalog", "table name required");
        }
        String ns = namespaceName == null ? "" : namespaceName;
        if (ns.isBlank()) return warehouse.resolve(table);
        return warehouse.resolve(ns).resolve(table);
    }

    LakeTable buildLakeTable(String namespaceName, String table, Path tablePath, PaimonSnapshot meta) {
        LakeSchema schema = meta.schema();
        PartitionSpec spec = PartitionSpec.builder().build();
        List<Path> files = PaimonSnapshot.discoverParquetFiles(tablePath);
        if (!files.isEmpty()) {
            String part = PaimonSnapshot.partitionPathOf(tablePath, files.get(0));
            if (part != null && part.contains("=")) {
                var keys = PaimonSnapshot.parseHivePartition(part).keySet();
                spec = PartitionSpec.builder().identityColumns(keys.toArray(new String[0])).build();
            }
        }
        LakeTable.Builder b = LakeTable.builder(LakeFormat.PAIMON, table, schema)
                .namespaceName(namespaceName == null ? "" : namespaceName)
                .location(tablePath.toString())
                .partitionSpec(spec)
                .capabilities(
                        LakeCapabilities.COLUMN_PROJECTION,
                        LakeCapabilities.PARTITION_PRUNING,
                        LakeCapabilities.INCREMENTAL_SCAN,
                        LakeCapabilities.HIGH_THROUGHPUT_APPEND);
        PaimonSnapshot.Snapshot latest = meta.latest();
        if (latest != null) {
            b.currentSnapshotId(latest.id());
            b.property("paimon.latest.snapshot", String.valueOf(latest.id()));
        }
        return b.build();
    }

    @Override
    public LakeFormat format() { return LakeFormat.PAIMON; }

    @Override
    public Set<LakeCapabilities> capabilities() { return CAPS; }

    @Override
    public List<String> listNamespaces() {
        ensureOpen();
        if (!Files.isDirectory(warehouse)) return List.of();
        List<String> out = new ArrayList<>();
        if (Files.isDirectory(warehouse.resolve("schema"))
                || Files.isDirectory(warehouse.resolve("snapshot"))) {
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
            throw new LakeException(LakeFormat.PAIMON, "listNamespaces", "failed " + warehouse, e);
        }
        return List.copyOf(out);
    }

    private boolean looksLikeNamespace(Path dir) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path child : stream) {
                if (Files.isDirectory(child.resolve("schema"))
                        || Files.isDirectory(child.resolve("snapshot"))) return true;
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
        if (Files.isDirectory(root.resolve("schema")) || Files.isDirectory(root.resolve("snapshot"))) {
            return List.of(root.getFileName() == null ? "table" : root.getFileName().toString());
        }
        List<String> out = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(root)) {
            for (Path p : stream) {
                if (Files.isDirectory(p.resolve("schema"))
                        || Files.isDirectory(p.resolve("snapshot"))
                        || hasParquet(p)) {
                    out.add(p.getFileName().toString());
                }
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.PAIMON, "listTables", "failed under " + root, e);
        }
        return List.copyOf(out);
    }

    private static boolean hasParquet(Path dir) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path p : stream) {
                String n = p.getFileName().toString().toLowerCase();
                if (n.endsWith(".parquet")) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    @Override
    public boolean tableExists(String namespaceName, String table) {
        try {
            Path path = resolveTablePath(namespaceName, table);
            return Files.isDirectory(path.resolve("schema"))
                    || Files.isDirectory(path.resolve("snapshot"))
                    || hasParquet(path);
        } catch (LakeException e) {
            return false;
        }
    }

    @Override
    public LakeTable loadTable(String namespaceName, String table) {
        Path path = resolveTablePath(namespaceName, table);
        if (!Files.isDirectory(path)) {
            throw new LakeException(LakeFormat.PAIMON, "loadTable", "not found: " + path);
        }
        PaimonSnapshot meta = PaimonSnapshot.load(path);
        String ns = namespaceName == null ? defaultNamespace : namespaceName;
        String name = (table == null || table.isBlank())
                ? (defaultTable != null ? defaultTable : path.getFileName().toString())
                : table;
        return buildLakeTable(ns, name, path, meta);
    }

    @Override
    public LakeTable createTable(String namespaceName, String table, LakeSchema schema,
                                 PartitionSpec partitionSpec, Map<String, String> props) {
        ensureOpen();
        if (table == null || table.isBlank()) {
            throw new LakeException(LakeFormat.PAIMON, "createTable", "table name required");
        }
        Path path = resolveTablePath(namespaceName, table);
        try {
            Files.createDirectories(path);
            PaimonSnapshot.initTable(path, schema);
        } catch (Exception e) {
            throw new LakeException(LakeFormat.PAIMON, "createTable", "failed at " + path, e);
        }
        return loadTable(namespaceName, table);
    }

    @Override
    public void dropTable(String namespaceName, String table, boolean ifExists) {
        ensureOpen();
        Path path = resolveTablePath(namespaceName, table);
        if (!Files.isDirectory(path)) {
            if (ifExists) return;
            throw new LakeException(LakeFormat.PAIMON, "dropTable", "not found: " + path);
        }
        try {
            deleteRecursive(path);
        } catch (Exception e) {
            throw new LakeException(LakeFormat.PAIMON, "dropTable", "failed at " + path, e);
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
        PaimonSnapshot meta = PaimonSnapshot.load(path);
        LakeTable lt = buildLakeTable(
                namespaceName == null ? defaultNamespace : namespaceName,
                table == null || table.isBlank() ? defaultTable : table,
                path, meta);
        return new PaimonScan(this, path, meta, lt);
    }

    @Override
    public LakeWrite write(String namespaceName, String table) {
        Path path = resolveTablePath(namespaceName, table);
        try {
            Files.createDirectories(path);
            if (!Files.isDirectory(path.resolve("snapshot"))) {
                PaimonSnapshot.initTable(path, LakeSchema.builder()
                        .add("value", org.bytedeco.pytorch.dataframe.Column.DType.STRING).build());
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.PAIMON, "write", "prepare failed " + path, e);
        }
        PaimonSnapshot meta = PaimonSnapshot.load(path);
        LakeTable lt = buildLakeTable(
                namespaceName == null ? defaultNamespace : namespaceName,
                table == null || table.isBlank() ? defaultTable : table,
                path, meta);
        return new PaimonWrite(this, path, lt);
    }

    @Override
    public LakeStream stream(String namespaceName, String table) {
        return new PaimonStream(this, namespaceName, table);
    }

    void ensureOpen() {
        if (closed.get()) throw new LakeException(LakeFormat.PAIMON, "catalog", "closed");
    }

    @Override
    public void close() {
        closed.set(true);
    }
}
