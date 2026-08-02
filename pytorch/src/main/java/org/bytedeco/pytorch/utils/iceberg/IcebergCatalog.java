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
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Iceberg catalog for local filesystem warehouses (HadoopTables-style layout, no Hadoop).
 *
 * <p>Uses {@link IcebergTable} + {@link LocalFsTableOperations} + {@link LocalFsFileIO}.
 * Scan / write / stream delegate to {@link IcebergScan}, {@link IcebergWrite},
 * {@link IcebergStream}.</p>
 *
 * <p>REST catalog mode is declared in {@link IcebergOptions.CatalogType#REST} but not
 * implemented in this lightweight path — open with {@code HADOOP_WAREHOUSE}.</p>
 *
 * @see <a href="https://iceberg.apache.org/">Apache Iceberg</a>
 */
public final class IcebergCatalog implements LakeCatalog {

    private static final Set<LakeCapabilities> CAPS = Set.copyOf(EnumSet.of(
            LakeCapabilities.COLUMN_PROJECTION,
            LakeCapabilities.PARTITION_PRUNING,
            LakeCapabilities.INCREMENTAL_SCAN,
            LakeCapabilities.HIGH_THROUGHPUT_APPEND
    ));

    private final IcebergOptions options;
    private final Path warehouse;
    private final String defaultNamespace;
    private final String defaultTable;
    private final IcebergTable tableHandle;
    private final boolean ownTableHandle;
    private final LakeMetrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);

    public IcebergCatalog(IcebergOptions options) {
        this(options, true, null);
    }

    public IcebergCatalog(IcebergOptions options, boolean ownTableHandle) {
        this(options, ownTableHandle, null);
    }

    public IcebergCatalog(IcebergOptions options, boolean ownTableHandle, LakeMetrics metrics) {
        this.options = Objects.requireNonNull(options, "options");
        this.ownTableHandle = ownTableHandle;
        this.metrics = metrics == null ? LakeMetrics.of("iceberg-catalog") : metrics;

        if (options.catalogType() == IcebergOptions.CatalogType.REST) {
            throw new LakeException(LakeFormat.ICEBERG, "catalog",
                    "REST catalog not implemented in lightweight path; use HADOOP_WAREHOUSE (local file://)");
        }
        if (options.warehouse() == null || options.warehouse().isBlank()) {
            throw new LakeException(LakeFormat.ICEBERG, "catalog", "warehouse required");
        }
        this.warehouse = Path.of(IcebergOptions.stripFileScheme(options.warehouse()))
                .toAbsolutePath().normalize();
        this.defaultNamespace = options.namespaceName() == null ? "" : options.namespaceName();
        this.defaultTable = options.table();

        if (this.defaultTable == null || this.defaultTable.isBlank()) {
            // catalog opened without a bound table — multi-table list/load still works
            this.tableHandle = null;
        } else {
            this.tableHandle = IcebergTable.load(warehouse, defaultNamespace, defaultTable, options);
        }
    }

    public static IcebergCatalog open(IcebergOptions options) {
        return new IcebergCatalog(options);
    }

    public static IcebergCatalog open(LakeOptions lakeOptions) {
        return open(IcebergOptions.fromLakeOptions(lakeOptions));
    }

    public IcebergOptions options() {
        return options;
    }

    public LakeMetrics metrics() {
        return metrics;
    }

    public Path warehouse() {
        return warehouse;
    }

    /** Bound table handle when options.table() was set; may be null for multi-table catalog. */
    public IcebergTable tableHandle() {
        ensureOpen();
        if (tableHandle == null) {
            throw new LakeException(LakeFormat.ICEBERG, "catalog",
                    "no default table bound; call loadTable(ns, name) or set options.table()");
        }
        return tableHandle;
    }

    IcebergTable requireTable(String namespaceName, String table) {
        ensureOpen();
        String ns = namespaceName == null ? "" : namespaceName;
        if (tableHandle != null
                && Objects.equals(nullToEmpty(tableHandle.namespaceName()), nullToEmpty(ns))
                && Objects.equals(tableHandle.tableName(), table)) {
            return tableHandle;
        }
        return IcebergTable.load(warehouse, ns, table, options);
    }

    private static String nullToEmpty(String s) {
        return s == null ? "" : s;
    }

    @Override
    public LakeFormat format() {
        return LakeFormat.ICEBERG;
    }

    @Override
    public Set<LakeCapabilities> capabilities() {
        return CAPS;
    }

    @Override
    public List<String> listNamespaces() {
        ensureOpen();
        if (!Files.isDirectory(warehouse)) {
            return List.of();
        }
        // Prefer explicit default namespace when bound; else scan first-level dirs that look like DBs
        List<String> out = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(warehouse)) {
            for (Path p : stream) {
                if (!Files.isDirectory(p)) continue;
                String name = p.getFileName().toString();
                if (name.startsWith(".")) continue;
                // namespace if it contains subdirs that look like tables (have metadata/)
                // or if defaultNamespace matches
                if (looksLikeNamespace(p) || name.equals(defaultNamespace)) {
                    out.add(name);
                }
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.ICEBERG, "listNamespaces",
                    "failed to list warehouse " + warehouse, e);
        }
        if (out.isEmpty() && defaultNamespace != null && !defaultNamespace.isBlank()) {
            return Collections.singletonList(defaultNamespace);
        }
        // un-namespaced tables live at warehouse root — expose empty ns
        if (hasRootTables()) {
            if (!out.contains("")) out.add(0, "");
        }
        return List.copyOf(out);
    }

    private boolean looksLikeNamespace(Path dir) {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
            for (Path child : stream) {
                if (Files.isDirectory(child.resolve("metadata"))) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    private boolean hasRootTables() {
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(warehouse)) {
            for (Path p : stream) {
                if (Files.isDirectory(p.resolve("metadata"))) return true;
            }
        } catch (Exception ignored) {
        }
        return false;
    }

    @Override
    public List<String> listTables(String namespaceName) {
        ensureOpen();
        Path root = (namespaceName == null || namespaceName.isBlank())
                ? warehouse
                : warehouse.resolve(namespaceName);
        if (!Files.isDirectory(root)) {
            return List.of();
        }
        List<String> out = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(root)) {
            for (Path p : stream) {
                if (Files.isDirectory(p.resolve("metadata"))) {
                    out.add(p.getFileName().toString());
                }
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.ICEBERG, "listTables",
                    "failed to list tables under " + root, e);
        }
        return List.copyOf(out);
    }

    @Override
    public boolean tableExists(String namespaceName, String table) {
        ensureOpen();
        if (table == null || table.isBlank()) return false;
        Path path = IcebergTable.resolveTablePath(warehouse, namespaceName, table);
        return Files.isDirectory(path.resolve("metadata"));
    }

    @Override
    public LakeTable loadTable(String namespaceName, String table) {
        IcebergTable handle = requireTable(namespaceName, table);
        return handle.lakeTable();
    }

    @Override
    public LakeTable createTable(String namespaceName, String table, LakeSchema schema,
                                 PartitionSpec partitionSpec, Map<String, String> props) {
        ensureOpen();
        if (table == null || table.isBlank()) {
            throw new LakeException(LakeFormat.ICEBERG, "createTable", "table name required");
        }
        if (schema == null) {
            throw new LakeException(LakeFormat.ICEBERG, "createTable", "schema required");
        }
        IcebergTable created = IcebergTable.create(
                warehouse, namespaceName, table, schema, partitionSpec, props);
        // if we own a default handle and names match, caller may reopen; return view
        try {
            return created.lakeTable();
        } finally {
            // created opens its own IO — close if not the bound default
            if (tableHandle == null
                    || !Objects.equals(created.tableName(), tableHandle.tableName())) {
                try {
                    created.close();
                } catch (Exception ignored) {
                }
            }
        }
    }

    @Override
    public void dropTable(String namespaceName, String table, boolean ifExists) {
        ensureOpen();
        Path path = IcebergTable.resolveTablePath(warehouse, namespaceName, table);
        if (!Files.isDirectory(path)) {
            if (ifExists) return;
            throw new LakeException(LakeFormat.ICEBERG, "dropTable", "table not found: " + path);
        }
        try {
            deleteRecursive(path);
        } catch (Exception e) {
            throw new LakeException(LakeFormat.ICEBERG, "dropTable",
                    "failed to drop table at " + path, e);
        }
    }

    private static void deleteRecursive(Path root) throws Exception {
        if (!Files.exists(root)) return;
        if (Files.isDirectory(root)) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(root)) {
                for (Path child : stream) {
                    deleteRecursive(child);
                }
            }
        }
        Files.deleteIfExists(root);
    }

    @Override
    public LakeScan scan(String namespaceName, String table) {
        return new IcebergScan(this, requireTable(namespaceName, table));
    }

    @Override
    public LakeWrite write(String namespaceName, String table) {
        return new IcebergWrite(this, requireTable(namespaceName, table));
    }

    @Override
    public LakeStream stream(String namespaceName, String table) {
        return new IcebergStream(this, requireTable(namespaceName, table));
    }

    void ensureOpen() {
        if (closed.get()) {
            throw new LakeException(LakeFormat.ICEBERG, "catalog", "catalog closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        if (ownTableHandle && tableHandle != null) {
            try {
                tableHandle.close();
            } catch (Exception ignored) {
            }
        }
    }
}
