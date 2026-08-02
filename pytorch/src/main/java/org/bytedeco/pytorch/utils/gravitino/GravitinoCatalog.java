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

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.utils.lake.LakeCapabilities;
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFactory;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeSchema;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;
import org.bytedeco.pytorch.utils.lake.PartitionSpec;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Gravitino federated {@link LakeCatalog}: resolve table metadata via REST/mock,
 * then delegate scan/write/stream to the concrete backend catalog.
 *
 * <p>Does not embed a Gravitino server — client only.</p>
 *
 * @see <a href="https://gravitino.apache.org/">Apache Gravitino</a>
 */
public final class GravitinoCatalog implements LakeCatalog {

    private static final Set<LakeCapabilities> CAPS = Set.copyOf(EnumSet.of(
            LakeCapabilities.REST_CATALOG,
            LakeCapabilities.COLUMN_PROJECTION,
            LakeCapabilities.PARTITION_PRUNING
    ));

    private final GravitinoOptions options;
    private final GravitinoMetalake metalake;
    private final LakeMetrics metrics;
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final List<LakeCatalog> openedBackends = new ArrayList<>();

    public GravitinoCatalog(GravitinoOptions options) {
        this.options = Objects.requireNonNull(options, "options");
        this.metalake = new GravitinoMetalake(options);
        this.metrics = LakeMetrics.of("gravitino-catalog");
    }

    public static GravitinoCatalog open(GravitinoOptions options) {
        return new GravitinoCatalog(options);
    }

    public static GravitinoCatalog open(LakeOptions lakeOptions) {
        return open(GravitinoOptions.fromLakeOptions(lakeOptions));
    }

    public GravitinoOptions options() { return options; }
    public GravitinoMetalake metalake() { return metalake; }
    public LakeMetrics metrics() { return metrics; }

    @Override
    public LakeFormat format() {
        return LakeFormat.GRAVITINO;
    }

    @Override
    public Set<LakeCapabilities> capabilities() {
        return CAPS;
    }

    @Override
    public List<String> listNamespaces() {
        ensureOpen();
        // namespaces ≈ catalogs (or catalog.schema)
        String cat = options.catalogName();
        if (cat != null && !cat.isBlank()) {
            return metalake.listSchemas(cat);
        }
        return metalake.listCatalogs();
    }

    @Override
    public List<String> listTables(String namespaceName) {
        ensureOpen();
        String catalog = options.catalogName();
        String schema = namespaceName;
        if (namespaceName != null && namespaceName.contains(".")) {
            String[] p = namespaceName.split("\\.", 2);
            catalog = p[0];
            schema = p[1];
        }
        if (catalog == null || catalog.isBlank()) {
            throw new LakeException(LakeFormat.GRAVITINO, "listTables",
                    "catalogName required (set options.catalogName or namespace as catalog.schema)");
        }
        if (schema == null || schema.isBlank()) {
            schema = options.schemaName();
        }
        if (schema == null || schema.isBlank()) {
            throw new LakeException(LakeFormat.GRAVITINO, "listTables", "schema required");
        }
        return metalake.listTables(catalog, schema);
    }

    @Override
    public boolean tableExists(String namespaceName, String table) {
        try {
            loadTable(namespaceName, table);
            return true;
        } catch (LakeException e) {
            return false;
        }
    }

    @Override
    public LakeTable loadTable(String namespaceName, String table) {
        ensureOpen();
        Names n = resolveNames(namespaceName, table);
        GravitinoResolver.Resolved resolved = metalake.resolveTable(n.catalog, n.schema, n.table);
        LakeSchema schema = LakeSchema.builder()
                .add("_gravitino_resolved", Column.DType.STRING)
                .build();
        // Prefer backend schema when backend can load quickly — best-effort
        try (LakeCatalog backend = openBackend(resolved)) {
            LakeTable bt = backend.loadTable(resolved.options().namespaceName(), resolved.options().table());
            schema = bt.schema();
            return LakeTable.builder(LakeFormat.GRAVITINO, n.table, schema)
                    .namespaceName(n.namespace())
                    .location(resolved.location())
                    .partitionSpec(bt.partitionSpec() == null
                            ? PartitionSpec.builder().build()
                            : bt.partitionSpec())
                    .properties(resolved.options().properties())
                    .property("gravitino.provider", resolved.provider() == null ? "" : resolved.provider())
                    .property("gravitino.backend", resolved.format().name())
                    .capabilities(LakeCapabilities.REST_CATALOG,
                            LakeCapabilities.COLUMN_PROJECTION,
                            LakeCapabilities.PARTITION_PRUNING)
                    .build();
        } catch (Exception e) {
            return LakeTable.builder(LakeFormat.GRAVITINO, n.table, schema)
                    .namespaceName(n.namespace())
                    .location(resolved.location())
                    .partitionSpec(PartitionSpec.builder().build())
                    .properties(resolved.options().properties())
                    .property("gravitino.provider", resolved.provider() == null ? "" : resolved.provider())
                    .property("gravitino.backend", resolved.format().name())
                    .capabilities(LakeCapabilities.REST_CATALOG)
                    .build();
        }
    }

    @Override
    public LakeTable createTable(String namespaceName, String table, LakeSchema schema,
                                 PartitionSpec partitionSpec, Map<String, String> props) {
        throw new LakeException(LakeFormat.GRAVITINO, "createTable",
                "DDL via Gravitino REST not implemented in client light path; "
                        + "create on backend then register in Gravitino");
    }

    @Override
    public void dropTable(String namespaceName, String table, boolean ifExists) {
        throw new LakeException(LakeFormat.GRAVITINO, "dropTable",
                "drop via Gravitino REST not implemented in client light path");
    }

    @Override
    public LakeScan scan(String namespaceName, String table) {
        ensureOpen();
        Names n = resolveNames(namespaceName, table);
        GravitinoResolver.Resolved resolved = metalake.resolveTable(n.catalog, n.schema, n.table);
        LakeCatalog backend = openBackend(resolved);
        return backend.scan(resolved.options().namespaceName(), resolved.options().table());
    }

    @Override
    public LakeWrite write(String namespaceName, String table) {
        ensureOpen();
        Names n = resolveNames(namespaceName, table);
        GravitinoResolver.Resolved resolved = metalake.resolveTable(n.catalog, n.schema, n.table);
        LakeCatalog backend = openBackend(resolved);
        return backend.write(resolved.options().namespaceName(), resolved.options().table());
    }

    @Override
    public LakeStream stream(String namespaceName, String table) {
        ensureOpen();
        Names n = resolveNames(namespaceName, table);
        GravitinoResolver.Resolved resolved = metalake.resolveTable(n.catalog, n.schema, n.table);
        LakeCatalog backend = openBackend(resolved);
        return backend.stream(resolved.options().namespaceName(), resolved.options().table());
    }

    /**
     * Resolve and return backend options without opening (for diagnostics).
     */
    public GravitinoResolver.Resolved resolve(String namespaceName, String table) {
        Names n = resolveNames(namespaceName, table);
        return metalake.resolveTable(n.catalog, n.schema, n.table);
    }

    private LakeCatalog openBackend(GravitinoResolver.Resolved resolved) {
        if (resolved.format() == LakeFormat.GRAVITINO) {
            throw new LakeException(LakeFormat.GRAVITINO, "resolve",
                    "backend resolved to GRAVITINO (circular); check provider/location properties");
        }
        LakeCatalog cat = LakeFactory.open(resolved.options());
        openedBackends.add(cat);
        return cat;
    }

    private Names resolveNames(String namespaceName, String table) {
        String catalog = options.catalogName();
        String schema = options.schemaName();
        String tbl = table != null ? table : options.table();
        if (namespaceName != null && !namespaceName.isBlank()) {
            String[] p = namespaceName.split("\\.");
            if (p.length >= 2) {
                catalog = p[0];
                schema = p[1];
            } else if (p.length == 1) {
                // single segment: treat as schema if catalog set, else catalog
                if (catalog != null && !catalog.isBlank()) schema = p[0];
                else catalog = p[0];
            }
        }
        if (catalog == null || catalog.isBlank()) {
            throw new LakeException(LakeFormat.GRAVITINO, "names",
                    "catalogName required");
        }
        if (schema == null || schema.isBlank()) {
            throw new LakeException(LakeFormat.GRAVITINO, "names", "schema required");
        }
        if (tbl == null || tbl.isBlank()) {
            throw new LakeException(LakeFormat.GRAVITINO, "names", "table required");
        }
        return new Names(catalog, schema, tbl);
    }

    private record Names(String catalog, String schema, String table) {
        String namespace() {
            return catalog + "." + schema;
        }
    }

    void ensureOpen() {
        if (closed.get()) {
            throw new LakeException(LakeFormat.GRAVITINO, "catalog", "closed");
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        for (LakeCatalog c : openedBackends) {
            try {
                c.close();
            } catch (Exception ignored) {
            }
        }
        openedBackends.clear();
        try {
            metalake.close();
        } catch (Exception ignored) {
        }
    }
}
