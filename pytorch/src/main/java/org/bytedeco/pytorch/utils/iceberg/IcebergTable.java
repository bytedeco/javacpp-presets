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

import org.apache.iceberg.BaseTable;
import org.apache.iceberg.PartitionField;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Snapshot;
import org.apache.iceberg.SortOrder;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableMetadata;
import org.apache.iceberg.TableOperations;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.Types;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.utils.lake.LakeCapabilities;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeSchema;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.PartitionSpec;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Iceberg table handle for local filesystem warehouses (no Hadoop).
 *
 * <p>Wraps {@link BaseTable} + {@link LocalFsTableOperations} and exposes a
 * unified {@link LakeTable} view for the Lake SPI. Not a subtype of
 * {@link LakeTable} (that type is a final value object) — use {@link #lakeTable()}.</p>
 *
 * <p>See DATA_LAKE_AI_ADAPTERS_PLAN.md §6.2 and
 * <a href="https://iceberg.apache.org/">Apache Iceberg</a>.</p>
 */
public final class IcebergTable implements AutoCloseable {

    private static final Pattern BUCKET = Pattern.compile("bucket\\[(\\d+)]", Pattern.CASE_INSENSITIVE);
    private static final Pattern TRUNCATE = Pattern.compile("truncate\\[(\\d+)]", Pattern.CASE_INSENSITIVE);

    private static final LakeCapabilities[] DEFAULT_CAPS = {
            LakeCapabilities.COLUMN_PROJECTION,
            LakeCapabilities.PARTITION_PRUNING,
            LakeCapabilities.INCREMENTAL_SCAN,
            LakeCapabilities.HIGH_THROUGHPUT_APPEND
    };

    private final Path tablePath;
    private final String namespaceName;
    private final String tableName;
    private final LocalFsFileIO fileIO;
    private final LocalFsTableOperations ops;
    private final Table icebergTable;
    private final boolean ownIo;
    private volatile LakeTable lakeView;

    private IcebergTable(Path tablePath,
                         String namespaceName,
                         String tableName,
                         LocalFsFileIO fileIO,
                         LocalFsTableOperations ops,
                         Table icebergTable,
                         boolean ownIo) {
        this.tablePath = Objects.requireNonNull(tablePath, "tablePath").toAbsolutePath().normalize();
        this.namespaceName = namespaceName == null ? "" : namespaceName;
        this.tableName = Objects.requireNonNull(tableName, "tableName");
        this.fileIO = fileIO == null ? new LocalFsFileIO() : fileIO;
        this.ops = Objects.requireNonNull(ops, "ops");
        this.icebergTable = Objects.requireNonNull(icebergTable, "icebergTable");
        this.ownIo = ownIo;
        this.lakeView = buildLakeTable();
    }

    /**
     * Load an existing Iceberg table from a local warehouse layout:
     * {@code warehouse/[namespace/]table/metadata/version-hint.text}.
     */
    public static IcebergTable load(Path warehouse, String namespaceName, String tableName, IcebergOptions options) {
        Objects.requireNonNull(warehouse, "warehouse");
        Objects.requireNonNull(tableName, "tableName");
        Path tablePath = resolveTablePath(warehouse, namespaceName, tableName);
        if (!Files.isDirectory(tablePath.resolve("metadata"))) {
            throw new LakeException(LakeFormat.ICEBERG, "loadTable",
                    "Iceberg metadata not found at " + tablePath.resolve("metadata"));
        }
        LocalFsFileIO io = new LocalFsFileIO();
        LocalFsTableOperations ops = new LocalFsTableOperations(tablePath, io);
        TableMetadata meta = ops.refresh();
        if (meta == null) {
            throw new LakeException(LakeFormat.ICEBERG, "loadTable",
                    "table does not exist: " + tablePath);
        }
        String fullName = (namespaceName == null || namespaceName.isBlank())
                ? tableName : namespaceName + "." + tableName;
        Table table = new BaseTable(ops, fullName);
        return new IcebergTable(tablePath, namespaceName, tableName, io, ops, table, true);
    }

    /**
     * Create a new Iceberg table under the warehouse (local FS only).
     */
    public static IcebergTable create(Path warehouse,
                                      String namespaceName,
                                      String tableName,
                                      LakeSchema lakeSchema,
                                      PartitionSpec lakePartitionSpec,
                                      Map<String, String> properties) {
        Objects.requireNonNull(warehouse, "warehouse");
        Objects.requireNonNull(tableName, "tableName");
        Objects.requireNonNull(lakeSchema, "schema");

        Path tablePath = resolveTablePath(warehouse, namespaceName, tableName);
        try {
            Files.createDirectories(tablePath.resolve("metadata"));
            Files.createDirectories(tablePath.resolve("data"));
        } catch (Exception e) {
            throw new LakeException(LakeFormat.ICEBERG, "createTable",
                    "failed to create table dirs at " + tablePath, e);
        }

        LocalFsFileIO io = new LocalFsFileIO();
        LocalFsTableOperations ops = new LocalFsTableOperations(tablePath, io);
        if (ops.tableExists()) {
            throw new LakeException(LakeFormat.ICEBERG, "createTable",
                    "table already exists: " + tablePath);
        }

        Schema schema = toIcebergSchema(lakeSchema);
        org.apache.iceberg.PartitionSpec spec = toIcebergPartitionSpec(schema, lakePartitionSpec);
        Map<String, String> props = new LinkedHashMap<>();
        if (properties != null) props.putAll(properties);
        props.putIfAbsent("write.format.default", "parquet");

        String location = LocalFsFileIO.toLocation(tablePath);
        TableMetadata metadata = TableMetadata.newTableMetadata(
                schema, spec, SortOrder.unsorted(), location, props);
        ops.commit(null, metadata);

        String fullName = (namespaceName == null || namespaceName.isBlank())
                ? tableName : namespaceName + "." + tableName;
        Table table = new BaseTable(ops, fullName);
        return new IcebergTable(tablePath, namespaceName, tableName, io, ops, table, true);
    }

    static Path resolveTablePath(Path warehouse, String namespaceName, String tableName) {
        Path wh = warehouse.toAbsolutePath().normalize();
        if (namespaceName != null && !namespaceName.isBlank()) {
            return wh.resolve(namespaceName).resolve(tableName);
        }
        return wh.resolve(tableName);
    }

    // ── accessors ────────────────────────────────────────────────────────────

    public Path tablePath() {
        return tablePath;
    }

    public String namespaceName() {
        return namespaceName;
    }

    public String tableName() {
        return tableName;
    }

    /** Underlying Iceberg {@link Table} (usually {@link BaseTable}). */
    public Table icebergTable() {
        return icebergTable;
    }

    public TableOperations operations() {
        return ops;
    }

    public LocalFsFileIO fileIO() {
        return fileIO;
    }

    /** Unified Lake SPI value object (immutable snapshot of metadata). */
    public LakeTable lakeTable() {
        LakeTable view = lakeView;
        if (view == null) {
            view = buildLakeTable();
            lakeView = view;
        }
        return view;
    }

    /** Refresh Iceberg metadata and rebuild the LakeTable view. */
    public LakeTable refresh() {
        icebergTable.refresh();
        lakeView = buildLakeTable();
        return lakeView;
    }

    public Long currentSnapshotId() {
        Snapshot s = icebergTable.currentSnapshot();
        return s == null ? null : s.snapshotId();
    }

    // ── schema / partition mapping ───────────────────────────────────────────

    private LakeTable buildLakeTable() {
        Long snap = currentSnapshotId();
        return LakeTable.builder(LakeFormat.ICEBERG, tableName, toLakeSchema(icebergTable.schema()))
                .namespaceName(namespaceName)
                .partitionSpec(toLakePartitionSpec(icebergTable.spec()))
                .location(icebergTable.location())
                .properties(icebergTable.properties())
                .capabilities(DEFAULT_CAPS)
                .currentSnapshotId(snap)
                .build();
    }

    static LakeSchema toLakeSchema(Schema schema) {
        Objects.requireNonNull(schema, "schema");
        LakeSchema.Builder b = LakeSchema.builder();
        for (Types.NestedField f : schema.columns()) {
            b.add(f.name(), toDataFrameType(f.type()), f.isOptional());
        }
        return b.build();
    }

    static Schema toIcebergSchema(LakeSchema lakeSchema) {
        Objects.requireNonNull(lakeSchema, "lakeSchema");
        List<Types.NestedField> fields = new ArrayList<>(lakeSchema.size());
        int id = 1;
        for (LakeSchema.Field f : lakeSchema.fields()) {
            Type t = toIcebergType(f.dtype());
            if (f.nullable()) {
                fields.add(Types.NestedField.optional(id++, f.name(), t));
            } else {
                fields.add(Types.NestedField.required(id++, f.name(), t));
            }
        }
        return new Schema(fields);
    }

    static Column.DType toDataFrameType(Type type) {
        if (type == null) return Column.DType.STRING;
        return switch (type.typeId()) {
            case BOOLEAN -> Column.DType.BOOLEAN;
            case INTEGER -> Column.DType.INT32;
            case LONG -> Column.DType.INT64;
            case FLOAT -> Column.DType.FLOAT32;
            case DOUBLE, DECIMAL -> Column.DType.FLOAT64;
            case DATE -> Column.DType.DATE;
            case TIME -> Column.DType.TIME;
            case TIMESTAMP, TIMESTAMP_NANO -> Column.DType.DATETIME;
            case STRING, UUID -> Column.DType.STRING;
            case BINARY, FIXED -> Column.DType.BINARY;
            case LIST -> Column.DType.LIST;
            case MAP -> Column.DType.MAP;
            case STRUCT -> Column.DType.STRUCT;
            default -> Column.DType.STRING;
        };
    }

    static Type toIcebergType(Column.DType dtype) {
        if (dtype == null) return Types.StringType.get();
        return switch (dtype) {
            case BOOLEAN -> Types.BooleanType.get();
            case INT32 -> Types.IntegerType.get();
            case INT64 -> Types.LongType.get();
            case FLOAT32 -> Types.FloatType.get();
            case FLOAT64 -> Types.DoubleType.get();
            case DATE -> Types.DateType.get();
            case TIME -> Types.TimeType.get();
            case DATETIME, DURATION -> Types.TimestampType.withZone();
            case BINARY, IMAGE, AUDIO, VIDEO, EMBEDDING, POINT_CLOUD -> Types.BinaryType.get();
            case LIST, VECTOR -> Types.ListType.ofOptional(Integer.MAX_VALUE - 10, Types.StringType.get());
            case MAP, STRUCT, JSON, GRAPH -> Types.StringType.get(); // serialized fallback
            case STRING, TENSOR -> Types.StringType.get();
        };
    }

    static PartitionSpec toLakePartitionSpec(org.apache.iceberg.PartitionSpec spec) {
        if (spec == null || spec.isUnpartitioned()) {
            return PartitionSpec.builder().build();
        }
        List<String> identity = new ArrayList<>();
        List<String> timeTruncate = new ArrayList<>();
        List<Integer> buckets = new ArrayList<>();
        Schema schema = spec.schema();
        for (PartitionField f : spec.fields()) {
            String transform = f.transform() == null ? "identity" : f.transform().toString();
            String source = schema == null ? f.name() : schema.findColumnName(f.sourceId());
            if (source == null || source.isBlank()) source = f.name();
            String lower = transform.toLowerCase();
            if ("identity".equals(lower)) {
                identity.add(source);
            } else if ("year".equals(lower) || "month".equals(lower)
                    || "day".equals(lower) || "hour".equals(lower)) {
                timeTruncate.add(lower + ":" + source);
            } else {
                Matcher bm = BUCKET.matcher(transform);
                if (bm.find()) {
                    try {
                        buckets.add(Integer.parseInt(bm.group(1)));
                    } catch (NumberFormatException ignored) {
                        identity.add(source);
                    }
                } else {
                    Matcher tm = TRUNCATE.matcher(transform);
                    if (tm.find()) {
                        timeTruncate.add("truncate" + tm.group(1) + ":" + source);
                    } else {
                        identity.add(source);
                    }
                }
            }
        }
        int[] bucketArr = new int[buckets.size()];
        for (int i = 0; i < buckets.size(); i++) bucketArr[i] = buckets.get(i);
        return PartitionSpec.builder()
                .identityColumns(identity.toArray(new String[0]))
                .timeTruncate(timeTruncate.toArray(new String[0]))
                .bucketColumns(bucketArr)
                .build();
    }

    static org.apache.iceberg.PartitionSpec toIcebergPartitionSpec(Schema schema, PartitionSpec lakeSpec) {
        if (lakeSpec == null) {
            return org.apache.iceberg.PartitionSpec.unpartitioned();
        }
        String[] identity = lakeSpec.identityColumns();
        String[] time = lakeSpec.timeTruncate();
        boolean hasIdentity = identity != null && identity.length > 0;
        boolean hasTime = time != null && time.length > 0;
        if (!hasIdentity && !hasTime) {
            return org.apache.iceberg.PartitionSpec.unpartitioned();
        }
        org.apache.iceberg.PartitionSpec.Builder b = org.apache.iceberg.PartitionSpec.builderFor(schema);
        if (hasIdentity) {
            for (String col : identity) {
                if (col != null && !col.isBlank() && schema.findField(col) != null) {
                    b.identity(col);
                }
            }
        }
        if (hasTime) {
            for (String t : time) {
                if (t == null || t.isBlank()) continue;
                // forms: "day", "hour", "day:col", "hour:event_time"
                String transform = t;
                String col = null;
                int colon = t.indexOf(':');
                if (colon > 0) {
                    transform = t.substring(0, colon).trim();
                    col = t.substring(colon + 1).trim();
                }
                if (col == null || col.isBlank()) {
                    // no column — skip rather than guess
                    continue;
                }
                if (schema.findField(col) == null) continue;
                switch (transform.toLowerCase()) {
                    case "year" -> b.year(col);
                    case "month" -> b.month(col);
                    case "day" -> b.day(col);
                    case "hour" -> b.hour(col);
                    default -> {
                        if (transform.toLowerCase().startsWith("truncate")) {
                            b.identity(col); // truncate width not carried in lake PartitionSpec cleanly
                        } else {
                            b.identity(col);
                        }
                    }
                }
            }
        }
        return b.build();
    }

    @Override
    public void close() {
        try {
            if (ownIo && fileIO != null) {
                fileIO.close();
            }
        } catch (Exception ignored) {
            // best-effort
        }
    }

    @Override
    public String toString() {
        return "IcebergTable{" + namespaceName + (namespaceName.isEmpty() ? "" : ".") + tableName
                + " loc=" + tablePath + "}";
    }
}
