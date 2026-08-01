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

import org.apache.iceberg.DataFile;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.TableScan;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.io.CloseableIterable;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;
import org.bytedeco.pytorch.utils.lake.ReplicaPolicy;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Iceberg batch scan → {@link DataFrame} via TableScan plan + existing Parquet reader.
 *
 * <p>Path (DATA_LAKE_AI_ADAPTERS_PLAN.md §6.2):
 * {@code TableScan → planFiles → DataFrame.readParquet} with column projection and
 * partition prune through Iceberg expressions.</p>
 *
 * <p>No Hadoop / Spark runtime — only {@code iceberg-core} + local FileIO.</p>
 */
public final class IcebergScan implements LakeScan {

    private final IcebergCatalog catalog;
    private final IcebergTable tableHandle;
    private final LakeTable lakeTable;
    private final LakeMetrics metrics;

    private String[] columns;
    private PartitionFilter partitionFilter;
    private String where;
    private Long snapshotId;
    private Long asOfTimeMs;
    private ReplicaPolicy replicas;
    private long limit = -1;
    private int batchRows;
    private int parallelism = 1;

    IcebergScan(IcebergCatalog catalog, IcebergTable tableHandle) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.tableHandle = Objects.requireNonNull(tableHandle, "tableHandle");
        this.lakeTable = tableHandle.lakeTable();
        this.metrics = catalog.metrics();
        this.batchRows = Math.max(1, catalog.options().batchRows());
        this.parallelism = Math.max(1, catalog.options().parallelism());
        this.replicas = catalog.options().replicaPolicy();
        if (catalog.options().columns() != null) this.columns = catalog.options().columns();
        if (catalog.options().partitionFilter() != null) this.partitionFilter = catalog.options().partitionFilter();
        if (catalog.options().snapshotId() != null) this.snapshotId = catalog.options().snapshotId();
        if (catalog.options().asOfTimeMs() != null) this.asOfTimeMs = catalog.options().asOfTimeMs();
    }

    @Override
    public LakeTable table() {
        return lakeTable;
    }

    IcebergTable tableHandle() {
        return tableHandle;
    }

    @Override
    public LakeScan columns(String... columns) {
        this.columns = columns;
        return this;
    }

    @Override
    public LakeScan filter(PartitionFilter filter) {
        this.partitionFilter = filter;
        return this;
    }

    @Override
    public LakeScan where(String expression) {
        this.where = expression;
        return this;
    }

    @Override
    public LakeScan snapshotId(Long snapshotId) {
        this.snapshotId = snapshotId;
        return this;
    }

    @Override
    public LakeScan asOfTimeMs(Long epochMs) {
        this.asOfTimeMs = epochMs;
        return this;
    }

    @Override
    public LakeScan replicas(ReplicaPolicy policy) {
        this.replicas = policy;
        return this;
    }

    @Override
    public LakeScan limit(long maxRows) {
        this.limit = maxRows;
        return this;
    }

    @Override
    public LakeScan batchRows(int batchRows) {
        this.batchRows = Math.max(1, batchRows);
        return this;
    }

    @Override
    public LakeScan parallelism(int parallelism) {
        this.parallelism = Math.max(1, parallelism);
        return this;
    }

    /** Build a configured Iceberg {@link TableScan} (snapshot / filter / projection). */
    TableScan buildTableScan() {
        TableScan scan = tableHandle.icebergTable().newScan();
        if (snapshotId != null) {
            scan = scan.useSnapshot(snapshotId);
        } else if (asOfTimeMs != null) {
            scan = scan.asOfTime(asOfTimeMs);
        }
        if (columns != null && columns.length > 0) {
            scan = scan.select(columns);
        }
        Expression expr = toIcebergExpression(partitionFilter);
        if (expr != null && expr != Expressions.alwaysTrue()) {
            scan = scan.filter(expr);
        }
        // free-form where is not pushed as Iceberg Expression (no SQL parser here);
        // residual row filter applied post-read when present is not implemented —
        // document: use PartitionFilter for prune; full residual needs engine SQL.
        return scan;
    }

    static Expression toIcebergExpression(PartitionFilter filter) {
        if (filter == null || filter.isEmpty()) {
            return Expressions.alwaysTrue();
        }
        Expression acc = null;
        for (PartitionFilter.Predicate p : filter.predicates()) {
            Expression e = predicateToExpression(p);
            if (e == null) continue;
            acc = acc == null ? e : Expressions.and(acc, e);
        }
        return acc == null ? Expressions.alwaysTrue() : acc;
    }

    private static Expression predicateToExpression(PartitionFilter.Predicate p) {
        String col = p.column();
        List<String> values = p.values();
        if (col == null || values == null || values.isEmpty()) return null;
        return switch (p.op()) {
            case EQ -> Expressions.equal(col, coerce(values.get(0)));
            case IN -> Expressions.in(col, values.stream().map(IcebergScan::coerce).toArray());
            case GT -> Expressions.greaterThan(col, coerce(values.get(0)));
            case GTE -> Expressions.greaterThanOrEqual(col, coerce(values.get(0)));
            case LT -> Expressions.lessThan(col, coerce(values.get(0)));
            case LTE -> Expressions.lessThanOrEqual(col, coerce(values.get(0)));
        };
    }

    /** Best-effort literal coercion for partition prune (string / long / double / bool). */
    private static Object coerce(String raw) {
        if (raw == null) return null;
        String s = raw.trim();
        if (s.isEmpty()) return s;
        if ("true".equalsIgnoreCase(s) || "false".equalsIgnoreCase(s)) {
            return Boolean.parseBoolean(s);
        }
        try {
            if (s.indexOf('.') >= 0 || s.indexOf('e') >= 0 || s.indexOf('E') >= 0) {
                return Double.parseDouble(s);
            }
            return Long.parseLong(s);
        } catch (NumberFormatException e) {
            return s;
        }
    }

    @Override
    public DataFrame collect() throws LakeException {
        long t0 = System.nanoTime();
        try {
            List<DataFrame> parts = new ArrayList<>();
            long rows = 0;
            try (CloseableIterable<FileScanTask> tasks = buildTableScan().planFiles()) {
                for (FileScanTask task : tasks) {
                    DataFrame df = readTask(task);
                    if (df == null || df.rowCount() == 0) continue;
                    parts.add(df);
                    rows += df.rowCount();
                    if (limit >= 0 && rows >= limit) break;
                }
            }
            DataFrame out;
            if (parts.isEmpty()) {
                out = emptyFrame();
            } else if (parts.size() == 1) {
                out = parts.get(0);
            } else {
                out = DataFrame.vstack(parts);
            }
            if (limit >= 0 && out.rowCount() > limit) {
                out = out.limit((int) Math.min(Integer.MAX_VALUE, limit));
            }
            if (columns != null && columns.length > 0 && out.rowCount() >= 0) {
                // projection may already be applied by Parquet reader when columns passed;
                // re-select if all columns present
                try {
                    out = out.select(columns);
                } catch (Exception ignored) {
                    // keep full frame if select fails (missing col names)
                }
            }
            metrics.recordRead(out.rowCount(), System.nanoTime() - t0);
            return out;
        } catch (LakeException e) {
            metrics.recordFailure();
            throw e;
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.ICEBERG, "scan.collect",
                    "failed to collect Iceberg scan for " + lakeTable.fullName(), e);
        }
    }

    private DataFrame emptyFrame() {
        DataFrame df = DataFrame.create();
        String[] names = columns != null && columns.length > 0
                ? columns
                : lakeTable.schema().names();
        var dtypes = lakeTable.schema().dtypes();
        for (int i = 0; i < names.length; i++) {
            var dt = i < dtypes.length ? dtypes[i] : org.bytedeco.pytorch.dataframe.Column.DType.STRING;
            // if projected subset, look up dtype from schema
            if (columns != null && columns.length > 0) {
                var f = lakeTable.schema().get(names[i]);
                dt = f != null ? f.dtype() : org.bytedeco.pytorch.dataframe.Column.DType.STRING;
            }
            df.addColumn(names[i], dt);
        }
        return df;
    }

    private DataFrame readTask(FileScanTask task) {
        DataFile file = task.file();
        if (file == null) return null;
        String path = file.path() == null ? null : file.path().toString();
        if (path == null || path.isBlank()) return null;
        path = LocalFsFileIO.toLocation(LocalFsFileIO.toPath(path));
        try {
            DataFrame df;
            if (columns != null && columns.length > 0) {
                try {
                    df = DataFrame.readParquet(path, columns);
                } catch (Exception colFail) {
                    df = DataFrame.readParquet(path);
                    try {
                        df = df.select(columns);
                    } catch (Exception ignored) {
                        // keep full
                    }
                }
            } else {
                df = DataFrame.readParquet(path);
            }
            return df;
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.ICEBERG, "scan.readFile",
                    "failed to read data file " + path, e);
        }
    }

    /** List planned data-file paths (for diagnostics / lightweight export). */
    public List<String> planDataFiles() {
        List<String> paths = new ArrayList<>();
        try (CloseableIterable<FileScanTask> tasks = buildTableScan().planFiles()) {
            for (FileScanTask task : tasks) {
                if (task.file() != null && task.file().path() != null) {
                    paths.add(task.file().path().toString());
                }
            }
        } catch (Exception e) {
            throw new LakeException(LakeFormat.ICEBERG, "scan.planFiles",
                    "failed to plan files", e);
        }
        return paths;
    }

    @Override
    public LakeStream stream() throws LakeException {
        return new IcebergFileStream(this, batchRows, catalog.options().idleStop(), limit);
    }

    /**
     * Micro-batch stream over planned data files (one or more files per poll batch).
     * Snapshot incremental streaming is handled by {@link IcebergStream} when present;
     * this path is the batch-scan-as-stream used by {@link LakeScan#stream()}.
     */
    static final class IcebergFileStream implements LakeStream {
        private final IcebergScan scan;
        private int batchRows;
        private Duration idleStop;
        private final long rowLimit;
        private final AtomicBoolean stopped = new AtomicBoolean(false);
        private final AtomicBoolean closed = new AtomicBoolean(false);

        private Iterator<FileScanTask> taskIt;
        private CloseableIterable<FileScanTask> taskIterable;
        private long rowsEmitted;
        private long batchesEmitted;
        private long maxBatches = Long.MAX_VALUE;
        private boolean primed;

        IcebergFileStream(IcebergScan scan, int batchRows, Duration idleStop, long rowLimit) {
            this.scan = scan;
            this.batchRows = Math.max(1, batchRows);
            this.idleStop = idleStop == null ? Duration.ofSeconds(30) : idleStop;
            this.rowLimit = rowLimit;
        }

        @Override
        public LakeStream batchRows(int batchRows) {
            this.batchRows = Math.max(1, batchRows);
            return this;
        }

        @Override
        public LakeStream idleStop(Duration idle) {
            this.idleStop = idle == null ? Duration.ofSeconds(30) : idle;
            return this;
        }

        @Override
        public LakeStream maxBatches(long maxBatches) {
            this.maxBatches = maxBatches <= 0 ? Long.MAX_VALUE : maxBatches;
            return this;
        }

        @Override
        public void commit() {
            // file-list stream: watermark is rows/batches already advanced in poll
        }

        @Override
        public void stop() {
            stopped.set(true);
        }

        @Override
        public boolean isStopped() {
            return stopped.get() || closed.get() || batchesEmitted >= maxBatches
                    || (rowLimit >= 0 && rowsEmitted >= rowLimit);
        }

        private void ensurePlan() {
            if (primed) return;
            try {
                taskIterable = scan.buildTableScan().planFiles();
                taskIt = taskIterable.iterator();
                primed = true;
            } catch (Exception e) {
                scan.metrics.recordFailure();
                throw new LakeException(LakeFormat.ICEBERG, "stream.plan",
                        "failed to plan Iceberg files", e);
            }
        }

        @Override
        public DataFrame poll() {
            if (isStopped()) return null;
            long t0 = System.nanoTime();
            ensurePlan();
            List<DataFrame> parts = new ArrayList<>();
            int rowsInBatch = 0;
            try {
                while (taskIt.hasNext() && rowsInBatch < batchRows) {
                    if (isStopped()) break;
                    FileScanTask task = taskIt.next();
                    DataFrame df = scan.readTask(task);
                    if (df == null || df.rowCount() == 0) continue;
                    if (rowLimit >= 0 && rowsEmitted + rowsInBatch + df.rowCount() > rowLimit) {
                        long remain = rowLimit - rowsEmitted - rowsInBatch;
                        if (remain <= 0) break;
                        df = df.limit((int) Math.min(Integer.MAX_VALUE, remain));
                    }
                    parts.add(df);
                    rowsInBatch += df.rowCount();
                }
            } catch (LakeException e) {
                throw e;
            } catch (Exception e) {
                scan.metrics.recordFailure();
                throw new LakeException(LakeFormat.ICEBERG, "stream.poll",
                        "failed reading Iceberg data file", e);
            }
            if (parts.isEmpty()) {
                stopped.set(true);
                return null;
            }
            try {
                DataFrame out = parts.size() == 1 ? parts.get(0) : DataFrame.vstack(parts);
                rowsEmitted += out.rowCount();
                batchesEmitted++;
                scan.metrics.recordBatch(out.rowCount());
                scan.metrics.recordRead(out.rowCount(), System.nanoTime() - t0);
                return out;
            } catch (Exception e) {
                scan.metrics.recordFailure();
                throw new LakeException(LakeFormat.ICEBERG, "stream.poll",
                        "failed to vstack batch", e);
            }
        }

        @Override
        public void close() {
            if (!closed.compareAndSet(false, true)) return;
            stopped.set(true);
            if (taskIterable != null) {
                try {
                    taskIterable.close();
                } catch (Exception ignored) {
                }
            }
        }
    }
}
