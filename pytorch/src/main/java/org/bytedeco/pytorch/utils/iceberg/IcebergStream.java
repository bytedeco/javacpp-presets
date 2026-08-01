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
import org.apache.iceberg.IncrementalAppendScan;
import org.apache.iceberg.Snapshot;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableScan;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.CloseableIterable;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;

import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Iceberg micro-batch stream aligned with {@code KafkaStream} / {@link LakeStream}.
 *
 * <p>Modes (DATA_LAKE_AI_ADAPTERS_PLAN.md §5.3 / §6.2):</p>
 * <ul>
 *   <li><b>Snapshot batch</b> — scan current (or pinned) snapshot data files once.</li>
 *   <li><b>Incremental</b> — {@link IncrementalAppendScan} from {@code fromSnapshotId}
 *       exclusive toward current; {@link #commit()} advances the watermark so the next
 *       poll only sees newer appends (at-least-once).</li>
 * </ul>
 *
 * <pre>{@code
 * try (LakeStream s = Iceberg.stream(opts)) {
 *   s.batchRows(4096).forEachBatch(df -> {
 *     // online train / feature join
 *     s.commit(); // advance snapshot cursor
 *   });
 * }
 * }</pre>
 *
 * <p>No Hadoop / Spark — uses {@code iceberg-core} + {@link DataFrame#readParquet}.</p>
 */
public final class IcebergStream implements LakeStream {

    private final IcebergCatalog catalog;
    private final IcebergTable tableHandle;
    private final LakeTable lakeTable;
    private final LakeMetrics metrics;
    private final boolean ownCatalog;

    private String[] columns;
    private PartitionFilter partitionFilter;
    private int batchRows;
    private Duration idleStop;
    private long maxBatches = Long.MAX_VALUE;
    private long rowLimit = -1;

    /** Exclusive start for incremental scans; null = full current snapshot once. */
    private Long fromSnapshotExclusive;
    /** Inclusive end pin; null = live current snapshot. */
    private Long toSnapshotInclusive;
    private boolean incremental;

    private final AtomicBoolean stopped = new AtomicBoolean(false);
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private CloseableIterable<FileScanTask> taskIterable;
    private Iterator<FileScanTask> taskIt;
    private boolean planExhausted;
    private boolean primed;

    private final AtomicLong committedSnapshotId = new AtomicLong(-1L);
    private long lastSeenSnapshotId = -1L;
    private long rowsEmitted;
    private long batchesEmitted;
    private long lastActivityMs = System.currentTimeMillis();
    private long idleWaitStartedMs = -1L;

    IcebergStream(IcebergCatalog catalog, IcebergTable tableHandle) {
        this(catalog, tableHandle, false);
    }

    IcebergStream(IcebergCatalog catalog, IcebergTable tableHandle, boolean ownCatalog) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.tableHandle = Objects.requireNonNull(tableHandle, "tableHandle");
        this.lakeTable = tableHandle.lakeTable();
        this.metrics = catalog.metrics();
        this.ownCatalog = ownCatalog;
        IcebergOptions opts = catalog.options();
        this.batchRows = Math.max(1, opts.batchRows());
        this.idleStop = opts.idleStop() == null ? Duration.ofSeconds(30) : opts.idleStop();
        this.columns = opts.columns();
        this.partitionFilter = opts.partitionFilter();
        if (opts.fromSnapshotId() != null) {
            this.fromSnapshotExclusive = opts.fromSnapshotId();
            this.incremental = true;
        }
        if (opts.snapshotId() != null) {
            this.toSnapshotInclusive = opts.snapshotId();
        }
        Long committed = committedSnapshotId.get();
        if (committed != null && committed >= 0) {
            this.fromSnapshotExclusive = committed;
            this.incremental = true;
        }
    }

    /** Open a stream from options (owns catalog lifecycle). */
    public static IcebergStream open(IcebergOptions options) {
        IcebergCatalog cat = IcebergCatalog.open(options);
        return new IcebergStream(cat, cat.tableHandle(), true);
    }

    public static IcebergStream open(org.bytedeco.pytorch.utils.lake.LakeOptions lakeOptions) {
        return open(IcebergOptions.fromLakeOptions(lakeOptions));
    }

    public IcebergStream columns(String... columns) {
        this.columns = columns;
        return this;
    }

    public IcebergStream filter(PartitionFilter filter) {
        this.partitionFilter = filter;
        return this;
    }

    /** Start exclusive snapshot for incremental append scan. */
    public IcebergStream fromSnapshotId(Long snapshotId) {
        this.fromSnapshotExclusive = snapshotId;
        this.incremental = snapshotId != null;
        resetPlan();
        return this;
    }

    /** Pin end snapshot (inclusive); null follows live head. */
    public IcebergStream toSnapshotId(Long snapshotId) {
        this.toSnapshotInclusive = snapshotId;
        resetPlan();
        return this;
    }

    public IcebergStream incremental(boolean incremental) {
        this.incremental = incremental;
        resetPlan();
        return this;
    }

    public IcebergStream rowLimit(long maxRows) {
        this.rowLimit = maxRows;
        return this;
    }

    public LakeTable table() {
        return lakeTable;
    }

    public IcebergTable tableHandle() {
        return tableHandle;
    }

    public LakeMetrics metrics() {
        return metrics;
    }

    /** Last snapshot id observed while planning (or -1). */
    public long lastSeenSnapshotId() {
        return lastSeenSnapshotId;
    }

    /** Watermark advanced by {@link #commit()} (exclusive lower bound for next plan). */
    public long committedSnapshotId() {
        return committedSnapshotId.get();
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

    /**
     * Advance the read watermark to the last planned snapshot so subsequent
     * incremental polls only see newer appends (at-least-once).
     */
    @Override
    public void commit() throws LakeException {
        if (lastSeenSnapshotId >= 0) {
            committedSnapshotId.set(lastSeenSnapshotId);
            fromSnapshotExclusive = lastSeenSnapshotId;
            incremental = true;
        }
    }

    @Override
    public void stop() {
        stopped.set(true);
    }

    @Override
    public boolean isStopped() {
        return stopped.get() || closed.get()
                || batchesEmitted >= maxBatches
                || (rowLimit >= 0 && rowsEmitted >= rowLimit);
    }

    private void resetPlan() {
        closePlan();
        primed = false;
        planExhausted = false;
        idleWaitStartedMs = -1L;
    }

    private void closePlan() {
        if (taskIterable != null) {
            try {
                taskIterable.close();
            } catch (Exception ignored) {
            }
        }
        taskIterable = null;
        taskIt = null;
    }

    private void ensurePlan() {
        if (primed && !planExhausted) return;
        if (planExhausted && !incremental) return;

        closePlan();
        try {
            tableHandle.icebergTable().refresh();
        } catch (Exception ignored) {
            // best-effort refresh
        }

        Table table = tableHandle.icebergTable();
        Snapshot current = table.currentSnapshot();
        if (current == null) {
            planExhausted = true;
            primed = true;
            taskIt = List.<FileScanTask>of().iterator();
            return;
        }

        long endId = toSnapshotInclusive != null ? toSnapshotInclusive : current.snapshotId();
        lastSeenSnapshotId = endId;

        Long startExcl = fromSnapshotExclusive;
        if (committedSnapshotId.get() >= 0) {
            startExcl = committedSnapshotId.get();
        }

        Expression filterExpr = IcebergScan.toIcebergExpression(partitionFilter);

        try {
            if (incremental && startExcl != null && startExcl >= 0 && !Objects.equals(startExcl, endId)) {
                IncrementalAppendScan inc = table.newIncrementalAppendScan();
                inc = inc.fromSnapshotExclusive(startExcl).toSnapshot(endId);
                if (columns != null && columns.length > 0) {
                    inc = inc.select(columns);
                }
                if (filterExpr != null) {
                    inc = inc.filter(filterExpr);
                }
                taskIterable = inc.planFiles();
            } else if (incremental && startExcl != null && Objects.equals(startExcl, endId)) {
                // nothing new
                planExhausted = true;
                primed = true;
                taskIt = List.<FileScanTask>of().iterator();
                return;
            } else {
                // full snapshot scan (once)
                TableScan scan = table.newScan().useSnapshot(endId);
                if (columns != null && columns.length > 0) {
                    scan = scan.select(columns);
                }
                if (filterExpr != null) {
                    scan = scan.filter(filterExpr);
                }
                taskIterable = scan.planFiles();
            }
            taskIt = taskIterable.iterator();
            primed = true;
            planExhausted = false;
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.ICEBERG, "stream.plan",
                    "failed to plan Iceberg stream files for " + lakeTable.fullName(), e);
        }
    }

    @Override
    public DataFrame poll() throws LakeException {
        if (isStopped()) return null;
        long t0 = System.nanoTime();

        // Loop instead of recursive re-poll while waiting for incremental snapshots
        // (avoids stack growth under long idleStop).
        while (!isStopped()) {
            ensurePlan();

            List<DataFrame> parts = new ArrayList<>();
            int rowsInBatch = 0;

            try {
                while (taskIt != null && taskIt.hasNext() && rowsInBatch < batchRows) {
                    if (isStopped()) break;
                    FileScanTask task = taskIt.next();
                    DataFrame df = readTask(task);
                    if (df == null || df.rowCount() == 0) continue;

                    if (rowLimit >= 0) {
                        long remain = rowLimit - rowsEmitted - rowsInBatch;
                        if (remain <= 0) break;
                        if (df.rowCount() > remain) {
                            df = df.limit((int) Math.min(Integer.MAX_VALUE, remain));
                        }
                    }
                    parts.add(df);
                    rowsInBatch += df.rowCount();
                    lastActivityMs = System.currentTimeMillis();
                    idleWaitStartedMs = -1L;
                }

                if (taskIt != null && !taskIt.hasNext()) {
                    planExhausted = true;
                    closePlan();
                }
            } catch (LakeException e) {
                throw e;
            } catch (Exception e) {
                metrics.recordFailure();
                throw new LakeException(LakeFormat.ICEBERG, "stream.poll",
                        "failed reading Iceberg data file", e);
            }

            if (!parts.isEmpty()) {
                try {
                    DataFrame out = parts.size() == 1 ? parts.get(0) : DataFrame.vstack(parts);
                    if (columns != null && columns.length > 0) {
                        try {
                            out = out.select(columns);
                        } catch (Exception ignored) {
                        }
                    }
                    rowsEmitted += out.rowCount();
                    batchesEmitted++;
                    metrics.recordBatch(out.rowCount());
                    metrics.recordRead(out.rowCount(), System.nanoTime() - t0);
                    return out;
                } catch (Exception e) {
                    metrics.recordFailure();
                    throw new LakeException(LakeFormat.ICEBERG, "stream.poll",
                            "failed to assemble stream batch", e);
                }
            }

            // No data this cycle
            if (!incremental) {
                stopped.set(true);
                return null;
            }

            long now = System.currentTimeMillis();
            if (idleWaitStartedMs < 0) idleWaitStartedMs = now;
            long idleMs = idleStop == null ? 0L : idleStop.toMillis();
            if (idleMs > 0 && now - idleWaitStartedMs > idleMs) {
                stopped.set(true);
                return null;
            }
            // replan against latest snapshot after a short yield
            primed = false;
            planExhausted = false;
            try {
                Thread.sleep(Math.min(200L, Math.max(10L, idleMs <= 0 ? 50L : idleMs / 10)));
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                stopped.set(true);
                return null;
            }
        }
        return null;
    }

    private DataFrame readTask(FileScanTask task) {
        DataFile file = task.file();
        if (file == null) return null;
        String path = file.path() == null ? null : file.path().toString();
        if (path == null || path.isBlank()) return null;
        path = LocalFsFileIO.toLocation(LocalFsFileIO.toPath(path));
        try {
            if (columns != null && columns.length > 0) {
                try {
                    return DataFrame.readParquet(path, columns);
                } catch (Exception colFail) {
                    DataFrame df = DataFrame.readParquet(path);
                    try {
                        return df.select(columns);
                    } catch (Exception ignored) {
                        return df;
                    }
                }
            }
            return DataFrame.readParquet(path);
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.ICEBERG, "stream.readFile",
                    "failed to read data file " + path, e);
        }
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        stopped.set(true);
        closePlan();
        if (ownCatalog) {
            try {
                catalog.close();
            } catch (Exception ignored) {
            }
        }
    }
}
