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

import org.apache.iceberg.AppendFiles;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.DataFiles;
import org.apache.iceberg.FileFormat;
import org.apache.iceberg.Metrics;
import org.apache.iceberg.OverwriteFiles;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Table;
import org.apache.iceberg.expressions.Expressions;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Iceberg write path: buffer {@link DataFrame}s → write Parquet data files →
 * {@link AppendFiles} / {@link OverwriteFiles} commit.
 *
 * <p>Per DATA_LAKE_AI_ADAPTERS_PLAN.md §5.2 / §6.2:</p>
 * <ul>
 *   <li>Primary path: Parquet under {@code data/} then DataFile append + commit.</li>
 *   <li>Idempotency: optional {@link #label(String)} stored in snapshot summary.</li>
 *   <li>No Hadoop — {@link LocalFsFileIO} + existing {@link DataFrame#writeParquet}.</li>
 * </ul>
 *
 * <p>Caller must {@link #commit()} explicitly (default {@link #close()} does not auto-commit).</p>
 */
public final class IcebergWrite implements LakeWrite {

    private final IcebergCatalog catalog;
    private final IcebergTable tableHandle;
    private final LakeTable lakeTable;
    private final LakeMetrics metrics;

    private Mode mode = Mode.APPEND;
    private PartitionFilter staticPartition;
    private String label;
    private final List<PendingFile> pending = new ArrayList<>();
    private final AtomicBoolean committed = new AtomicBoolean(false);
    private final AtomicBoolean aborted = new AtomicBoolean(false);
    private final AtomicBoolean closed = new AtomicBoolean(false);

    private record PendingFile(Path path, long recordCount, long fileSizeBytes, String partitionPath) {}

    IcebergWrite(IcebergCatalog catalog, IcebergTable tableHandle) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.tableHandle = Objects.requireNonNull(tableHandle, "tableHandle");
        this.lakeTable = tableHandle.lakeTable();
        this.metrics = catalog.metrics();
    }

    @Override
    public LakeTable table() {
        return lakeTable;
    }

    IcebergTable tableHandle() {
        return tableHandle;
    }

    @Override
    public LakeWrite mode(Mode mode) {
        this.mode = mode == null ? Mode.APPEND : mode;
        return this;
    }

    @Override
    public LakeWrite partition(PartitionFilter staticPartition) {
        this.staticPartition = staticPartition;
        return this;
    }

    @Override
    public LakeWrite label(String label) {
        this.label = label;
        return this;
    }

    @Override
    public LakeWrite write(DataFrame df) throws LakeException {
        if (aborted.get()) {
            throw new LakeException(LakeFormat.ICEBERG, "write", "aborted");
        }
        if (committed.get()) {
            throw new LakeException(LakeFormat.ICEBERG, "write", "already committed");
        }
        if (closed.get()) {
            throw new LakeException(LakeFormat.ICEBERG, "write", "closed");
        }
        Objects.requireNonNull(df, "dataframe");
        if (df.rowCount() == 0) {
            return this;
        }

        long t0 = System.nanoTime();
        try {
            Path dataRoot = tableHandle.tablePath().resolve("data");
            Files.createDirectories(dataRoot);

            String partitionPath = partitionPathFromFilter(staticPartition);
            Path dir = dataRoot;
            if (partitionPath != null && !partitionPath.isBlank()) {
                dir = dataRoot.resolve(partitionPath);
                Files.createDirectories(dir);
            }

            String fileName = "part-"
                    + (label != null && !label.isBlank() ? sanitize(label) + "-" : "")
                    + UUID.randomUUID().toString().replace("-", "")
                    + ".parquet";
            Path out = dir.resolve(fileName);

            df.writeParquet(out.toString());
            long size = Files.size(out);
            pending.add(new PendingFile(out, df.rowCount(), size, partitionPath));
            metrics.recordWrite(df.rowCount(), size, System.nanoTime() - t0);
            return this;
        } catch (LakeException e) {
            metrics.recordFailure();
            throw e;
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.ICEBERG, "write",
                    "failed to write Parquet data file for " + lakeTable.fullName(), e);
        }
    }

    @Override
    public void commit() throws LakeException {
        if (aborted.get()) {
            throw new LakeException(LakeFormat.ICEBERG, "commit", "aborted");
        }
        if (!committed.compareAndSet(false, true)) {
            return;
        }
        if (pending.isEmpty()) {
            return;
        }

        long t0 = System.nanoTime();
        Table table = tableHandle.icebergTable();
        try {
            table.refresh();
        } catch (Exception ignored) {
        }

        PartitionSpec spec = table.spec();
        List<DataFile> dataFiles = new ArrayList<>(pending.size());
        long totalRows = 0;
        long totalBytes = 0;

        try {
            for (PendingFile pf : pending) {
                String location = LocalFsFileIO.toLocation(pf.path());
                DataFiles.Builder builder = DataFiles.builder(spec)
                        .withPath(location)
                        .withFormat(FileFormat.PARQUET)
                        .withFileSizeInBytes(pf.fileSizeBytes())
                        .withRecordCount(pf.recordCount())
                        .withMetrics(new Metrics(pf.recordCount()));
                if (pf.partitionPath() != null && !pf.partitionPath().isBlank() && spec.isPartitioned()) {
                    try {
                        builder.withPartitionPath(pf.partitionPath());
                    } catch (Exception ignored) {
                        // unpartitioned or incompatible path — still append file
                    }
                }
                dataFiles.add(builder.build());
                totalRows += pf.recordCount();
                totalBytes += pf.fileSizeBytes();
            }

            if (mode == Mode.UPSERT) {
                throw new LakeException(LakeFormat.ICEBERG, "commit",
                        "UPSERT/MOR not supported in lightweight path; use APPEND or OVERWRITE");
            }

            if (mode == Mode.OVERWRITE) {
                OverwriteFiles overwrite = table.newOverwrite();
                // full-table overwrite when no static partition filter; otherwise row-filter path
                if (staticPartition != null && !staticPartition.isEmpty()) {
                    overwrite.overwriteByRowFilter(IcebergScan.toIcebergExpression(staticPartition));
                } else {
                    overwrite.overwriteByRowFilter(Expressions.alwaysTrue());
                }
                for (DataFile df : dataFiles) {
                    overwrite.addFile(df);
                }
                if (label != null && !label.isBlank()) {
                    overwrite.set("jnitorch.commit.label", label);
                }
                overwrite.commit();
            } else {
                AppendFiles append = table.newAppend();
                for (DataFile df : dataFiles) {
                    append.appendFile(df);
                }
                if (label != null && !label.isBlank()) {
                    append.set("jnitorch.commit.label", label);
                }
                append.commit();
            }

            pending.clear();
            try {
                tableHandle.refresh();
            } catch (Exception ignored) {
            }
            metrics.recordWrite(totalRows, totalBytes, System.nanoTime() - t0);
        } catch (LakeException e) {
            // allow retry: roll back committed flag if commit itself failed before success
            committed.set(false);
            metrics.recordFailure();
            throw e;
        } catch (Exception e) {
            committed.set(false);
            metrics.recordFailure();
            throw new LakeException(LakeFormat.ICEBERG, "commit",
                    "Iceberg commit failed for " + lakeTable.fullName(), e);
        }
    }

    @Override
    public void abort() {
        aborted.set(true);
        deletePendingFiles();
        pending.clear();
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        // no auto-commit — drop uncommitted files
        if (!committed.get()) {
            deletePendingFiles();
        }
        pending.clear();
    }

    private void deletePendingFiles() {
        for (PendingFile pf : pending) {
            try {
                Files.deleteIfExists(pf.path());
            } catch (Exception ignored) {
            }
        }
    }

    /**
     * Build Hive-style partition path from EQ predicates only, e.g. {@code dt=2026-08-01/region=cn}.
     */
    static String partitionPathFromFilter(PartitionFilter filter) {
        if (filter == null || filter.isEmpty()) return null;
        var eq = filter.equalityMap();
        if (eq.isEmpty()) return null;
        StringBuilder sb = new StringBuilder();
        for (var e : eq.entrySet()) {
            if (sb.length() > 0) sb.append('/');
            sb.append(sanitize(e.getKey())).append('=').append(sanitize(e.getValue()));
        }
        return sb.toString();
    }

    private static String sanitize(String s) {
        if (s == null) return "null";
        // keep path-safe characters
        return s.replaceAll("[^A-Za-z0-9._\\-+=]", "_");
    }
}
