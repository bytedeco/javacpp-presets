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
 * Paimon light write: buffer DataFrames → Parquet under table path → snapshot marker.
 *
 * <p>Supports APPEND and OVERWRITE. Full primary-key UPSERT / changelog merge
 * requires the optional paimon-core client (honest boundary).</p>
 */
public final class PaimonWrite implements LakeWrite {

    private final PaimonCatalog catalog;
    private final Path tablePath;
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

    PaimonWrite(PaimonCatalog catalog, Path tablePath, LakeTable lakeTable) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.tablePath = Objects.requireNonNull(tablePath, "tablePath");
        this.lakeTable = Objects.requireNonNull(lakeTable, "lakeTable");
        this.metrics = catalog.metrics();
    }

    @Override
    public LakeTable table() {
        return lakeTable;
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
        if (aborted.get()) throw new LakeException(LakeFormat.PAIMON, "write", "aborted");
        if (committed.get()) throw new LakeException(LakeFormat.PAIMON, "write", "already committed");
        if (closed.get()) throw new LakeException(LakeFormat.PAIMON, "write", "closed");
        Objects.requireNonNull(df, "dataframe");
        if (df.rowCount() == 0) return this;

        long t0 = System.nanoTime();
        try {
            Files.createDirectories(tablePath);

            String partitionPath = partitionPathFromFilter(staticPartition);
            Path dir = tablePath;
            if (partitionPath != null && !partitionPath.isBlank()) {
                dir = tablePath.resolve(partitionPath);
                Files.createDirectories(dir);
            }

            String fileName = (label != null && !label.isBlank() ? sanitize(label) + "-" : "")
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
            throw new LakeException(LakeFormat.PAIMON, "write",
                    "failed to write Parquet for " + lakeTable.fullName(), e);
        }
    }

    @Override
    public void commit() throws LakeException {
        if (aborted.get()) throw new LakeException(LakeFormat.PAIMON, "commit", "aborted");
        if (!committed.compareAndSet(false, true)) return;
        if (pending.isEmpty()) return;

        long t0 = System.nanoTime();
        try {
            if (mode == Mode.UPSERT) {
                committed.set(false);
                throw new LakeException(LakeFormat.PAIMON, "commit",
                        "UPSERT/changelog full merge not supported in lightweight path; use APPEND or OVERWRITE");
            }

            if (mode == Mode.OVERWRITE) {
                List<Path> existing = PaimonSnapshot.discoverParquetFiles(tablePath);
                for (Path old : existing) {
                    boolean isPending = false;
                    for (PendingFile pf : pending) {
                        if (pf.path().equals(old)) {
                            isPending = true;
                            break;
                        }
                    }
                    if (!isPending) {
                        try {
                            Files.deleteIfExists(old);
                        } catch (Exception ignored) {
                        }
                    }
                }
            }

            long totalRows = 0;
            long totalBytes = 0;
            List<Path> files = new ArrayList<>(pending.size());
            for (PendingFile pf : pending) {
                files.add(pf.path());
                totalRows += pf.recordCount();
                totalBytes += pf.fileSizeBytes();
            }

            long snapId = PaimonSnapshot.nextSnapshotId(tablePath);
            PaimonSnapshot.writeSnapshot(tablePath, snapId, files, totalRows);
            pending.clear();
            metrics.recordWrite(totalRows, totalBytes, System.nanoTime() - t0);
        } catch (LakeException e) {
            committed.set(false);
            metrics.recordFailure();
            throw e;
        } catch (Exception e) {
            committed.set(false);
            metrics.recordFailure();
            throw new LakeException(LakeFormat.PAIMON, "commit",
                    "failed to commit Paimon snapshot for " + lakeTable.fullName(), e);
        }
    }

    @Override
    public void abort() {
        aborted.set(true);
        for (PendingFile pf : pending) {
            try {
                Files.deleteIfExists(pf.path());
            } catch (Exception ignored) {
            }
        }
        pending.clear();
    }

    @Override
    public void close() {
        closed.set(true);
    }

    static String partitionPathFromFilter(PartitionFilter filter) {
        if (filter == null || filter.isEmpty()) return null;
        StringBuilder sb = new StringBuilder();
        for (PartitionFilter.Predicate p : filter.predicates()) {
            if (p.op() != PartitionFilter.Op.EQ || p.values().isEmpty()) continue;
            if (sb.length() > 0) sb.append('/');
            sb.append(p.column()).append('=').append(p.values().get(0));
        }
        return sb.length() == 0 ? null : sb.toString();
    }

    static String sanitize(String s) {
        return s.replaceAll("[^a-zA-Z0-9._-]", "_");
    }
}
