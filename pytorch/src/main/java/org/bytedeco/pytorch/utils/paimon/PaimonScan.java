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

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeException;
import org.bytedeco.pytorch.utils.lake.LakeFormat;
import org.bytedeco.pytorch.utils.lake.LakeMetrics;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;
import org.bytedeco.pytorch.utils.lake.ReplicaPolicy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Paimon batch scan → {@link DataFrame} via snapshot metadata + Parquet data files.
 */
public final class PaimonScan implements LakeScan {

    private final PaimonCatalog catalog;
    private final Path tablePath;
    private final PaimonSnapshot meta;
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

    PaimonScan(PaimonCatalog catalog, Path tablePath, PaimonSnapshot meta, LakeTable lakeTable) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.tablePath = Objects.requireNonNull(tablePath, "tablePath");
        this.meta = Objects.requireNonNull(meta, "meta");
        this.lakeTable = Objects.requireNonNull(lakeTable, "lakeTable");
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

    Path tablePath() { return tablePath; }
    PaimonSnapshot meta() { return meta; }

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

    public List<Path> planDataFiles() {
        PaimonSnapshot.Snapshot snap = meta.resolve(snapshotId, asOfTimeMs);
        Set<Path> files = new LinkedHashSet<>();
        if (snap != null) {
            for (String hint : snap.dataFileHints()) {
                Path p = resolveDataPath(hint);
                if (p != null && Files.isRegularFile(p)) files.add(p);
            }
        }
        if (files.isEmpty()) {
            files.addAll(PaimonSnapshot.discoverParquetFiles(tablePath));
        }
        List<Path> planned = new ArrayList<>();
        for (Path f : files) {
            String part = PaimonSnapshot.partitionPathOf(tablePath, f);
            if (!PaimonSnapshot.partitionMatches(part, partitionFilter)) continue;
            planned.add(f);
        }
        planned.sort(Path::compareTo);
        return planned;
    }

    private Path resolveDataPath(String raw) {
        if (raw == null || raw.isBlank()) return null;
        String s = raw.startsWith("file://") ? raw.substring(7) : raw;
        Path p = Path.of(s);
        if (p.isAbsolute() && Files.isRegularFile(p)) return p.toAbsolutePath().normalize();
        Path rel = tablePath.resolve(s).toAbsolutePath().normalize();
        return Files.isRegularFile(rel) ? rel : (Files.isRegularFile(p) ? p.toAbsolutePath().normalize() : rel);
    }

    @Override
    public DataFrame collect() throws LakeException {
        long t0 = System.nanoTime();
        try {
            List<Path> files = planDataFiles();
            List<DataFrame> parts = new ArrayList<>();
            long rows = 0;
            for (Path f : files) {
                DataFrame df = readFile(f);
                if (df == null || df.rowCount() == 0) continue;
                parts.add(df);
                rows += df.rowCount();
                if (limit >= 0 && rows >= limit) break;
            }
            DataFrame out;
            if (parts.isEmpty()) {
                out = emptyFrame();
            } else if (parts.size() == 1) {
                out = parts.get(0);
            } else {
                out = vstack(parts);
            }
            if (limit >= 0 && out.rowCount() > limit) {
                out = out.limit((int) Math.min(Integer.MAX_VALUE, limit));
            }
            if (columns != null && columns.length > 0) {
                try {
                    out = out.select(columns);
                } catch (Exception ignored) {
                }
            }
            metrics.recordRead(out.rowCount(), System.nanoTime() - t0);
            return out;
        } catch (LakeException e) {
            metrics.recordFailure();
            throw e;
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.PAIMON, "scan.collect",
                    "failed Paimon scan for " + lakeTable.fullName(), e);
        }
    }

    DataFrame readFile(Path file) {
        String path = file.toAbsolutePath().normalize().toString();
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
            throw new LakeException(LakeFormat.PAIMON, "scan.readFile", "failed to read " + path, e);
        }
    }

    private DataFrame emptyFrame() {
        DataFrame df = DataFrame.create();
        String[] names = columns != null && columns.length > 0
                ? columns
                : lakeTable.schema().names();
        for (String name : names) {
            var f = lakeTable.schema().get(name);
            Column.DType dt = f != null ? f.dtype() : Column.DType.STRING;
            df.addColumn(name, dt);
        }
        return df;
    }

    static DataFrame vstack(List<DataFrame> parts) {
        if (parts == null || parts.isEmpty()) return DataFrame.create();
        if (parts.size() == 1) return parts.get(0);
        try {
            return DataFrame.vstack(parts);
        } catch (Exception e) {
            try {
                return DataFrame.concat(parts, 0);
            } catch (Exception e2) {
                throw new LakeException(LakeFormat.PAIMON, "scan.vstack",
                        "failed to concatenate " + parts.size() + " frames", e2);
            }
        }
    }

    @Override
    public LakeStream stream() throws LakeException {
        return new PaimonFileStream(this, batchRows, catalog.options().idleStop(), limit);
    }

    static final class PaimonFileStream implements LakeStream {
        private final PaimonScan scan;
        private int batchRows;
        private Duration idleStop;
        private final long rowLimit;
        private final AtomicBoolean stopped = new AtomicBoolean(false);
        private final AtomicBoolean closed = new AtomicBoolean(false);
        private Iterator<Path> fileIt;
        private long rowsEmitted;
        private long batchesEmitted;
        private long maxBatches = Long.MAX_VALUE;
        private boolean primed;

        PaimonFileStream(PaimonScan scan, int batchRows, Duration idleStop, long rowLimit) {
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
        public void commit() {}

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
            fileIt = scan.planDataFiles().iterator();
            primed = true;
        }

        @Override
        public DataFrame poll() throws LakeException {
            if (isStopped()) return null;
            ensurePlan();
            if (fileIt == null || !fileIt.hasNext()) {
                stopped.set(true);
                return null;
            }
            List<DataFrame> acc = new ArrayList<>();
            long accRows = 0;
            while (fileIt.hasNext() && accRows < batchRows) {
                if (rowLimit >= 0 && rowsEmitted + accRows >= rowLimit) break;
                Path f = fileIt.next();
                DataFrame df = scan.readFile(f);
                if (df == null || df.rowCount() == 0) continue;
                acc.add(df);
                accRows += df.rowCount();
            }
            if (acc.isEmpty()) {
                stopped.set(true);
                return null;
            }
            DataFrame out = acc.size() == 1 ? acc.get(0) : vstack(acc);
            if (rowLimit >= 0 && rowsEmitted + out.rowCount() > rowLimit) {
                long keep = rowLimit - rowsEmitted;
                out = out.limit((int) Math.min(Integer.MAX_VALUE, keep));
            }
            rowsEmitted += out.rowCount();
            batchesEmitted++;
            scan.metrics.recordBatch(out.rowCount());
            return out;
        }

        @Override
        public void close() {
            closed.set(true);
            stopped.set(true);
        }
    }
}
