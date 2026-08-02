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
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.PartitionFilter;

import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Paimon micro-batch stream aligned with {@link LakeStream} / KafkaStream.
 *
 * <p>Modes:</p>
 * <ul>
 *   <li><b>Snapshot</b> — scan all known parquet files once for resolved snapshot.</li>
 *   <li><b>Incremental</b> — detect new snapshots after {@code fromSnapshotId} exclusive;
 *       {@link #commit()} advances watermark (at-least-once).</li>
 * </ul>
 *
 * <p>Full changelog consumption requires optional paimon-core (honest boundary).</p>
 */
public final class PaimonStream implements LakeStream {

    private final PaimonCatalog catalog;
    private final String namespaceName;
    private final String table;
    private final LakeMetrics metrics;
    private final boolean ownCatalog;

    private String[] columns;
    private PartitionFilter partitionFilter;
    private int batchRows;
    private Duration idleStop;
    private long maxBatches = Long.MAX_VALUE;
    private long rowLimit = -1;

    private String fromSnapshotIdExclusive;
    private boolean incremental;

    private final AtomicBoolean stopped = new AtomicBoolean(false);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final AtomicReference<String> committedSnapshot = new AtomicReference<>(null);

    private Iterator<Path> fileIt;
    private List<PaimonSnapshot.Snapshot> pendingSnapshots;
    private int snapshotIdx;
    private boolean primed;
    private long rowsEmitted;
    private long batchesEmitted;
    private String lastSeenSnapshotId;
    private long lastSeenSnapshotTime;
    private PaimonSnapshot snapshotMeta;

    PaimonStream(PaimonCatalog catalog, String namespaceName, String table) {
        this(catalog, namespaceName, table, false);
    }

    PaimonStream(PaimonCatalog catalog, String namespaceName, String table, boolean ownCatalog) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.namespaceName = namespaceName;
        this.table = table;
        this.metrics = catalog.metrics();
        this.ownCatalog = ownCatalog;
        this.batchRows = Math.max(1, catalog.options().batchRows());
        this.idleStop = catalog.options().idleStop();
        this.columns = catalog.options().columns();
        this.partitionFilter = catalog.options().partitionFilter();
        this.fromSnapshotIdExclusive = catalog.options().fromSnapshotId();
        if (fromSnapshotIdExclusive != null) {
            this.incremental = true;
        }
    }

    public static PaimonStream open(PaimonOptions options) {
        PaimonCatalog cat = PaimonCatalog.open(options);
        return new PaimonStream(cat, options.namespaceName(), options.table(), true);
    }

    public PaimonStream columns(String... columns) {
        this.columns = columns;
        return this;
    }

    public PaimonStream filter(PartitionFilter filter) {
        this.partitionFilter = filter;
        return this;
    }

    public PaimonStream fromSnapshot(String snapshotIdExclusive) {
        this.fromSnapshotIdExclusive = snapshotIdExclusive;
        this.incremental = true;
        return this;
    }

    public PaimonStream incremental(boolean enabled) {
        this.incremental = enabled;
        return this;
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

    public PaimonStream limit(long maxRows) {
        this.rowLimit = maxRows;
        return this;
    }

    @Override
    public void commit() throws LakeException {
        if (lastSeenSnapshotId != null) {
            committedSnapshot.set(lastSeenSnapshotId);
            fromSnapshotIdExclusive = lastSeenSnapshotId;
        }
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
        Path tablePath = catalog.resolveTablePath(namespaceName, table);
        snapshotMeta = PaimonSnapshot.load(tablePath);

        if (incremental) {
            long fromId = 0;
            try {
                fromId = Long.parseLong(fromSnapshotIdExclusive);
            } catch (NumberFormatException ignored) {
            }
            pendingSnapshots = snapshotMeta.after(fromId);
            snapshotIdx = 0;
            fileIt = List.<Path>of().iterator();
            advanceSnapshotFiles(tablePath);
        } else {
            PaimonScan scan = (PaimonScan) catalog.scan(namespaceName, table);
            if (columns != null) scan.columns(columns);
            if (partitionFilter != null) scan.filter(partitionFilter);
            fileIt = scan.planDataFiles().iterator();
            pendingSnapshots = List.of();
            PaimonSnapshot.Snapshot latest = snapshotMeta.latest();
            if (latest != null) {
                lastSeenSnapshotId = String.valueOf(latest.id());
                lastSeenSnapshotTime = latest.timeMillis();
            }
        }
        primed = true;
    }

    private void advanceSnapshotFiles(Path tablePath) {
        while (snapshotIdx < pendingSnapshots.size()) {
            PaimonSnapshot.Snapshot snap = pendingSnapshots.get(snapshotIdx++);
            lastSeenSnapshotId = String.valueOf(snap.id());
            lastSeenSnapshotTime = snap.timeMillis();
            List<Path> files = new ArrayList<>();
            for (String hint : snap.dataFileHints()) {
                Path p = resolveDataPath(hint);
                if (p != null && java.nio.file.Files.isRegularFile(p)) {
                    String part = PaimonSnapshot.partitionPathOf(tablePath, p);
                    if (PaimonSnapshot.partitionMatches(part, partitionFilter)) {
                        files.add(p.toAbsolutePath().normalize());
                    }
                }
            }
            if (files.isEmpty()) {
                // fall back to all parquet in table (covers hint-less snapshots)
                for (Path p : PaimonSnapshot.discoverParquetFiles(tablePath)) {
                    String part = PaimonSnapshot.partitionPathOf(tablePath, p);
                    if (PaimonSnapshot.partitionMatches(part, partitionFilter)) {
                        files.add(p);
                    }
                }
            }
            if (!files.isEmpty()) {
                fileIt = files.iterator();
                return;
            }
        }
        fileIt = List.<Path>of().iterator();
    }

    private Path resolveDataPath(String raw) {
        if (raw == null || raw.isBlank()) return null;
        String s = raw.startsWith("file://") ? raw.substring(7) : raw;
        Path p = Path.of(s);
        if (p.isAbsolute() && java.nio.file.Files.isRegularFile(p)) {
            return p.toAbsolutePath().normalize();
        }
        Path tablePath = catalog.resolveTablePath(namespaceName, table);
        Path rel = tablePath.resolve(s).toAbsolutePath().normalize();
        return java.nio.file.Files.isRegularFile(rel) ? rel
                : (java.nio.file.Files.isRegularFile(p) ? p.toAbsolutePath().normalize() : rel);
    }

    @Override
    public DataFrame poll() throws LakeException {
        if (isStopped()) return null;
        try {
            ensurePlan();
            Path tablePath = catalog.resolveTablePath(namespaceName, table);

            List<DataFrame> acc = new ArrayList<>();
            long accRows = 0;
            while (accRows < batchRows) {
                if (rowLimit >= 0 && rowsEmitted + accRows >= rowLimit) break;
                if (fileIt == null || !fileIt.hasNext()) {
                    if (incremental && snapshotIdx < pendingSnapshots.size()) {
                        advanceSnapshotFiles(tablePath);
                        if (fileIt == null || !fileIt.hasNext()) continue;
                    } else {
                        break;
                    }
                }
                Path f = fileIt.next();
                DataFrame df = readFile(f);
                if (df == null || df.rowCount() == 0) continue;
                acc.add(df);
                accRows += df.rowCount();
            }

            if (acc.isEmpty()) {
                if (!incremental) stopped.set(true);
                return null;
            }
            DataFrame out = acc.size() == 1 ? acc.get(0) : PaimonScan.vstack(acc);
            if (rowLimit >= 0 && rowsEmitted + out.rowCount() > rowLimit) {
                long keep = rowLimit - rowsEmitted;
                out = out.limit((int) Math.min(Integer.MAX_VALUE, keep));
            }
            rowsEmitted += out.rowCount();
            batchesEmitted++;
            metrics.recordBatch(out.rowCount());
            return out;
        } catch (LakeException e) {
            metrics.recordFailure();
            throw e;
        } catch (Exception e) {
            metrics.recordFailure();
            throw new LakeException(LakeFormat.PAIMON, "stream.poll", "Paimon stream failed", e);
        }
    }

    private DataFrame readFile(Path file) throws Exception {
        String path = file.toAbsolutePath().normalize().toString();
        if (columns != null && columns.length > 0) {
            try {
                return DataFrame.readParquet(path, columns);
            } catch (Exception e) {
                DataFrame df = DataFrame.readParquet(path);
                try {
                    return df.select(columns);
                } catch (Exception ignored) {
                    return df;
                }
            }
        }
        return DataFrame.readParquet(path);
    }

    public String committedSnapshot() {
        return committedSnapshot.get();
    }

    public long lastSeenSnapshotTime() {
        return lastSeenSnapshotTime;
    }

    @Override
    public void close() {
        if (!closed.compareAndSet(false, true)) return;
        stopped.set(true);
        if (ownCatalog) {
            try {
                catalog.close();
            } catch (Exception ignored) {
            }
        }
    }
}
