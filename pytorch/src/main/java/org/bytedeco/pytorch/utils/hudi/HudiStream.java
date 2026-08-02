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
package org.bytedeco.pytorch.utils.hudi;

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
import java.util.concurrent.atomic.AtomicReference;

/**
 * Hudi micro-batch stream aligned with {@link LakeStream} / KafkaStream.
 *
 * <p>Modes:</p>
 * <ul>
 *   <li><b>Snapshot</b> — scan base parquet once for resolved instant.</li>
 *   <li><b>Incremental</b> — commit instants after {@code fromInstant} exclusive;
 *       {@link #commit()} advances watermark (at-least-once).</li>
 * </ul>
 */
public final class HudiStream implements LakeStream {

    private final HudiCatalog catalog;
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

    private String fromInstantExclusive;
    private String toInstantInclusive;
    private boolean incremental;

    private final AtomicBoolean stopped = new AtomicBoolean(false);
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final AtomicReference<String> committedInstant = new AtomicReference<>(null);

    private Iterator<Path> fileIt;
    private List<HudiTimeline.Instant> pendingInstants;
    private int instantIdx;
    private boolean primed;
    private long rowsEmitted;
    private long batchesEmitted;
    private String lastSeenInstant;

    HudiStream(HudiCatalog catalog, String namespaceName, String table) {
        this(catalog, namespaceName, table, false);
    }

    HudiStream(HudiCatalog catalog, String namespaceName, String table, boolean ownCatalog) {
        this.catalog = Objects.requireNonNull(catalog, "catalog");
        this.namespaceName = namespaceName;
        this.table = table;
        this.metrics = catalog.metrics();
        this.ownCatalog = ownCatalog;
        this.batchRows = Math.max(1, catalog.options().batchRows());
        this.idleStop = catalog.options().idleStop();
        this.columns = catalog.options().columns();
        this.partitionFilter = catalog.options().partitionFilter();
        this.fromInstantExclusive = catalog.options().fromInstantTime();
        if (fromInstantExclusive != null) {
            this.incremental = true;
        }
    }

    public static HudiStream open(HudiOptions options) {
        HudiCatalog cat = HudiCatalog.open(options);
        return new HudiStream(cat, options.namespaceName(), options.table(), true);
    }

    public HudiStream columns(String... columns) {
        this.columns = columns;
        return this;
    }

    public HudiStream filter(PartitionFilter filter) {
        this.partitionFilter = filter;
        return this;
    }

    public HudiStream fromInstant(String instantExclusive) {
        this.fromInstantExclusive = instantExclusive;
        this.incremental = true;
        return this;
    }

    public HudiStream toInstant(String instantInclusive) {
        this.toInstantInclusive = instantInclusive;
        return this;
    }

    public HudiStream incremental(boolean enabled) {
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

    public HudiStream limit(long maxRows) {
        this.rowLimit = maxRows;
        return this;
    }

    @Override
    public void commit() throws LakeException {
        if (lastSeenInstant != null) {
            committedInstant.set(lastSeenInstant);
            fromInstantExclusive = lastSeenInstant;
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
        HudiTimeline timeline = HudiTimeline.load(tablePath);

        if (incremental) {
            pendingInstants = timeline.instantsAfter(fromInstantExclusive, toInstantInclusive);
            instantIdx = 0;
            fileIt = List.<Path>of().iterator();
            advanceInstantFiles(tablePath, timeline);
        } else {
            HudiScan scan = (HudiScan) catalog.scan(namespaceName, table);
            if (columns != null) scan.columns(columns);
            if (partitionFilter != null) scan.filter(partitionFilter);
            fileIt = scan.planDataFiles().iterator();
            pendingInstants = List.of();
            HudiTimeline.Instant latest = timeline.latestCompleted();
            if (latest != null) lastSeenInstant = latest.instantTime();
        }
        primed = true;
    }

    private void advanceInstantFiles(Path tablePath, HudiTimeline timeline) {
        while (instantIdx < pendingInstants.size()) {
            HudiTimeline.Instant inst = pendingInstants.get(instantIdx++);
            lastSeenInstant = inst.instantTime();
            List<String> meta = timeline.dataFilesFromCommit(inst);
            List<Path> files = new ArrayList<>();
            if (!meta.isEmpty()) {
                for (String m : meta) {
                    Path p = Path.of(m.startsWith("file://") ? m.substring(7) : m);
                    if (!p.isAbsolute()) p = tablePath.resolve(m);
                    if (java.nio.file.Files.isRegularFile(p)) {
                        String part = HudiTimeline.partitionPathOf(tablePath, p);
                        if (HudiTimeline.partitionMatches(part, partitionFilter)) {
                            files.add(p.toAbsolutePath().normalize());
                        }
                    }
                }
            }
            if (files.isEmpty() && instantIdx >= pendingInstants.size()) {
                for (Path p : timeline.discoverParquetFiles()) {
                    String part = HudiTimeline.partitionPathOf(tablePath, p);
                    if (HudiTimeline.partitionMatches(part, partitionFilter)) files.add(p);
                }
            }
            if (!files.isEmpty()) {
                fileIt = files.iterator();
                return;
            }
        }
        fileIt = List.<Path>of().iterator();
    }

    @Override
    public DataFrame poll() throws LakeException {
        if (isStopped()) return null;
        try {
            ensurePlan();
            Path tablePath = catalog.resolveTablePath(namespaceName, table);
            HudiTimeline timeline = HudiTimeline.load(tablePath);

            List<DataFrame> acc = new ArrayList<>();
            long accRows = 0;
            while (accRows < batchRows) {
                if (rowLimit >= 0 && rowsEmitted + accRows >= rowLimit) break;
                if (fileIt == null || !fileIt.hasNext()) {
                    if (incremental && instantIdx < pendingInstants.size()) {
                        advanceInstantFiles(tablePath, timeline);
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
            DataFrame out = acc.size() == 1 ? acc.get(0) : HudiScan.vstack(acc);
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
            throw new LakeException(LakeFormat.HUDI, "stream.poll", "Hudi stream failed", e);
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

    public String committedInstant() {
        return committedInstant.get();
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
