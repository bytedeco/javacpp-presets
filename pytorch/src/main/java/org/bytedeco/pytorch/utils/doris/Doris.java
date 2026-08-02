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
package org.bytedeco.pytorch.utils.doris;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;

import java.util.Map;
import java.util.Objects;

/**
 * Apache Doris facade — MySQL-protocol query + HTTP Stream Load + Lake SPI.
 *
 * <pre>{@code
 * DorisOptions opts = DorisOptions.builder()
 *     .feHost("fe1").queryPort(9030).httpPort(8030)
 *     .database("rec").table("user_features")
 *     .username("root").password("")
 *     .build();
 *
 * // Query
 * DataFrame df = Doris.query(opts, "SELECT * FROM rec.user_features WHERE user_id = 1");
 *
 * // High-throughput write
 * Doris.streamLoad(opts, df);
 *
 * // Lake catalog
 * try (DorisCatalog cat = Doris.open(opts)) {
 *     LakeTable t = cat.loadTable("rec", "user_features");
 *     DataFrame batch = cat.scan("rec", "events")
 *         .filter(PartitionFilter.eq("dt", "2026-08-01"))
 *         .limit(100_000)
 *         .collect();
 * }
 * }</pre>
 *
 * @see <a href="https://doris.apache.org/">Apache Doris</a>
 * @see DorisOptions
 * @see DorisCatalog
 * @see DorisStreamLoad
 */
public final class Doris {
    private Doris() {}

    public static DorisCatalog open(DorisOptions options) {
        return DorisCatalog.open(options);
    }

    public static DorisCatalog open(LakeOptions lakeOptions) {
        return open(DorisOptions.fromLakeOptions(lakeOptions));
    }

    public static DorisCatalog open(String feHost, String database) {
        return open(DorisOptions.builder().feHost(feHost).database(database).build());
    }

    public static DorisPool pool(DorisOptions options) {
        return DorisPool.open(options);
    }

    /** Run SQL and materialize a DataFrame. */
    public static DataFrame query(DorisOptions options, String sql) {
        try (DorisCatalog cat = open(options)) {
            return cat.query(sql);
        }
    }

    public static DataFrame query(String jdbcOrDorisUri, String sql) {
        return query(DorisOptions.fromUri(jdbcOrDorisUri), sql);
    }

    /** Read full table (prefer {@link #scan} for large tables). */
    public static DataFrame readTable(DorisOptions options) {
        Objects.requireNonNull(options.database(), "database");
        Objects.requireNonNull(options.table(), "table");
        try (DorisCatalog cat = open(options)) {
            return cat.read(options.database(), options.table());
        }
    }

    public static DataFrame pointQuery(DorisOptions options, Map<String, Object> keys, String... columns) {
        try (DorisCatalog cat = open(options)) {
            return cat.pointQuery(options.database(), options.table(), keys, columns);
        }
    }

    /** High-throughput Stream Load of a DataFrame. */
    public static DorisStreamLoad.Result streamLoad(DorisOptions options, DataFrame df) {
        try (DorisStreamLoad loader = new DorisStreamLoad(options)) {
            return loader.load(df);
        }
    }

    public static DorisStreamLoad.Result streamLoad(DorisOptions options, DataFrame df, String label) {
        try (DorisStreamLoad loader = new DorisStreamLoad(options)) {
            return loader.load(df, label);
        }
    }

    /** Append DataFrame via Stream Load using options.database/table. */
    public static void write(DorisOptions options, DataFrame df) {
        streamLoad(options, df);
    }

    public static LakeStream stream(DorisOptions options) {
        Objects.requireNonNull(options.database(), "database");
        Objects.requireNonNull(options.table(), "table");
        // Caller owns catalog lifecycle via stream.close → connection release;
        // for short-lived API we open catalog and let stream hold it.
        DorisCatalog cat = open(options);
        LakeStream s = cat.stream(options.database(), options.table());
        return new ClosingStream(s, cat);
    }

    public static LakeWrite writer(DorisOptions options) {
        DorisCatalog cat = open(options);
        LakeWrite w = cat.write(options.database(), options.table());
        return new ClosingWrite(w, cat);
    }

    public static LakeTable table(DorisOptions options) {
        try (DorisCatalog cat = open(options)) {
            return cat.loadTable(options.database(), options.table());
        }
    }

    public static DorisCatalog.DorisScan scan(DorisOptions options) {
        DorisCatalog cat = open(options);
        // scan is not AutoCloseable; use catalog.scan after open for long sessions
        return (DorisCatalog.DorisScan) cat.scan(options.database(), options.table());
    }

    /** Generate CREATE TABLE DDL (does not execute). */
    public static String createTableDdl(DorisOptions options,
                                        org.bytedeco.pytorch.utils.lake.LakeSchema schema,
                                        org.bytedeco.pytorch.utils.lake.PartitionSpec partitionSpec) {
        try (DorisCatalog cat = open(options)) {
            return cat.buildCreateTableDdl(options.database(), options.table(), schema, partitionSpec, null);
        }
    }

    private static final class ClosingStream implements LakeStream {
        private final LakeStream delegate;
        private final LakeCatalog catalog;

        ClosingStream(LakeStream delegate, LakeCatalog catalog) {
            this.delegate = delegate;
            this.catalog = catalog;
        }

        @Override public LakeStream batchRows(int batchRows) { delegate.batchRows(batchRows); return this; }
        @Override public LakeStream idleStop(java.time.Duration idle) { delegate.idleStop(idle); return this; }
        @Override public LakeStream maxBatches(long maxBatches) { delegate.maxBatches(maxBatches); return this; }
        @Override public void commit() { delegate.commit(); }
        @Override public void stop() { delegate.stop(); }
        @Override public boolean isStopped() { return delegate.isStopped(); }
        @Override public DataFrame poll() { return delegate.poll(); }
        @Override public void close() {
            try { delegate.close(); } finally { catalog.close(); }
        }
    }

    private static final class ClosingWrite implements LakeWrite {
        private final LakeWrite delegate;
        private final LakeCatalog catalog;

        ClosingWrite(LakeWrite delegate, LakeCatalog catalog) {
            this.delegate = delegate;
            this.catalog = catalog;
        }

        @Override public LakeTable table() { return delegate.table(); }
        @Override public LakeWrite mode(Mode mode) { delegate.mode(mode); return this; }
        @Override public LakeWrite partition(org.bytedeco.pytorch.utils.lake.PartitionFilter staticPartition) {
            delegate.partition(staticPartition); return this;
        }
        @Override public LakeWrite label(String label) { delegate.label(label); return this; }
        @Override public LakeWrite write(DataFrame df) { delegate.write(df); return this; }
        @Override public void commit() { delegate.commit(); }
        @Override public void abort() { delegate.abort(); }
        @Override public void close() {
            try { delegate.close(); } finally { catalog.close(); }
        }
    }
}
