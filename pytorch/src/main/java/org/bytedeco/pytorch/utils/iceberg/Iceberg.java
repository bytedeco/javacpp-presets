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

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;

/**
 * Apache Iceberg adapter facade (local warehouse + Parquet, no Hadoop runtime).
 *
 * <p>Implementation notes (per DATA_LAKE_AI_ADAPTERS_PLAN.md §6.2):</p>
 * <ul>
 *   <li>{@link IcebergCatalog} — HadoopTables-style local warehouse via {@link LocalFsTableOperations}</li>
 *   <li>{@link IcebergScan} — {@code TableScan.planFiles()} → {@link DataFrame#readParquet}</li>
 *   <li>{@link IcebergWrite} — write Parquet → {@code AppendFiles}/{@code OverwriteFiles} commit</li>
 *   <li>{@link IcebergStream} — snapshot / {@code IncrementalAppendScan} micro-batches</li>
 * </ul>
 *
 * <pre>{@code
 * IcebergOptions opts = IcebergOptions.builder()
 *     .warehouse("/tmp/warehouse")
 *     .namespaceName("rec")
 *     .table("events")
 *     .build();
 *
 * try (LakeCatalog cat = Iceberg.openCatalog(opts)) {
 *     DataFrame df = Iceberg.scan(opts).columns("user_id", "item_id").collect();
 *     try (LakeWrite w = Iceberg.write(opts)) {
 *         w.mode(LakeWrite.Mode.APPEND).write(df).commit();
 *     }
 *     try (LakeStream s = Iceberg.stream(opts)) {
 *         s.batchRows(4096).forEachBatch(batch -> s.commit());
 *     }
 * }
 * }</pre>
 *
 * @see <a href="https://iceberg.apache.org/">Apache Iceberg</a>
 */
public final class Iceberg {
    private Iceberg() {}

    public static LakeCatalog openCatalog(IcebergOptions options) {
        return IcebergCatalog.open(options);
    }

    public static LakeCatalog openCatalog(LakeOptions options) {
        return IcebergCatalog.open(options);
    }

    public static IcebergCatalog open(IcebergOptions options) {
        return IcebergCatalog.open(options);
    }

    public static LakeTable table(IcebergOptions options) {
        try (IcebergCatalog cat = IcebergCatalog.open(options)) {
            return cat.loadTable(options.namespaceName(), options.table());
        }
    }

    public static LakeScan scan(IcebergOptions options) {
        // Caller owns catalog lifecycle via scan collect; bind catalog to options table
        IcebergCatalog cat = IcebergCatalog.open(options);
        return cat.scan(options.namespaceName(), options.table());
    }

    public static LakeScan scan(LakeOptions options) {
        return scan(IcebergOptions.fromLakeOptions(options));
    }

    public static DataFrame read(IcebergOptions options) {
        try (IcebergCatalog cat = IcebergCatalog.open(options)) {
            return cat.scan(options.namespaceName(), options.table()).collect();
        }
    }

    public static DataFrame read(LakeOptions options) {
        return read(IcebergOptions.fromLakeOptions(options));
    }

    public static LakeWrite write(IcebergOptions options) {
        IcebergCatalog cat = IcebergCatalog.open(options);
        // Wrap so closing the write also closes the owned catalog
        LakeWrite inner = cat.write(options.namespaceName(), options.table());
        return new ClosingWrite(inner, cat);
    }

    public static LakeWrite write(LakeOptions options) {
        return write(IcebergOptions.fromLakeOptions(options));
    }

    public static LakeStream stream(IcebergOptions options) {
        return IcebergStream.open(options);
    }

    public static LakeStream stream(LakeOptions options) {
        return IcebergStream.open(options);
    }

    /** LakeWrite that closes an owned catalog on close/abort. */
    private static final class ClosingWrite implements LakeWrite {
        private final LakeWrite delegate;
        private final IcebergCatalog catalog;

        ClosingWrite(LakeWrite delegate, IcebergCatalog catalog) {
            this.delegate = delegate;
            this.catalog = catalog;
        }

        @Override public LakeTable table() { return delegate.table(); }
        @Override public LakeWrite mode(Mode mode) { delegate.mode(mode); return this; }
        @Override public LakeWrite partition(org.bytedeco.pytorch.utils.lake.PartitionFilter p) {
            delegate.partition(p); return this;
        }
        @Override public LakeWrite label(String label) { delegate.label(label); return this; }
        @Override public LakeWrite write(DataFrame df) { delegate.write(df); return this; }
        @Override public void commit() { delegate.commit(); }
        @Override public void abort() {
            try { delegate.abort(); } finally { catalog.close(); }
        }
        @Override public void close() {
            try { delegate.close(); } finally { catalog.close(); }
        }
    }
}
