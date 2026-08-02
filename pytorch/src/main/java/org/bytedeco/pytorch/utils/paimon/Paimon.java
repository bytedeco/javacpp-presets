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
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;

/**
 * Apache Paimon adapter facade (schema/snapshot + Parquet, no paimon-core runtime).
 *
 * <p>Implementation notes (DATA_LAKE_AI_ADAPTERS_PLAN.md §6.3):</p>
 * <ul>
 *   <li>{@link PaimonCatalog} — local warehouse with schema/snapshot layout</li>
 *   <li>{@link PaimonSnapshot} — parse snapshot JSON + schema metadata</li>
 *   <li>{@link PaimonScan} — plan parquet files → {@link DataFrame#readParquet}</li>
 *   <li>{@link PaimonWrite} — write parquet + bump snapshot marker files</li>
 *   <li>{@link PaimonStream} — snapshot incremental micro-batches (at-least-once)</li>
 * </ul>
 *
 * <pre>{@code
 * PaimonOptions opts = PaimonOptions.builder()
 *     .warehouse("/tmp/paimon_wh")
 *     .namespaceName("rec")
 *     .table("events")
 *     .build();
 *
 * try (LakeCatalog cat = Paimon.openCatalog(opts)) {
 *     DataFrame df = Paimon.scan(opts).columns("user_id", "item_id").collect();
 *     try (LakeWrite w = Paimon.write(opts)) {
 *         w.mode(LakeWrite.Mode.APPEND).write(df).commit();
 *     }
 *     try (LakeStream s = Paimon.stream(opts)) {
 *         s.batchRows(4096).forEachBatch(batch -> s.commit());
 *     }
 * }
 * }</pre>
 *
 * @see <a href="https://paimon.apache.org/">Apache Paimon</a>
 */
public final class Paimon {
    private Paimon() {}

    public static LakeCatalog openCatalog(PaimonOptions options) {
        return PaimonCatalog.open(options);
    }

    public static LakeCatalog openCatalog(LakeOptions options) {
        return PaimonCatalog.open(options);
    }

    public static PaimonCatalog open(PaimonOptions options) {
        return PaimonCatalog.open(options);
    }

    public static LakeTable table(PaimonOptions options) {
        try (PaimonCatalog cat = PaimonCatalog.open(options)) {
            return cat.loadTable(options.namespaceName(), options.table());
        }
    }

    public static LakeScan scan(PaimonOptions options) {
        PaimonCatalog cat = PaimonCatalog.open(options);
        return cat.scan(options.namespaceName(), options.table());
    }

    public static LakeScan scan(LakeOptions options) {
        return scan(PaimonOptions.fromLakeOptions(options));
    }

    public static DataFrame read(PaimonOptions options) {
        try (PaimonCatalog cat = PaimonCatalog.open(options)) {
            return cat.scan(options.namespaceName(), options.table()).collect();
        }
    }

    public static DataFrame read(LakeOptions options) {
        return read(PaimonOptions.fromLakeOptions(options));
    }

    public static LakeWrite write(PaimonOptions options) {
        PaimonCatalog cat = PaimonCatalog.open(options);
        LakeWrite inner = cat.write(options.namespaceName(), options.table());
        return new ClosingWrite(inner, cat);
    }

    public static LakeWrite write(LakeOptions options) {
        return write(PaimonOptions.fromLakeOptions(options));
    }

    public static LakeStream stream(PaimonOptions options) {
        return PaimonStream.open(options);
    }

    public static LakeStream stream(LakeOptions options) {
        return PaimonStream.open(PaimonOptions.fromLakeOptions(options));
    }

    private static final class ClosingWrite implements LakeWrite {
        private final LakeWrite delegate;
        private final PaimonCatalog catalog;

        ClosingWrite(LakeWrite delegate, PaimonCatalog catalog) {
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
