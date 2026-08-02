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
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;

/**
 * Apache Hudi adapter facade (timeline + Parquet COW, no Hadoop runtime).
 *
 * <p>Implementation notes (DATA_LAKE_AI_ADAPTERS_PLAN.md §6.4):</p>
 * <ul>
 *   <li>{@link HudiCatalog} — local base path / warehouse layout</li>
 *   <li>{@link HudiTimeline} — {@code .hoodie} commit/deltacommit parse</li>
 *   <li>{@link HudiScan} — plan base parquet → {@link DataFrame#readParquet}</li>
 *   <li>{@link HudiWrite} — write parquet + commit marker</li>
 *   <li>{@link HudiStream} — snapshot / incremental instant micro-batches</li>
 * </ul>
 *
 * <pre>{@code
 * HudiOptions opts = HudiOptions.builder()
 *     .basePath("/tmp/hudi_wh")
 *     .namespaceName("rec")
 *     .table("events")
 *     .build();
 *
 * try (LakeCatalog cat = Hudi.openCatalog(opts)) {
 *     DataFrame df = Hudi.scan(opts).columns("user_id", "item_id").collect();
 *     try (LakeWrite w = Hudi.write(opts)) {
 *         w.mode(LakeWrite.Mode.APPEND).write(df).commit();
 *     }
 *     try (LakeStream s = Hudi.stream(opts)) {
 *         s.batchRows(4096).forEachBatch(batch -> s.commit());
 *     }
 * }
 * }</pre>
 *
 * @see <a href="https://hudi.apache.org/">Apache Hudi</a>
 */
public final class Hudi {
    private Hudi() {}

    public static LakeCatalog openCatalog(HudiOptions options) {
        return HudiCatalog.open(options);
    }

    public static LakeCatalog openCatalog(LakeOptions options) {
        return HudiCatalog.open(options);
    }

    public static HudiCatalog open(HudiOptions options) {
        return HudiCatalog.open(options);
    }

    public static LakeTable table(HudiOptions options) {
        try (HudiCatalog cat = HudiCatalog.open(options)) {
            return cat.loadTable(options.namespaceName(), options.table());
        }
    }

    public static LakeScan scan(HudiOptions options) {
        HudiCatalog cat = HudiCatalog.open(options);
        return cat.scan(options.namespaceName(), options.table());
    }

    public static LakeScan scan(LakeOptions options) {
        return scan(HudiOptions.fromLakeOptions(options));
    }

    public static DataFrame read(HudiOptions options) {
        try (HudiCatalog cat = HudiCatalog.open(options)) {
            return cat.scan(options.namespaceName(), options.table()).collect();
        }
    }

    public static DataFrame read(LakeOptions options) {
        return read(HudiOptions.fromLakeOptions(options));
    }

    public static LakeWrite write(HudiOptions options) {
        HudiCatalog cat = HudiCatalog.open(options);
        LakeWrite inner = cat.write(options.namespaceName(), options.table());
        return new ClosingWrite(inner, cat);
    }

    public static LakeWrite write(LakeOptions options) {
        return write(HudiOptions.fromLakeOptions(options));
    }

    public static LakeStream stream(HudiOptions options) {
        return HudiStream.open(options);
    }

    public static LakeStream stream(LakeOptions options) {
        return HudiStream.open(HudiOptions.fromLakeOptions(options));
    }

    private static final class ClosingWrite implements LakeWrite {
        private final LakeWrite delegate;
        private final HudiCatalog catalog;

        ClosingWrite(LakeWrite delegate, HudiCatalog catalog) {
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
