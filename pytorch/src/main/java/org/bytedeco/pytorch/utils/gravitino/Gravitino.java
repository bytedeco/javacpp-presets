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
package org.bytedeco.pytorch.utils.gravitino;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeScan;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;

/**
 * Apache Gravitino adapter facade (REST federation client, no server embed).
 *
 * <p>Implementation notes (DATA_LAKE_AI_ADAPTERS_PLAN.md §6.5):</p>
 * <ul>
 *   <li>{@link GravitinoMetalake} — JDK HttpClient + Gson REST / mock registry</li>
 *   <li>{@link GravitinoResolver} — provider/location → LakeFormat + LakeOptions</li>
 *   <li>{@link GravitinoCatalog} — LakeCatalog that delegates to backend adapters</li>
 * </ul>
 *
 * <pre>{@code
 * // Offline mock for tests
 * GravitinoOptions opts = GravitinoOptions.builder()
 *     .mockRegistryPath("/tmp/gravitino-mock")
 *     .metalake("ml")
 *     .catalogName("lake")
 *     .schemaName("rec")
 *     .table("events")
 *     .build();
 * try (GravitinoCatalog cat = Gravitino.open(opts)) {
 *     cat.metalake().registerMockTable("ml.lake.rec.events", "lakehouse-iceberg",
 *         "/tmp/warehouse/rec/events", Map.of("format", "iceberg"));
 *     DataFrame df = cat.scan("lake.rec", "events").collect();
 * }
 * }</pre>
 *
 * @see <a href="https://gravitino.apache.org/">Apache Gravitino</a>
 * @see <a href="https://datastrato.ai/">Datastrato</a>
 */
public final class Gravitino {
    private Gravitino() {}

    public static LakeCatalog openCatalog(GravitinoOptions options) {
        return GravitinoCatalog.open(options);
    }

    public static LakeCatalog openCatalog(LakeOptions options) {
        return GravitinoCatalog.open(options);
    }

    public static GravitinoCatalog open(GravitinoOptions options) {
        return GravitinoCatalog.open(options);
    }

    public static LakeTable table(GravitinoOptions options) {
        try (GravitinoCatalog cat = GravitinoCatalog.open(options)) {
            return cat.loadTable(ns(options), options.table());
        }
    }

    public static LakeScan scan(GravitinoOptions options) {
        GravitinoCatalog cat = GravitinoCatalog.open(options);
        return cat.scan(ns(options), options.table());
    }

    public static LakeScan scan(LakeOptions options) {
        return scan(GravitinoOptions.fromLakeOptions(options));
    }

    public static DataFrame read(GravitinoOptions options) {
        try (GravitinoCatalog cat = GravitinoCatalog.open(options)) {
            return cat.scan(ns(options), options.table()).collect();
        }
    }

    public static DataFrame read(LakeOptions options) {
        return read(GravitinoOptions.fromLakeOptions(options));
    }

    public static LakeWrite write(GravitinoOptions options) {
        GravitinoCatalog cat = GravitinoCatalog.open(options);
        LakeWrite inner = cat.write(ns(options), options.table());
        return new ClosingWrite(inner, cat);
    }

    public static LakeWrite write(LakeOptions options) {
        return write(GravitinoOptions.fromLakeOptions(options));
    }

    public static LakeStream stream(GravitinoOptions options) {
        GravitinoCatalog cat = GravitinoCatalog.open(options);
        LakeStream inner = cat.stream(ns(options), options.table());
        return new ClosingStream(inner, cat);
    }

    public static LakeStream stream(LakeOptions options) {
        return stream(GravitinoOptions.fromLakeOptions(options));
    }

    private static String ns(GravitinoOptions o) {
        if (o.catalogName() != null && o.schemaName() != null) {
            return o.catalogName() + "." + o.schemaName();
        }
        return o.schemaName();
    }

    private static final class ClosingWrite implements LakeWrite {
        private final LakeWrite delegate;
        private final GravitinoCatalog catalog;

        ClosingWrite(LakeWrite delegate, GravitinoCatalog catalog) {
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

    private static final class ClosingStream implements LakeStream {
        private final LakeStream delegate;
        private final GravitinoCatalog catalog;

        ClosingStream(LakeStream delegate, GravitinoCatalog catalog) {
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
}
