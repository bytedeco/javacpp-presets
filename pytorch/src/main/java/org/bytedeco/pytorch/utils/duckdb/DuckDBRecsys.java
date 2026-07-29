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
package org.bytedeco.pytorch.utils.duckdb;

import org.bytedeco.pytorch.dataframe.DataFrame;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Facade wiring {@link DuckDBFeatureStore} + {@link DuckDBAnalytics} +
 * {@link DuckDBMultimodal} for search / recommendation / ads offline pipelines.
 *
 * <p>One-stop entry for the common loop:
 * <ol>
 *   <li>open engineered DuckDB (threads/memory)</li>
 *   <li>ingest parquet event logs</li>
 *   <li>materialize user/item/sequence features</li>
 *   <li>point-in-time or snapshot join → training table</li>
 *   <li>negative sample (two-tower) / time split</li>
 *   <li>export parquet; compute offline metrics</li>
 * </ol>
 *
 * <pre>{@code
 * try (DuckDBRecsys rec = DuckDBRecsys.open(Path.of("rec.duckdb"))) {
 *     rec.featureStore().ensureEventLog("events");
 *     rec.featureStore().ingestParquetEvents("events", "logs/**.parquet", true);
 *     rec.featureStore().materializeUserAgg("events", "user_feat", "user_id", "ts", 7);
 *     DataFrame train = rec.buildTrainingSnapshot(
 *         "events", "user_feat", null, "user_id", "item_id",
 *         List.of("ctr_7d", "impress_7d"), List.of());
 *     rec.featureStore().exportTrainingParquet(train, "train.parquet");
 *     DataFrame ctr = rec.analytics().dailyCtrCvr("events");
 * }
 * }</pre>
 */
public final class DuckDBRecsys implements AutoCloseable {

    private final DuckDB db;
    private final boolean ownsDb;
    private final DuckDBFeatureStore featureStore;
    private final DuckDBAnalytics analytics;
    private final DuckDBMultimodal multimodal;

    public DuckDBRecsys(DuckDB db) {
        this(db, false);
    }

    public DuckDBRecsys(DuckDB db, boolean ownsDb) {
        this.db = Objects.requireNonNull(db, "db");
        this.ownsDb = ownsDb;
        this.featureStore = new DuckDBFeatureStore(db);
        this.analytics = new DuckDBAnalytics(db);
        this.multimodal = new DuckDBMultimodal(db);
    }

    public static DuckDBRecsys open(Path dbFile) throws Exception {
        DuckDB db = DuckDB.open(dbFile, DuckDBConfig.offlineFeatureEngineering());
        return new DuckDBRecsys(db, true);
    }

    public static DuckDBRecsys open(Path dbFile, DuckDBConfig config) throws Exception {
        DuckDB db = DuckDB.open(dbFile, config == null
                ? DuckDBConfig.offlineFeatureEngineering() : config);
        return new DuckDBRecsys(db, true);
    }

    public static DuckDBRecsys inMemory() throws Exception {
        return new DuckDBRecsys(DuckDB.inMemory(DuckDBConfig.offlineFeatureEngineering()), true);
    }

    public DuckDB db() { return db; }
    public DuckDBFeatureStore featureStore() { return featureStore; }
    public DuckDBAnalytics analytics() { return analytics; }
    public DuckDBMultimodal multimodal() { return multimodal; }

    /**
     * Snapshot assemble (no time travel) of user/item features onto events.
     */
    public DataFrame buildTrainingSnapshot(String eventTable,
                                           String userFeatTable, String itemFeatTable,
                                           String userCol, String itemCol,
                                           List<String> userCols, List<String> itemCols)
            throws Exception {
        return featureStore.assembleBatch(eventTable, userFeatTable, itemFeatTable,
                userCol, itemCol, userCols, itemCols);
    }

    /**
     * End-to-end daily job sketch: user agg → item agg → sequence → train table export.
     */
    public void runDailyMaterialize(String eventTable, int windowDays, int seqLen,
                                    String trainOutParquet) throws Exception {
        featureStore.ensureEventLog(eventTable);
        featureStore.materializeUserAgg(eventTable, "user_feat", "user_id", "ts", windowDays);
        featureStore.materializeItemAgg(eventTable, "item_feat", "item_id", "ts", windowDays);
        featureStore.materializeUserSequence(eventTable, "user_seq", "user_id", "item_id", "ts",
                seqLen);
        DataFrame train = featureStore.assembleBatch(
                eventTable, "user_feat", "item_feat",
                "user_id", "item_id",
                List.of("impress_" + windowDays + "d", "click_" + windowDays + "d",
                        "ctr_" + windowDays + "d"),
                List.of("impress_" + windowDays + "d", "click_" + windowDays + "d",
                        "ctr_" + windowDays + "d"));
        if (trainOutParquet != null) {
            featureStore.exportTrainingParquet(train, trainOutParquet);
        } else {
            db.replaceWithDataFrame("train_samples", train);
        }
    }

    @Override
    public void close() {
        if (ownsDb) db.close();
    }
}
