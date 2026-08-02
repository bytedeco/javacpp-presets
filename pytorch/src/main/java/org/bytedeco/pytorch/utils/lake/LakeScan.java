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
package org.bytedeco.pytorch.utils.lake;

import org.bytedeco.pytorch.dataframe.DataFrame;

/**
 * Fluent batch scan builder → {@link DataFrame} or {@link LakeStream}.
 *
 * <pre>{@code
 * DataFrame df = catalog.scan("db", "events")
 *     .columns("user_id", "item_id", "ts")
 *     .filter(PartitionFilter.eq("dt", "2026-08-01"))
 *     .limit(1_000_000)
 *     .collect();
 * }</pre>
 */
public interface LakeScan {

    LakeTable table();

    LakeScan columns(String... columns);

    LakeScan filter(PartitionFilter filter);

    /** Free-form predicate (SQL WHERE fragment or engine expression). */
    LakeScan where(String expression);

    LakeScan snapshotId(Long snapshotId);

    LakeScan asOfTimeMs(Long epochMs);

    LakeScan replicas(ReplicaPolicy policy);

    LakeScan limit(long maxRows);

    LakeScan batchRows(int batchRows);

    LakeScan parallelism(int parallelism);

    /** Materialize entire result (caller must ensure size is safe). */
    DataFrame collect() throws LakeException;

    /** Streaming micro-batches over the same scan plan. */
    LakeStream stream() throws LakeException;
}
