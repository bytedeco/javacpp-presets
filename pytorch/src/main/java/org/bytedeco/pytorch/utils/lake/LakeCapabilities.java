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

/**
 * Capabilities a concrete lake adapter actually supports.
 *
 * <p>Used for honest reporting and graceful fallback (e.g. "Iceberg scan supports partition prune
 * but not predicate pushdown yet").</p>
 */
public enum LakeCapabilities {
    /** Full column projection. */
    COLUMN_PROJECTION,
    /** Partition pruning (no unnecessary files). */
    PARTITION_PRUNING,
    /** Predicate pushdown (SQL / expression filter). */
    PREDICATE_PUSHDOWN,
    /** Incremental snapshot / changelog scan. */
    INCREMENTAL_SCAN,
    /** Upsert / merge-on-write semantics. */
    UPSERT,
    /** Stream load / bulk HTTP ingest. */
    STREAM_LOAD,
    /** High-concurrency point-query path. */
    POINT_QUERY,
    /** High-throughput append-only write. */
    HIGH_THROUGHPUT_APPEND,
    /** REST catalog federation support. */
    REST_CATALOG,
    /** Arrow / IPC interop. */
    ARROW_INTEROP
}
