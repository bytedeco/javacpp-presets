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
 * Supported lake / OLAP backends exposed through the unified SPI.
 */
public enum LakeFormat {
    /** Apache Doris (MySQL protocol + Stream Load). */
    DORIS,
    /** Apache Iceberg open table format. */
    ICEBERG,
    /** Apache Paimon (lakehouse streaming table). */
    PAIMON,
    /** Apache Hudi. */
    HUDI,
    /** Apache Gravitino federated catalog (resolves to a concrete backend). */
    GRAVITINO,
    /** Daft-style Arrow/Parquet interop bridge (no embedded Rust runner). */
    DAFT,
    /** Plain Parquet directory treated as a partitioned lake table. */
    PARQUET
}
