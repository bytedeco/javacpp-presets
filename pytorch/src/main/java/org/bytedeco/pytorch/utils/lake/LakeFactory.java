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

import org.bytedeco.pytorch.utils.doris.DorisCatalog;
import org.bytedeco.pytorch.utils.doris.DorisOptions;

import java.util.Objects;

/**
 * Factory to open any supported lake / OLAP backend via {@link LakeOptions#format}.
 *
 * <p>Currently only Doris is implemented. New adapters are added under {@code utils.*}
 * and registered here when they implement {@link LakeCatalog}.</p>
 *
 * <p>Usage:</p>
 * <pre>{@code
 * LakeOptions opts = LakeOptions.builder(LakeFormat.DORIS)
 *     .uri("doris://user:pass@fe-host:9030/db")
 *     .namespaceName("rec")
 *     .table("user_features")
 *     .build();
 *
 * try (LakeCatalog cat = LakeFactory.open(opts)) {
 *     // ... scan / write / stream
 * }
 * }</pre>
 */
public final class LakeFactory {
    private LakeFactory() {}

    /**
     * Open catalog for the given format / URI.
     */
    public static LakeCatalog open(LakeOptions options) {
        Objects.requireNonNull(options, "options");
        LakeFormat fmt = options.format();
        if (fmt == null) throw new IllegalArgumentException("format required");
        return switch (fmt) {
            case DORIS -> new DorisCatalog(DorisOptions.fromLakeOptions(options));
            case ICEBERG, PAIMON, HUDI, GRAVITINO, DAFT, PARQUET ->
                    throw new LakeException(fmt, "open",
                            "format " + fmt + " adapter not registered yet — see DATA_LAKE_AI_ADAPTERS_PLAN.md");
        };
    }

    /**
     * Convenience: open Doris catalog from JDBC URI.
     */
    public static LakeCatalog openDoris(String jdbcUrl) {
        return open(LakeOptions.of(LakeFormat.DORIS, jdbcUrl));
    }
}
