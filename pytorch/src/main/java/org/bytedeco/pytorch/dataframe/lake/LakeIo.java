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
package org.bytedeco.pytorch.dataframe.lake;

import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.utils.lake.LakeCatalog;
import org.bytedeco.pytorch.utils.lake.LakeFactory;
import org.bytedeco.pytorch.utils.lake.LakeOptions;
import org.bytedeco.pytorch.utils.lake.LakeStream;
import org.bytedeco.pytorch.utils.lake.LakeTable;
import org.bytedeco.pytorch.utils.lake.LakeWrite;

import java.util.Map;

///**
// * DataFrame → lake I/O facade (read*/to* methods).
// *
// * <p>Usage:</p>
// * <pre>{@code
// * DataFrame df = DataFrame.readDoris(opts, "SELECT * FROM rec.user_features");
// * df.toDoris(opts);
// * }</pre>
// *
// * @see LakeOptions
// * @see LakeFactory
// * @see LakeCatalog
// * @see LakeStream
// */
public final class LakeIo {
    private LakeIo() {}

    public static DataFrame readDoris(LakeOptions options, String sql) {
        try (LakeCatalog cat = LakeFactory.open(options)) {
            return cat.read(options.namespaceName(), options.table());
        }
    }

    public static DataFrame readIceberg(LakeOptions options) {
        throw new UnsupportedOperationException("Iceberg not yet implemented");
    }

    public static DataFrame readPaimon(LakeOptions options) {
        throw new UnsupportedOperationException("Paimon not yet implemented");
    }

    public static DataFrame readHudi(LakeOptions options) {
        throw new UnsupportedOperationException("Hudi not yet implemented");
    }

    public static DataFrame readGravitino(LakeOptions options) {
        throw new UnsupportedOperationException("Gravitino not yet implemented");
    }

    public static DataFrame readDaft(LakeOptions options) {
        throw new UnsupportedOperationException("Daft not yet implemented");
    }

    public static DataFrame readLake(LakeOptions options) {
        try (LakeCatalog cat = LakeFactory.open(options)) {
            return cat.read(options.namespaceName(), options.table());
        }
    }

    public static void toDoris(LakeOptions options, DataFrame df) {
        try (LakeCatalog cat = LakeFactory.open(options)) {
            try (LakeWrite w = cat.write(options.namespaceName(), options.table())) {
                w.mode(LakeWrite.Mode.APPEND).write(df).commit();
            }
        }
    }

    public static void toIceberg(LakeOptions options, DataFrame df) {
        throw new UnsupportedOperationException("Iceberg not yet implemented");
    }

    public static void toPaimon(LakeOptions options, DataFrame df) {
        throw new UnsupportedOperationException("Paimon not yet implemented");
    }

    public static void toHudi(LakeOptions options, DataFrame df) {
        throw new UnsupportedOperationException("Hudi not yet implemented");
    }

    public static void toGravitino(LakeOptions options, DataFrame df) {
        throw new UnsupportedOperationException("Gravitino not yet implemented");
    }

    public static void toDaft(LakeOptions options, DataFrame df) {
        throw new UnsupportedOperationException("Daft not yet implemented");
    }

    public static void toLake(LakeOptions options, DataFrame df) {
        try (LakeCatalog cat = LakeFactory.open(options)) {
            try (LakeWrite w = cat.write(options.namespaceName(), options.table())) {
                w.mode(LakeWrite.Mode.APPEND).write(df).commit();
            }
        }
    }

    public static LakeStream streamLake(LakeOptions options) {
        try (LakeCatalog cat = LakeFactory.open(options)) {
            return cat.stream(options.namespaceName(), options.table());
        }
    }

    public static DataFrame read(String lakeFormat, LakeOptions options, String sql) {
        return switch (lakeFormat.toUpperCase()) {
            case "DORIS" -> readDoris(options, sql);
            case "ICEBERG" -> readIceberg(options);
            case "PAIMON" -> readPaimon(options);
            case "HUDI" -> readHudi(options);
            case "GRAVITINO" -> readGravitino(options);
            case "DAFT" -> readDaft(options);
            default -> throw new IllegalArgumentException("unsupported lakeFormat: " + lakeFormat);
        };
    }
}
