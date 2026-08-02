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

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.dataset.DataFrameDataset;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Lake scan results → Tensor / training dataset helpers for feature engineering
 * and online training.
 *
 * <p>Delegates numeric packing to {@link DataFrame#toTensor(String...)} and
 * {@link TensorBridge}; does not reimplement dtype conversion.</p>
 *
 * <pre>{@code
 * DataFrame batch = stream.poll();
 * Tensor x = LakeTensorBridge.features(batch, "f1", "f2", "f3");
 * Tensor y = LakeTensorBridge.labels(batch, "label");
 * }</pre>
 */
public final class LakeTensorBridge {
    private LakeTensorBridge() {}

    /**
     * Pack selected numeric columns into a 2-D float feature tensor {@code [N, D]}.
     */
    public static Tensor features(DataFrame df, String... featureColumns) {
        Objects.requireNonNull(df, "dataframe");
        if (featureColumns == null || featureColumns.length == 0) {
            return df.toTensor();
        }
        return df.toTensor(featureColumns);
    }

    /**
     * Single label column as 1-D / column tensor.
     */
    public static Tensor labels(DataFrame df, String labelColumn) {
        Objects.requireNonNull(df, "dataframe");
        Objects.requireNonNull(labelColumn, "labelColumn");
        return df.toTensor(labelColumn);
    }

    /**
     * Dense feature matrix + label column pair for supervised online steps.
     */
    public static Tensor[] featuresAndLabels(DataFrame df, String labelColumn, String... featureColumns) {
        return new Tensor[]{features(df, featureColumns), labels(df, labelColumn)};
    }

    /**
     * Build a {@link DataFrameDataset} ready for DataLoader-style iteration
     * (reuses dataframe.dataset stack). All non-label columns become features.
     */
    public static DataFrameDataset toDataset(DataFrame df) {
        Objects.requireNonNull(df, "dataframe");
        try {
            return DataFrameDataset.builder(df).build();
        } catch (Exception e) {
            throw new IllegalStateException("Failed to build DataFrameDataset from lake frame", e);
        }
    }

    /**
     * Convert lake {@link LakeSchema} field list into an empty typed DataFrame shell.
     */
    public static DataFrame emptyFrame(LakeSchema schema) {
        Objects.requireNonNull(schema, "schema");
        DataFrame df = DataFrame.create();
        for (LakeSchema.Field f : schema.fields()) {
            df.addColumn(f.name(), f.dtype());
        }
        return df;
    }

    /**
     * Project only requested columns if present (missing columns skipped with warning via empty).
     */
    public static DataFrame project(DataFrame df, String... columns) {
        Objects.requireNonNull(df, "dataframe");
        if (columns == null || columns.length == 0) return df;
        List<String> keep = new ArrayList<>();
        for (String c : columns) {
            try {
                df.column(c);
                keep.add(c);
            } catch (Exception ignored) {
                // skip missing
            }
        }
        if (keep.isEmpty()) return DataFrame.create();
        return df.select(keep.toArray(new String[0]));
    }

    /**
     * Row-oriented maps for FeaturePlatform offline put (entity keys + features).
     */
    public static List<Map<String, Object>> toFeatureRows(DataFrame df) {
        Objects.requireNonNull(df, "dataframe");
        return df.toRecords();
    }

    /**
     * Inverse: offline feature rows → DataFrame (schema inferred from first row + overrides).
     */
    public static DataFrame fromFeatureRows(List<Map<String, Object>> rows, LakeSchema schema) {
        Objects.requireNonNull(rows, "rows");
        DataFrame df = schema != null ? emptyFrame(schema) : DataFrame.create();
        if (rows.isEmpty()) return df;

        if (schema == null) {
            Map<String, Object> first = rows.get(0);
            for (Map.Entry<String, Object> e : first.entrySet()) {
                df.addColumn(e.getKey(), inferDtype(e.getValue()));
            }
        }
        for (Map<String, Object> row : rows) {
            int ri = df.addEmptyRow();
            for (Column col : df.columns()) {
                df.set(ri, col.name(), row.get(col.name()));
            }
        }
        return df;
    }

    private static Column.DType inferDtype(Object v) {
        if (v == null) return Column.DType.STRING;
        if (v instanceof Boolean) return Column.DType.BOOLEAN;
        if (v instanceof Integer || v instanceof Short || v instanceof Byte) return Column.DType.INT32;
        if (v instanceof Long) return Column.DType.INT64;
        if (v instanceof Float) return Column.DType.FLOAT32;
        if (v instanceof Double || v instanceof Number) return Column.DType.FLOAT64;
        if (v instanceof byte[]) return Column.DType.BINARY;
        return Column.DType.STRING;
    }
}
