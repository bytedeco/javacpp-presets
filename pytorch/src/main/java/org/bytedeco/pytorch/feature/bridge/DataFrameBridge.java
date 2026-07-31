/*
 * Bridge between feature platform row maps and dataframe.DataFrame.
 */
package org.bytedeco.pytorch.feature.bridge;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.feature.offline.TrainingDataset;
import org.bytedeco.pytorch.feature.serving.FeatureVector;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Convert platform rows ↔ DataFrame for offline FE / export. */
public final class DataFrameBridge {

    private DataFrameBridge() {}

    public static DataFrame fromRows(List<Map<String, Object>> rows) {
        DataFrame df = DataFrame.create();
        if (rows == null || rows.isEmpty()) return df;

        List<String> cols = new ArrayList<>(rows.get(0).keySet());
        for (Map<String, Object> r : rows) {
            for (String k : r.keySet()) {
                if (!cols.contains(k)) cols.add(k);
            }
        }
        for (String c : cols) {
            df.addColumn(c, inferDtype(rows, c));
        }
        for (Map<String, Object> r : rows) {
            int idx = df.addRow();
            for (String c : cols) {
                df.set(idx, c, r.get(c));
            }
        }
        return df;
    }

    public static DataFrame fromTrainingDataset(TrainingDataset dataset) {
        Objects.requireNonNull(dataset, "dataset");
        return fromRows(dataset.rows());
    }

    public static DataFrame fromFeatureVectors(List<FeatureVector> vectors) {
        List<Map<String, Object>> rows = new ArrayList<>();
        if (vectors != null) {
            for (FeatureVector v : vectors) {
                rows.add(RecommendFeatureBridge.toRawMap(v));
            }
        }
        return fromRows(rows);
    }

    public static List<Map<String, Object>> toRows(DataFrame df) {
        Objects.requireNonNull(df, "df");
        List<Map<String, Object>> rows = new ArrayList<>();
        List<String> names = df.getColumnNames();
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            Map<String, Object> row = new LinkedHashMap<>();
            for (String name : names) {
                row.put(name, df.get(i, name));
            }
            rows.add(row);
        }
        return rows;
    }

    /**
     * Project a DataFrame onto an ordered column subset (missing cols skipped with warning via empty).
     * Used by FeatureIngest to keep entity + timestamp + feature columns only.
     */
    public static DataFrame selectColumns(DataFrame df, List<String> columns) {
        Objects.requireNonNull(df, "df");
        if (columns == null || columns.isEmpty()) return df;
        DataFrame out = DataFrame.create();
        List<String> keep = new ArrayList<>();
        for (String c : columns) {
            if (c != null && df.hasColumn(c) && !keep.contains(c)) keep.add(c);
        }
        if (keep.isEmpty()) return out;
        for (String c : keep) {
            Column src = df.column(c);
            out.addColumn(c, src.dtype());
        }
        int n = df.rowCount();
        for (int i = 0; i < n; i++) {
            int idx = out.addRow();
            for (String c : keep) {
                out.set(idx, c, df.get(i, c));
            }
        }
        return out;
    }

    /**
     * Infer a simple schema map column → suggested ValueType name for registry auto-register.
     */
    public static Map<String, String> inferSchema(DataFrame df) {
        Objects.requireNonNull(df, "df");
        Map<String, String> schema = new LinkedHashMap<>();
        for (String c : df.getColumnNames()) {
            Column.DType dt = df.column(c).dtype();
            String vt;
            switch (dt) {
                case INT32: vt = "INT32"; break;
                case INT64: vt = "INT64"; break;
                case FLOAT32: vt = "FLOAT32"; break;
                case FLOAT64: vt = "FLOAT64"; break;
                case BOOLEAN: vt = "BOOL"; break;
                case VECTOR:
                case EMBEDDING: vt = "EMBEDDING"; break;
                case LIST: vt = "INT64_LIST"; break;
                default: vt = "STRING"; break;
            }
            schema.put(c, vt);
        }
        return schema;
    }

    /**
     * Append a constant long timestamp column if missing (ingest helper).
     */
    public static DataFrame ensureEventTimestamp(DataFrame df, String col, long epochMs) {
        Objects.requireNonNull(df, "df");
        String c = col != null ? col : "event_timestamp";
        if (df.hasColumn(c)) return df;
        df.addColumn(c, Column.DType.INT64);
        for (int i = 0; i < df.rowCount(); i++) {
            df.set(i, c, epochMs);
        }
        return df;
    }

    /**
     * Build a dense double matrix [n, f] for numeric columns (training export helper).
     * Non-numeric → NaN.
     */
    public static double[][] toDenseMatrix(DataFrame df, List<String> columns) {
        Objects.requireNonNull(df, "df");
        List<String> cols = columns != null ? columns : df.getColumnNames();
        int n = df.rowCount();
        int f = cols.size();
        double[][] m = new double[n][f];
        for (int j = 0; j < f; j++) {
            String c = cols.get(j);
            if (!df.hasColumn(c)) {
                for (int i = 0; i < n; i++) m[i][j] = Double.NaN;
                continue;
            }
            for (int i = 0; i < n; i++) {
                Object v = df.get(i, c);
                m[i][j] = v instanceof Number ? ((Number) v).doubleValue() : Double.NaN;
            }
        }
        return m;
    }

    private static Column.DType inferDtype(List<Map<String, Object>> rows, String col) {
        for (Map<String, Object> r : rows) {
            Object v = r.get(col);
            if (v == null) continue;
            if (v instanceof float[] || v instanceof double[]) return Column.DType.VECTOR;
            if (v instanceof long[] || v instanceof int[] || v instanceof List) return Column.DType.LIST;
            if (v instanceof Long || v instanceof Integer || v instanceof Short || v instanceof Byte) {
                return Column.DType.INT64;
            }
            if (v instanceof Double || v instanceof Float) return Column.DType.FLOAT64;
            if (v instanceof Boolean) return Column.DType.BOOLEAN;
            return Column.DType.STRING;
        }
        return Column.DType.STRING;
    }
}
