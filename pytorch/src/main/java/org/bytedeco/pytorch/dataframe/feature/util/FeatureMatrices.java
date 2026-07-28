package org.bytedeco.pytorch.dataframe.feature.util;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;

import java.util.ArrayList;
import java.util.List;

/**
 * Dense matrix helpers shared by sklearn-style transformers and multi-backend bridges.
 */
public final class FeatureMatrices {
    private FeatureMatrices() {}

    public static double[][] fromDf(DataFrame df, String... cols) {
        if (cols == null || cols.length == 0) {
            List<String> numeric = numericColumnNames(df);
            cols = numeric.toArray(new String[0]);
        }
        int n = df.rowCount();
        int d = cols.length;
        double[][] out = new double[n][d];
        for (int j = 0; j < d; j++) {
            Column c = df.column(cols[j]);
            for (int i = 0; i < n; i++) {
                out[i][j] = DataValues.asDouble(c.get(i));
            }
        }
        return out;
    }

    public static double[] columnAsDoubles(DataFrame df, String col) {
        Column c = df.column(col);
        double[] y = new double[df.rowCount()];
        for (int i = 0; i < y.length; i++) {
            double v = DataValues.asDouble(c.get(i));
            y[i] = Double.isNaN(v) ? 0.0 : v;
        }
        return y;
    }

    public static DataFrame toDf(double[][] matrix, String... names) {
        if (matrix == null) return DataFrame.create();
        int n = matrix.length;
        int d = n == 0 ? 0 : matrix[0].length;
        if (names == null || names.length != d) {
            names = new String[d];
            for (int j = 0; j < d; j++) names[j] = "f" + j;
        }
        DataFrame df = DataFrame.create();
        for (String name : names) df.addColumn(name, Column.DType.FLOAT64);
        for (int i = 0; i < n; i++) {
            Object[] row = new Object[d];
            for (int j = 0; j < d; j++) {
                double v = matrix[i][j];
                row[j] = Double.isNaN(v) ? null : v;
            }
            df.addRow(row);
        }
        return df;
    }

    public static DataFrame replaceColumns(DataFrame src, String[] cols, double[][] matrix) {
        DataFrame out = src.copy();
        int n = out.rowCount();
        for (int j = 0; j < cols.length; j++) {
            String col = cols[j];
            if (!out.hasColumn(col)) out.addColumn(col, Column.DType.FLOAT64);
            Column c = out.column(col);
            while (c.size() < n) c.add(null);
            for (int i = 0; i < n; i++) {
                double v = matrix[i][j];
                c.set(i, Double.isNaN(v) ? null : v);
            }
        }
        return out;
    }

    public static DataFrame appendColumns(DataFrame src, String[] names, double[][] matrix) {
        DataFrame out = src.copy();
        int n = out.rowCount();
        for (int j = 0; j < names.length; j++) {
            String name = uniqueName(out, names[j]);
            out.addColumn(name, Column.DType.FLOAT64);
            Column c = out.column(name);
            while (c.size() < n) c.add(null);
            for (int i = 0; i < n; i++) {
                double v = matrix[i][j];
                c.set(i, Double.isNaN(v) ? null : v);
            }
        }
        return out;
    }

    public static String uniqueName(DataFrame df, String base) {
        if (!df.hasColumn(base)) return base;
        int k = 1;
        String name = base + "_" + k;
        while (df.hasColumn(name)) {
            k++;
            name = base + "_" + k;
        }
        return name;
    }

    public static List<String> numericColumnNames(DataFrame df) {
        List<String> out = new ArrayList<>();
        for (Column c : df.columns()) {
            Column.DType d = c.dtype();
            if (d == Column.DType.INT32 || d == Column.DType.INT64
                || d == Column.DType.FLOAT32 || d == Column.DType.FLOAT64) {
                out.add(c.name());
            }
        }
        return out;
    }

    public static boolean isMissing(Object v) {
        if (v == null) return true;
        double d = DataValues.asDouble(v);
        return Double.isNaN(d);
    }

    public static boolean allFinite(double[][] m) {
        if (m == null) return true;
        for (double[] row : m) {
            if (row == null) return false;
            for (double v : row) {
                if (Double.isNaN(v) || Double.isInfinite(v)) return false;
            }
        }
        return true;
    }

    public static double maxAbsDiff(double[][] a, double[][] b) {
        if (a == null || b == null) return Double.POSITIVE_INFINITY;
        if (a.length != b.length) return Double.POSITIVE_INFINITY;
        double max = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i].length != b[i].length) return Double.POSITIVE_INFINITY;
            for (int j = 0; j < a[i].length; j++) {
                double da = a[i][j], db = b[i][j];
                if (Double.isNaN(da) && Double.isNaN(db)) continue;
                if (Double.isNaN(da) || Double.isNaN(db)) return Double.POSITIVE_INFINITY;
                max = Math.max(max, Math.abs(da - db));
            }
        }
        return max;
    }

    public static double[][] copyMatrix(double[][] src) {
        if (src == null) return null;
        double[][] out = new double[src.length][];
        for (int i = 0; i < src.length; i++) out[i] = src[i] == null ? null : src[i].clone();
        return out;
    }

    public static double percentileSorted(List<Double> sorted, double p) {
        if (sorted == null || sorted.isEmpty()) return 0.0;
        if (p <= 0) return sorted.get(0);
        if (p >= 1) return sorted.get(sorted.size() - 1);
        double idx = p * (sorted.size() - 1);
        int lo = (int) Math.floor(idx);
        int hi = (int) Math.ceil(idx);
        if (lo == hi) return sorted.get(lo);
        double w = idx - lo;
        return sorted.get(lo) * (1 - w) + sorted.get(hi) * w;
    }
}
