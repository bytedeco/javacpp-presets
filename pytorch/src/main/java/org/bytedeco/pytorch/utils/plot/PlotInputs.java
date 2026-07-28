package org.bytedeco.pytorch.utils.plot;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;

/**
 * Unified conversion of plot inputs from {@code double[]}/{@code double[][]},
 * {@link NDArray} (numpy), {@link Tensor} (javacpp-pytorch), and {@link DataFrame}
 * columns into the primitive arrays consumed by AWT charts.
 *
 * <p>All Seaborn / Matplotlib overloads should route through this helper so the
 * three backends share one conversion policy.
 */
public final class PlotInputs {
    private PlotInputs() {}

    // ---- 1D --------------------------------------------------------------------

    public static double[] asDouble1D(double[] a) {
        if (a == null) throw new IllegalArgumentException("null double[]");
        return a;
    }

    public static double[] asDouble1D(NDArray a) {
        if (a == null) throw new IllegalArgumentException("null NDArray");
        return a.asDoubleArray();
    }

    public static double[] asDouble1D(Tensor t) {
        return TensorPlotUtils.asDouble1D(t);
    }

    public static double[] asDouble1D(Column col) {
        if (col == null) throw new IllegalArgumentException("null Column");
        return col.asDoubleArray();
    }

    public static double[] asDouble1D(DataFrame df, String col) {
        if (df == null) throw new IllegalArgumentException("null DataFrame");
        return df.column(col).asDoubleArray();
    }

    // ---- 2D --------------------------------------------------------------------

    public static double[][] asDouble2D(double[][] m) {
        if (m == null) throw new IllegalArgumentException("null double[][]");
        return m;
    }

    /**
     * Convert NDArray to row-major matrix. Rank 1 → single-row; rank 2 → (rows, cols);
     * higher ranks → first plane via reshape to leading two dims product is rejected
     * (use Tensor path for multi-plane).
     */
    public static double[][] asDouble2D(NDArray a) {
        if (a == null) throw new IllegalArgumentException("null NDArray");
        int nd = a.ndim();
        if (nd == 0) return new double[][]{{a.getDouble(0)}};
        if (nd == 1) {
            double[] row = a.asDoubleArray();
            return new double[][]{row};
        }
        if (nd == 2) {
            int rows = (int) a.shape[0];
            int cols = (int) a.shape[1];
            double[][] m = new double[rows][cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i][j] = a.getDouble(i * cols + j);
            return m;
        }
        // rank > 2: flatten leading dims into rows using last dim as cols
        int cols = (int) a.shape[nd - 1];
        long leading = a.size / cols;
        if (leading > Integer.MAX_VALUE) throw new IllegalArgumentException("NDArray too large");
        int rows = (int) leading;
        double[] flat = a.asDoubleArray();
        double[][] m = new double[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(flat, i * cols, m[i], 0, cols);
        return m;
    }

    public static double[][] asDouble2D(Tensor t) {
        TensorPlotUtils.requireNonNull(t);
        TensorPlotUtils.rejectScalar(t);
        int r = TensorPlotUtils.rank(t);
        if (r <= 2) return TensorPlotUtils.asMatrix2D(t);
        return TensorPlotUtils.firstPlaneAsMatrix(t);
    }

    /** Extract multiple numeric columns from a DataFrame as a row-major matrix (nRows × nCols). */
    public static double[][] fromColumns(DataFrame df, String... cols) {
        if (df == null) throw new IllegalArgumentException("null DataFrame");
        if (cols == null || cols.length == 0) throw new IllegalArgumentException("no columns");
        int n = df.rowCount();
        double[][] m = new double[n][cols.length];
        for (int c = 0; c < cols.length; c++) {
            double[] col = df.column(cols[c]).asDoubleArray();
            for (int r = 0; r < n; r++) m[r][c] = r < col.length ? col[r] : Double.NaN;
        }
        return m;
    }

    // ---- helpers ----------------------------------------------------------------

    public static void requireSameLength(double[] a, double[] b, String what) {
        if (a == null || b == null) throw new IllegalArgumentException(what + ": null array");
        if (a.length != b.length)
            throw new IllegalArgumentException(what + ": length mismatch " + a.length + " vs " + b.length);
    }

    public static double[] indexArray(int n) {
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = i;
        return x;
    }

    public static String[] indexLabels(int n) {
        String[] s = new String[n];
        for (int i = 0; i < n; i++) s[i] = String.valueOf(i);
        return s;
    }

    /** Build a minimal 2-column DataFrame from x/y arrays (for DF-only seaborn APIs). */
    public static DataFrame xyDataFrame(String xName, double[] x, String yName, double[] y) {
        requireSameLength(x, y, "xyDataFrame");
        DataFrame df = DataFrame.create();
        df.addColumn(xName, Column.DType.FLOAT64);
        df.addColumn(yName, Column.DType.FLOAT64);
        for (int i = 0; i < x.length; i++) df.addRow(x[i], y[i]);
        return df;
    }

    /** Build a group/value DataFrame from labeled 1D groups. */
    public static DataFrame groupValueDataFrame(String groupCol, String valueCol,
                                                 String[] labels, double[]... groups) {
        if (labels == null || groups == null || labels.length != groups.length)
            throw new IllegalArgumentException("labels/groups length mismatch");
        DataFrame df = DataFrame.create();
        df.addColumn(groupCol, Column.DType.STRING);
        df.addColumn(valueCol, Column.DType.FLOAT64);
        for (int g = 0; g < groups.length; g++) {
            String lab = labels[g] == null ? ("G" + g) : labels[g];
            double[] vals = groups[g];
            if (vals == null) continue;
            for (double v : vals) df.addRow(lab, v);
        }
        return df;
    }

    public static DataFrame groupValueDataFrame(String groupCol, String valueCol,
                                                 String[] labels, NDArray... groups) {
        double[][] g = new double[groups.length][];
        for (int i = 0; i < groups.length; i++) g[i] = asDouble1D(groups[i]);
        return groupValueDataFrame(groupCol, valueCol, labels, g);
    }

    public static DataFrame groupValueDataFrame(String groupCol, String valueCol,
                                                 String[] labels, Tensor... groups) {
        double[][] g = new double[groups.length][];
        for (int i = 0; i < groups.length; i++) g[i] = asDouble1D(groups[i]);
        return groupValueDataFrame(groupCol, valueCol, labels, g);
    }
}
