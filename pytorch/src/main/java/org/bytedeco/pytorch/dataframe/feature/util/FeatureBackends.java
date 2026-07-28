package org.bytedeco.pytorch.dataframe.feature.util;

import org.bytedeco.pytorch.Tensor;
import org.bytedeco.pytorch.data.numpy.NDArray;
import org.bytedeco.pytorch.data.numpy.NP;
import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.tensor.TensorBridge;
import org.bytedeco.pytorch.global.torch;

/**
 * Multi-backend bridge for feature engineering:
 * {@code double[][]} ↔ {@link DataFrame} ↔ {@link NDArray} ↔ {@link Tensor}.
 *
 * <p>Transformers stay DataFrame-primary (sklearn-style). Numeric-only flows can
 * round-trip through numpy / Tensor without forking every estimator.
 */
public final class FeatureBackends {
    private FeatureBackends() {}

    // ---- double[][] ↔ DataFrame ----

    public static double[][] toMatrix(DataFrame df, String... cols) {
        return FeatureMatrices.fromDf(df, cols);
    }

    public static DataFrame fromMatrix(double[][] matrix, String... names) {
        return FeatureMatrices.toDf(matrix, names);
    }

    // ---- double[][] ↔ NDArray ----

    public static NDArray toNdArray(double[][] matrix) {
        if (matrix == null) return NP.array(new double[0], 0L);
        int n = matrix.length;
        int d = n == 0 ? 0 : matrix[0].length;
        double[] flat = new double[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, flat, i * d, d);
        }
        return NP.array(flat, (long) n, (long) d);
    }

    public static double[][] fromNdArray(NDArray arr) {
        if (arr == null) return new double[0][0];
        long[] shape = arr.shape;
        if (shape.length == 0) {
            return new double[][]{{arr.getDouble(0)}};
        }
        if (shape.length == 1) {
            int n = (int) shape[0];
            double[][] out = new double[n][1];
            for (int i = 0; i < n; i++) out[i][0] = arr.getDouble(i);
            return out;
        }
        int n = (int) shape[0];
        int d = (int) shape[1];
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = arr.getDouble(i * d + j);
            }
        }
        return out;
    }

    public static NDArray dfToNdArray(DataFrame df, String... cols) {
        return toNdArray(toMatrix(df, cols));
    }

    public static DataFrame ndArrayToDf(NDArray arr, String... names) {
        return fromMatrix(fromNdArray(arr), names);
    }

    // ---- double[][] ↔ Tensor ----

    public static Tensor toTensor(double[][] matrix) {
        NDArray arr = toNdArray(matrix);
        return NP.toTensor(arr);
    }

    public static double[][] fromTensor(Tensor t) {
        if (t == null || t.isNull()) return new double[0][0];
        // Prefer contiguous CPU double view via NP bridge when possible
        try {
            NDArray arr = NP.fromTensor(t);
            return fromNdArray(arr);
        } catch (Throwable ignored) {
            // Fallback: copy via DataFrame helper path
        }
        try {
            Tensor cpu = t;
            if (t.is_cuda()) cpu = t.cpu();
            if (!cpu.is_contiguous()) cpu = cpu.contiguous();
            long[] shape = cpu.shape();
            if (shape.length == 1) {
                int n = (int) shape[0];
                double[][] out = new double[n][1];
                float[] flat = new float[n];
                // use double if available
                if (cpu.scalar_type() == torch.ScalarType.Double) {
                    org.bytedeco.javacpp.DoublePointer ptr = cpu.data_ptr_double();
                    for (int i = 0; i < n; i++) out[i][0] = ptr.get(i);
                } else {
                    Tensor f = cpu.to(torch.ScalarType.Float);
                    org.bytedeco.javacpp.FloatPointer fp = f.data_ptr_float();
                    for (int i = 0; i < n; i++) out[i][0] = fp.get(i);
                }
                return out;
            }
            int n = (int) shape[0];
            int d = shape.length > 1 ? (int) shape[1] : 1;
            double[][] out = new double[n][d];
            if (cpu.scalar_type() == torch.ScalarType.Double) {
                org.bytedeco.javacpp.DoublePointer ptr = cpu.data_ptr_double();
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < d; j++)
                        out[i][j] = ptr.get((long) i * d + j);
            } else {
                Tensor f = cpu.to(torch.ScalarType.Float);
                org.bytedeco.javacpp.FloatPointer fp = f.data_ptr_float();
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < d; j++)
                        out[i][j] = fp.get((long) i * d + j);
            }
            return out;
        } catch (Throwable e) {
            throw new IllegalStateException("fromTensor failed: " + e.getMessage(), e);
        }
    }

    public static Tensor dfToTensor(DataFrame df, String... cols) {
        if (cols != null && cols.length > 0) {
            return df.toTensor(cols);
        }
        return toTensor(toMatrix(df));
    }

    public static DataFrame tensorToDf(Tensor t, String... names) {
        return fromMatrix(fromTensor(t), names);
    }

    // ---- numeric StandardScaler on raw matrix (backend parity helper) ----

    public static final class MatrixStandardScaler {
        private double[] mean;
        private double[] std;
        private boolean fitted;

        public MatrixStandardScaler fit(double[][] X) {
            int n = X.length, d = X[0].length;
            mean = new double[d];
            std = new double[d];
            for (int j = 0; j < d; j++) {
                double sum = 0, sumSq = 0;
                int c = 0;
                for (int i = 0; i < n; i++) {
                    double v = X[i][j];
                    if (Double.isNaN(v)) continue;
                    sum += v;
                    sumSq += v * v;
                    c++;
                }
                mean[j] = c == 0 ? 0 : sum / c;
                double var = c < 2 ? 0 : (sumSq - c * mean[j] * mean[j]) / c;
                std[j] = Math.sqrt(Math.max(0, var));
                if (std[j] == 0) std[j] = 1.0;
            }
            fitted = true;
            return this;
        }

        public double[][] transform(double[][] X) {
            if (!fitted) throw new IllegalStateException("MatrixStandardScaler not fitted");
            int n = X.length, d = X[0].length;
            double[][] out = new double[n][d];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < d; j++) {
                    double v = X[i][j];
                    out[i][j] = Double.isNaN(v) ? Double.NaN : (v - mean[j]) / std[j];
                }
            }
            return out;
        }

        public double[][] fitTransform(double[][] X) {
            return fit(X).transform(X);
        }

        public double[] getMean() { return mean; }
        public double[] getStd() { return std; }
    }

    /** Max abs relative/absolute difference for backend parity asserts. */
    public static double maxAbsDiff(double[][] a, double[][] b) {
        return FeatureMatrices.maxAbsDiff(a, b);
    }

    /** Extract numeric column as double[] (NaN for missing). */
    public static double[] columnToArray(DataFrame df, String col) {
        Column c = df.column(col);
        double[] out = new double[df.rowCount()];
        for (int i = 0; i < out.length; i++) {
            Object v = c.get(i);
            if (v == null) out[i] = Double.NaN;
            else out[i] = DataValues.asDouble(v);
        }
        return out;
    }
}
