package org.bytedeco.pytorch.dataframe.feature.imputation;

import org.bytedeco.pytorch.dataframe.Column;
import org.bytedeco.pytorch.dataframe.DataFrame;
import org.bytedeco.pytorch.dataframe.DataValues;
import org.bytedeco.pytorch.dataframe.feature.BaseTransformer;
import org.bytedeco.pytorch.dataframe.feature.util.DenseLinalg;
import org.bytedeco.pytorch.dataframe.feature.util.FeatureMatrices;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Iterative / MICE-style imputer (sklearn IterativeImputer core).
 *
 * <p>Each feature with missing values is modeled as a linear function of the others;
 * predictions fill missing cells, then the process cycles for {@code maxIter} rounds
 * (or until max absolute change &lt; {@code tol}).
 *
 * <p>Initial fill uses column means learned in {@link #fit}. Transform reuses those
 * means and re-runs the chained regression on the input (train-stat safe).
 */
public class IterativeImputer extends BaseTransformer {
    private static final long serialVersionUID = 2L;

    private int maxIter = 10;
    private double tol = 1e-3;
    private final Map<String, Double> columnMeans = new HashMap<>();
    /** Per-target OLS coefficients: intercept + coef for each other feature (ordered). */
    private final Map<String, double[]> models = new HashMap<>();
    private boolean replace = true;

    public IterativeImputer(String... columns) {
        this(10, columns);
    }

    public IterativeImputer(int maxIter, String... columns) {
        super(columns); // CRITICAL: was missing in old ctor
        this.maxIter = Math.max(1, maxIter);
    }

    public IterativeImputer setTol(double tol) {
        this.tol = tol;
        return this;
    }

    public IterativeImputer setMaxIter(int maxIter) {
        this.maxIter = Math.max(1, maxIter);
        return this;
    }

    public IterativeImputer setReplace(boolean replace) {
        this.replace = replace;
        return this;
    }

    @Override
    public BaseTransformer fit(DataFrame X) {
        if (columns == null || columns.isEmpty()) {
            this.columns = new ArrayList<>(FeatureMatrices.numericColumnNames(X));
        }
        columnMeans.clear();
        models.clear();

        String[] cols = columns.toArray(new String[0]);
        int n = X.rowCount();
        int d = cols.length;
        double[][] data = new double[n][d];
        boolean[][] missing = new boolean[n][d];

        for (int j = 0; j < d; j++) {
            Column c = X.column(cols[j]);
            double sum = 0;
            int cnt = 0;
            for (int i = 0; i < n; i++) {
                Object v = c.get(i);
                boolean miss = v == null || Double.isNaN(DataValues.asDouble(v));
                missing[i][j] = miss;
                if (!miss) {
                    double dv = DataValues.asDouble(v);
                    data[i][j] = dv;
                    sum += dv;
                    cnt++;
                }
            }
            double mean = cnt == 0 ? 0.0 : sum / cnt;
            columnMeans.put(cols[j], mean);
            for (int i = 0; i < n; i++) {
                if (missing[i][j]) data[i][j] = mean;
            }
        }

        // Chained equations on training matrix; learn final OLS models per feature
        double[][] work = DenseLinalg.copy(data);
        for (int iter = 0; iter < maxIter; iter++) {
            double maxDelta = 0;
            for (int t = 0; t < d; t++) {
                // Fit OLS: target t ~ other features, using currently filled rows
                // Use all rows (with current imputations) — classic MICE round
                double[][] design = new double[n][d]; // intercept + d-1 others, padded
                // Actually design width = 1 + (d-1) = d
                double[] y = new double[n];
                for (int i = 0; i < n; i++) {
                    design[i][0] = 1.0;
                    int p = 1;
                    for (int j = 0; j < d; j++) {
                        if (j == t) continue;
                        design[i][p++] = work[i][j];
                    }
                    y[i] = work[i][t];
                }
                // XtX, XtY
                int pdim = d; // 1 intercept + d-1
                double[][] XtX = new double[pdim][pdim];
                double[] XtY = new double[pdim];
                for (int i = 0; i < n; i++) {
                    for (int a = 0; a < pdim; a++) {
                        XtY[a] += design[i][a] * y[i];
                        for (int b = a; b < pdim; b++) {
                            XtX[a][b] += design[i][a] * design[i][b];
                        }
                    }
                }
                for (int a = 0; a < pdim; a++)
                    for (int b = a + 1; b < pdim; b++)
                        XtX[b][a] = XtX[a][b];
                // ridge for stability
                for (int a = 0; a < pdim; a++) XtX[a][a] += 1e-6;

                double[] beta = DenseLinalg.solve(XtX, XtY);
                models.put(cols[t], beta);

                // update missing only
                for (int i = 0; i < n; i++) {
                    if (!missing[i][t]) continue;
                    double pred = beta[0];
                    int p = 1;
                    for (int j = 0; j < d; j++) {
                        if (j == t) continue;
                        pred += beta[p++] * work[i][j];
                    }
                    maxDelta = Math.max(maxDelta, Math.abs(pred - work[i][t]));
                    work[i][t] = pred;
                }
            }
            if (maxDelta < tol) break;
        }

        fitted = true;
        return this;
    }

    @Override
    public DataFrame transform(DataFrame X) throws Exception {
        requireFitted();
        String[] cols = columns.toArray(new String[0]);
        int n = X.rowCount();
        int d = cols.length;
        double[][] work = new double[n][d];
        boolean[][] missing = new boolean[n][d];

        for (int j = 0; j < d; j++) {
            Column c = X.column(cols[j]);
            double mean = columnMeans.getOrDefault(cols[j], 0.0);
            for (int i = 0; i < n; i++) {
                Object v = c.get(i);
                boolean miss = v == null || Double.isNaN(DataValues.asDouble(v));
                missing[i][j] = miss;
                work[i][j] = miss ? mean : DataValues.asDouble(v);
            }
        }

        // iterate using learned models (and optionally refit lightly)
        for (int iter = 0; iter < maxIter; iter++) {
            double maxDelta = 0;
            for (int t = 0; t < d; t++) {
                double[] beta = models.get(cols[t]);
                if (beta == null) continue;
                for (int i = 0; i < n; i++) {
                    if (!missing[i][t]) continue;
                    double pred = beta[0];
                    int p = 1;
                    for (int j = 0; j < d; j++) {
                        if (j == t) continue;
                        pred += beta[p++] * work[i][j];
                    }
                    maxDelta = Math.max(maxDelta, Math.abs(pred - work[i][t]));
                    work[i][t] = pred;
                }
            }
            if (maxDelta < tol) break;
        }

        DataFrame result = X.copy();
        for (int j = 0; j < d; j++) {
            String outName = replace ? cols[j] : cols[j] + "_iterative_imputed";
            if (!replace) {
                if (result.hasColumn(outName)) result.removeColumn(outName);
                result.addColumn(outName, Column.DType.FLOAT64);
                Column oc = result.column(outName);
                while (oc.size() < n) oc.add(null);
            }
            Column dst = result.column(outName);
            for (int i = 0; i < n; i++) dst.set(i, work[i][j]);
        }
        return result;
    }

    public Map<String, Double> getColumnMeans() { return columnMeans; }
    public Map<String, double[]> getModels() { return models; }
}
