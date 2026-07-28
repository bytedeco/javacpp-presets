package org.bytedeco.pytorch.dataframe.ml.regression;

import org.bytedeco.pytorch.dataframe.feature.base.BaseRegressor;
import java.util.*;

/** 正交匹配追踪（稀疏线性回归） */
public class OrthogonalMatchingPursuit extends BaseRegressor {
    private int nNonzeroCoefs; // target sparsity
    private double[] coef; private double intercept;

    public OrthogonalMatchingPursuit() { this(10); }
    public OrthogonalMatchingPursuit(int nNonzeroCoefs) { this.nNonzeroCoefs = nNonzeroCoefs; }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        int k = Math.min(nNonzeroCoefs, d);
        coef = new double[d]; intercept = 0;
        // Center y
        double yMean = 0; for (double v : y) yMean += v; yMean /= n;
        double[] r = Arrays.copyOf(y, n); for (int i = 0; i < n; i++) r[i] -= yMean;

        Set<Integer> support = new LinkedHashSet<>();
        for (int iter = 0; iter < k; iter++) {
            // Find most correlated atom
            int best = -1; double bestCorr = -1;
            for (int j = 0; j < d; j++) {
                if (support.contains(j)) continue;
                double corr = 0; for (int i = 0; i < n; i++) corr += X[i][j] * r[i];
                if (Math.abs(corr) > bestCorr) { bestCorr = Math.abs(corr); best = j; }
            }
            if (best < 0) break;
            support.add(best);
            // Solve least squares on support
            int[] sup = support.stream().mapToInt(Integer::intValue).toArray();
            double[][] Xs = new double[n][sup.length];
            for (int i = 0; i < n; i++) for (int j = 0; j < sup.length; j++) Xs[i][j] = X[i][sup[j]];
            double[] ws = solveLS(Xs, r);
            // Update residual
            double[] pred = new double[n];
            for (int i = 0; i < n; i++) for (int j = 0; j < sup.length; j++) pred[i] += ws[j] * Xs[i][j];
            for (int i = 0; i < n; i++) r[i] = y[i] - yMean - pred[i];
            // Store coefficients
            for (int j = 0; j < sup.length; j++) coef[sup[j]] = ws[j];
        }
        intercept = yMean; for (int j = 0; j < d; j++) intercept -= coef[j] * colMean(X, j);
        fitted = true; return this;
    }

    private double[] solveLS(double[][] A, double[] b) {
        int n = A.length, d = A[0].length;
        double[][] ATA = new double[d][d]; double[] ATb = new double[d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                ATb[j] += A[i][j] * b[i];
                for (int k = 0; k < d; k++) ATA[j][k] += A[i][j] * A[i][k];
            }
        }
        for (int j = 0; j < d; j++) ATA[j][j] += 1e-10;
        return LinearRegression.gaussianElimination(ATA, ATb);
    }

    private double colMean(double[][] X, int j) {
        double s = 0; for (double[] row : X) s += row[j]; return s / X.length;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) { p[i] = intercept; for (int j = 0; j < coef.length; j++) p[i] += coef[j]*X[i][j]; }
        return p;
    }

    @Override
    public Map<String, Object> getParams() {
        return new LinkedHashMap<>(Map.of("n_nonzero_coefs", nNonzeroCoefs));
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("n_nonzero_coefs")) nNonzeroCoefs = ((Number) params.get("n_nonzero_coefs")).intValue();
    }
}

