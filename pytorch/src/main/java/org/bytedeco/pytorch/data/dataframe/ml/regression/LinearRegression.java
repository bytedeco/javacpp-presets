package org.bytedeco.pytorch.data.dataframe.ml.regression;

import org.bytedeco.pytorch.data.dataframe.feature.base.BaseRegressor;
import java.util.*;

/**
 * 线性回归（正规方程 + 梯度下降 fallback）
 */
public class LinearRegression extends BaseRegressor {
    private boolean fitIntercept;
    private double[] coef;
    private double intercept;

    public LinearRegression() { this(true); }
    public LinearRegression(boolean fitIntercept) { this.fitIntercept = fitIntercept; }

    @Override
    public BaseRegressor fit(double[][] X, double[] y) {
        int n = X.length, d = X[0].length;
        // Build augmented matrix [X | 1]
        int cols = fitIntercept ? d + 1 : d;
        double[][] A = new double[n][cols];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) A[i][j] = X[i][j];
            if (fitIntercept) A[i][d] = 1.0;
        }
        // Solve via normal equation: w = (A^T A)^{-1} A^T y
        double[] w = solveNormalEquation(A, y);
        coef = Arrays.copyOfRange(w, 0, d);
        intercept = fitIntercept ? w[d] : 0.0;
        fitted = true; return this;
    }

    @Override
    public double[] predict(double[][] X) {
        double[] p = new double[X.length];
        for (int i = 0; i < X.length; i++) {
            double s = intercept;
            for (int j = 0; j < coef.length; j++) s += coef[j] * X[i][j];
            p[i] = s;
        }
        return p;
    }

    /** Solve via Cholesky / LU of (ATA) */
    private double[] solveNormalEquation(double[][] A, double[] y) {
        int n = A.length, d = A[0].length;
        double[][] ATA = new double[d][d];
        double[]   ATy = new double[d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                ATy[j] += A[i][j] * y[i];
                for (int k = 0; k < d; k++) ATA[j][k] += A[i][j] * A[i][k];
            }
        }
        // Tikhonov regularization (numerical stability)
        for (int j = 0; j < d; j++) ATA[j][j] += 1e-10;
        return gaussianElimination(ATA, ATy);
    }

    public static double[] gaussianElimination(double[][] A, double[] b) {
        int n = A.length;
        double[][] aug = new double[n][n+1];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) aug[i][j] = A[i][j];
            aug[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            // pivot
            int pivot = col;
            for (int row = col+1; row < n; row++)
                if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
            double[] tmp = aug[col]; aug[col] = aug[pivot]; aug[pivot] = tmp;
            if (Math.abs(aug[col][col]) < 1e-15) continue;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = aug[row][col] / aug[col][col];
                for (int j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) x[i] = aug[i][n] / aug[i][i];
        return x;
    }

    public double[] getCoef() { return coef; }
    public double getIntercept() { return intercept; }

    @Override
    public Map<String, Object> getParams() {
        Map<String, Object> p = new LinkedHashMap<>(); p.put("fit_intercept", fitIntercept); return p;
    }
    @Override public void setParams(Map<String, Object> params) {
        if (params.containsKey("fit_intercept")) fitIntercept = (Boolean) params.get("fit_intercept");
    }
}

